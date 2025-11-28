import datetime
import json
import os
import signal
import subprocess
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from debug_gym import version as dg_version
from debug_gym.agents.base_agent import AGENT_REGISTRY, AgentArgs, create_agent
from debug_gym.agents.utils import load_config, save_patch, save_trajectory
from debug_gym.gym.envs import select_env
from debug_gym.gym.envs.r2egym import load_r2egym_dataset
from debug_gym.gym.envs.swe_bench import load_swebench_dataset
from debug_gym.gym.envs.swe_smith import load_swesmith_dataset
from debug_gym.gym.terminals import select_terminal
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.llms.base import LLM
from debug_gym.llms.human import Human
from debug_gym.logger import DebugGymLogger, load_previous_run_status


class AgentTimeoutException(BaseException):
    """Custom exception to handle timeouts in agent
    execution. Inherits from BaseException to ensure
    it is not caught by agent exception handling."""

    pass


def set_signal(timeout_seconds):
    """Set a signal handler for timeouts.
    Only works on Unix-like systems."""

    def timeout_handler(signum, frame):
        """Signal handler for timeout."""
        raise AgentTimeoutException

    if timeout_seconds > 0:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)


def run_agent(args, task_name: str, task_data: dict, config: dict):
    set_signal(args.timeout)
    success = True
    env = None

    # Flag to not report errors from the agent, since they report
    # errors themselves and we want to avoid double reporting.
    report_progress_error = True

    exp_path = Path(config["output_path"]) / config["uuid"]
    task_path = exp_path / task_name

    task_logger = DebugGymLogger(
        task_name,
        log_dir=task_path,
        level=args.logging_level,
        mode="w" if args.force_all else "a",
    )
    try:
        previous_run = load_previous_run_status(task_path, task_name)
        if (
            not args.force_all
            and previous_run is not None
            and previous_run.status in ["resolved", "unresolved"]
        ):
            task_logger.debug(f"Previous run found: {task_path}")
            success = previous_run.status == "resolved"
            task_logger.debug(f"Previous run status: {previous_run.status}")
            if not args.force_failed or success:
                status = "skip-resolved" if success else "skip-unresolved"
                task_logger.report_progress(
                    problem_id=previous_run.problem_id,
                    step=previous_run.step,
                    total_steps=previous_run.total_steps,
                    score=previous_run.score,
                    max_score=previous_run.max_score,
                    status=status,
                )
                task_logger.debug(f"Skipping {task_name}, already done.")
                return success

        task_logger.report_progress(
            problem_id=task_name,
            step=0,
            total_steps=1,
            score=0,
            max_score=None,
            status="running",
        )

        env = create_env(config, task_data, task_logger)
        add_tools(env, config, task_logger)

        llm = LLM.instantiate(
            llm_name=config["llm_name"],
            llm_config_file_path=config.get("llm_config_file_path"),
            logger=task_logger,
        )

        agent_args = AgentArgs.from_dict(config)
        agent = create_agent(
            config["agent_type"],
            agent_args=agent_args,
            llm=llm,
            logger=task_logger,
        )

        try:
            success = agent.run(env, debug=args.debug)
        except KeyboardInterrupt:
            task_logger.error("Agent run was interrupted by user.")
            task_logger.report_progress(
                problem_id=task_name,
                step=1,
                total_steps=1,
                score=0,
                max_score=None,
                status="error",
            )
            success = False
            raise
        except AgentTimeoutException:
            task_logger.error(
                f"Timeout: Problem `{task_name}` exceeded "
                f"the time limit of {args.timeout} seconds."
            )
            task_logger.report_progress(
                problem_id=task_name,
                step=1,
                total_steps=1,
                score=0,
                max_score=None,
                status="error",
            )
            success = False
            raise
        except:
            report_progress_error = False
            raise

        # save trajectory
        save_trajectory(agent, task_name, task_path, task_logger)

        # optionally apply patch
        if config["save_patch"]:
            save_patch(env, task_path, task_logger)

    except Exception as e:
        task_logger.error(
            f"Task Error: {task_name} - {e!r}. Run with --very-verbose "
            f"or check {task_logger.log_file} for more information."
        )
        task_logger.debug(
            f"Task {task_name} generated an exception: {e!r}. Traceback: {traceback.format_exc()}"
        )
        if report_progress_error:
            task_logger.report_progress(
                problem_id=task_name,
                step=1,
                total_steps=1,
                score=0,
                max_score=None,
                status="error",
            )
        if args.debug:
            raise e

        success = False
    finally:
        # Close env and cancel any pending alarm
        signal.alarm(0)
        if env:
            env.close()
    return success


def create_env(config: dict, task_data: dict, logger: DebugGymLogger):
    terminal = select_terminal(config.get("terminal"), logger, uuid=config["uuid"])
    env_class = select_env(config.get("benchmark"))
    env = env_class(
        task_data=task_data,
        terminal=terminal,
        logger=logger,
        **config.get("env", {}),
    )
    return env


def add_tools(env, config: dict, logger: DebugGymLogger):
    """Add tools to the environment"""
    for tool in config["tools"]:
        tool_config = {}
        if isinstance(tool, dict):
            assert len(tool) == 1, "Tool dict must have exactly one key"
            tool, tool_config = list(tool.items())[0]

        tool_instantiated = Toolbox.get_tool(tool, **tool_config)
        env.add_tool(tool_instantiated)
        logger.debug(f"Adding tool to toolbox: {tool_instantiated.__class__.__name__}")


def dump_experiment_info(config: dict, args: dict):
    """Dump experiment information to a JSONL file.
    Each line is one experiment run with its metadata."""

    try:  # Get git commit hash
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = ""

    try:  # Get git diff
        git_diff = subprocess.check_output(
            ["git", "diff"], cwd=os.path.dirname(__file__)
        ).decode()
    except Exception:
        git_diff = ""

    version_info = {
        "debug_gym_version": dg_version.__version__,
        "datetime": datetime.datetime.now().isoformat(),
        "git_hash": git_hash,
        "git_diff": git_diff,
        "config": config,
        "args": vars(args),
        "python_version": os.sys.version,
    }

    file = Path(config["output_path"]) / config["uuid"] / "experiment_info.jsonl"
    with open(file, "a") as f:
        f.write(f"{json.dumps(version_info)}\n")


def main():
    config, args = load_config()
    config["uuid"] = config.get("uuid", str(uuid.uuid4()))
    exp_output_path = Path(config["output_path"]) / config["uuid"]
    exp_output_path.mkdir(parents=True, exist_ok=True)
    logger = DebugGymLogger("debug-gym", level=args.logging_level)
    logger.info(f"Experiment log path: {exp_output_path}")
    dump_experiment_info(config, args)

    # Create the environment to get the list of problems to run.
    dataset_info = {
        "dataset_id": config.get("env", {}).get("dataset_id"),
        "dataset_revision": config.get("env", {}).get("dataset_revision"),
        "problems": config.get("problems", "all"),
        "prepull_images": config.get("env", {}).get("prepull_images", False),
    }
    dataset = select_env(config.get("benchmark")).load_dataset(**dataset_info)
    problems = sorted(dataset)

    if args.list:
        print(f"\n# Available problems in {config.get('benchmark', 'config')}:")
        for problem in problems:
            print(f" - {problem}")

        # list agent
        print("\n# Available agents:")
        for agent in AGENT_REGISTRY:
            print(f" - {agent}")

        return

    llm = LLM.instantiate(
        llm_name=config["llm_name"],
        llm_config_file_path=config.get("llm_config_file_path"),
        logger=logger,
    )

    # Stop live progress display if --no-live-display is set
    # or in Human mode (avoid conflicts with prompt_toolkit)
    if args.no_live_display or isinstance(llm, Human) or args.debug:
        logger.set_no_live()

    num_workers = args.num_workers or int(os.environ.get("DEBUG_GYM_WORKERS", 1))
    if args.debug:
        logger.warning("Running in debug mode, num_workers set to 1")
        num_workers = 1
    # make sure number of workers is in range [1, len(problems)]
    num_workers = min(max(1, num_workers), len(problems))
    logger.info(f"Running with {num_workers} workers")

    with logger.rich_progress(problems, max_display=args.max_display):
        if num_workers == 1:  # run sequentially for easier debugging
            for problem in problems:
                try:
                    success = run_agent(args, problem, dataset[problem], config)
                except AgentTimeoutException:
                    pass  # Handled in run_agent, just continue
                except (KeyboardInterrupt, Exception) as e:
                    raise e
        else:
            with ProcessPoolExecutor(
                num_workers, initializer=DebugGymLogger.set_as_worker
            ) as executor:
                futures = {
                    executor.submit(
                        run_agent, args, problem, dataset[problem], config
                    ): problem
                    for problem in problems
                }
                for future in as_completed(futures):
                    if future.cancelled():
                        continue
                    try:
                        problem = futures[future]
                        success = future.result()
                    except AgentTimeoutException:
                        pass  # Handled in run_agent, just continue
                    except (KeyboardInterrupt, Exception) as e:
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise e


if __name__ == "__main__":
    main()
