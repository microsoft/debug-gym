import os
import signal
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from debug_gym.agents.base_agent import AGENT_REGISTRY, create_agent
from debug_gym.agents.utils import load_config, save_patch, save_trajectory
from debug_gym.experiment import create_env, dump_experiment_info
from debug_gym.gym.envs import load_dataset
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
        llm = LLM.instantiate(**config.get("llm", {}), logger=task_logger)
        agent = create_agent(config.get("agent", {}), logger=task_logger)

        try:
            success = agent.run(env, llm, debug=args.debug)
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
        save_trajectory(agent, task_path, task_logger)

        # optionally apply patch
        if config.get("save_patch", True):
            try:
                save_patch(env, task_path, task_logger)
            except Exception as patch_error:
                # Terminal may be unavailable (e.g., pod died), log and continue
                task_logger.warning(f"Could not save patch: {patch_error!r}")

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


def main():
    config, args = load_config()
    config["uuid"] = config.get("uuid", str(uuid.uuid4()))
    config["output_path"] = config.get("output_path", "exps")
    exp_output_path = Path(config["output_path"]) / config["uuid"]
    exp_output_path.mkdir(parents=True, exist_ok=True)
    logger = DebugGymLogger("debug-gym", level=args.logging_level)
    logger.debug(f"Experiment config: {config}")
    logger.info(f"Experiment log path: {exp_output_path}")
    dump_experiment_info(config, args)

    # Load the dataset based on the information found in the config.
    if config.get("task_data") is not None:
        dataset = {f"custom-task": config["task_data"]}
    else:
        dataset = load_dataset(config["dataset"], logger=logger)

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

    # Try to instantiate the LLM once to catch configuration errors early.
    llm = LLM.instantiate(**config.get("llm", {}), logger=logger)

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
                    run_agent(args, problem, dataset[problem], config)
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
                        future.result()
                    except AgentTimeoutException:
                        pass  # Handled in run_agent, just continue
                    except (KeyboardInterrupt, Exception) as e:
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise e


if __name__ == "__main__":
    main()
