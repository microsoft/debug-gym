import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from termcolor import colored
from tqdm import tqdm

from debug_gym.agents.base_agent import create_agent
from debug_gym.agents.utils import load_config
from debug_gym.gym.envs import select_env
from debug_gym.gym.terminal import select_terminal
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.logger import DebugGymLogger


class BreakTaskLoop(Exception):
    pass


def run_agent(args, problem, config):
    exp_path = Path(config["output_path"]) / config["uuid"] / problem

    task_logger = DebugGymLogger(
        problem,
        log_dir=exp_path,
        level=args.logging_level,
        mode="w" if args.force_all else "a",
    )
    env = None
    try:
        previous_run = exp_path / "debug_gym.jsonl"
        if not args.force_all and os.path.exists(previous_run):
            task_logger.debug(f"Previous run found: {previous_run}")
            with open(previous_run) as reader:
                success = json.load(reader)["success"]

            task_logger.debug(f"Previous run success: {success}")
            if not args.force_failed or success:
                task_logger.info("Skipped, already done.")
                return success

        env = create_env(config, task_logger)
        add_tools(env, config, task_logger)
        agent = create_agent(
            config["agent_type"],
            config=config,
            env=env,
            logger=task_logger,
        )
        success = agent.run(task_name=problem, debug=args.debug)

        # optionally apply patch
        if config["save_patch"]:
            agent.save_patch(task_name=problem)

        # save log
        agent.log(task_name=problem)
    except KeyboardInterrupt:
        raise BreakTaskLoop

    except Exception as e:
        task_logger.warning(
            f"Task Error: {problem} - {e!r}. Run with --very-verbose or check {task_logger.log_file} for more information."
        )
        task_logger.debug(
            f"Task {problem} generated an exception: {e!r}", exc_info=True
        )
        if args.debug:
            raise e

        success = False
    finally:
        if env:
            env.close()

    task_logger.info(f"Completed, log saved at: {task_logger.log_file}")
    return success


def create_env(config: dict, logger: DebugGymLogger):
    terminal = select_terminal(config.get("terminal"), logger)
    env_class = select_env(config.get("benchmark"))
    env = env_class(**config["env_kwargs"], terminal=terminal, logger=logger)
    return env


def add_tools(env, config: dict, logger: DebugGymLogger):
    """Add tools to the environment"""
    for tool in config["tools"]:
        kwargs = {}
        if tool == "pdb":
            kwargs["persistent_breakpoints"] = config["persistent_breakpoints"]
            kwargs["auto_list"] = config["auto_list"]

        tool_instantiated = Toolbox.get_tool(tool, **kwargs)
        env.add_tool(tool_instantiated)
        logger.debug(f"Adding tool to toolbox: {tool_instantiated.__class__.__name__}")


def main():
    config, args = load_config()
    logger = DebugGymLogger("debug-gym", level=args.logging_level)

    config["uuid"] = config.get("uuid", str(uuid.uuid4()))
    logger.warning(f"Experiment log path: {config['output_path']}/{config['uuid']}")

    # Figure out which problems to solve.
    problems = config.get("problems", ["custom"])
    if problems == "all" and "benchmark" in config:
        env = create_env(config, logger=logger)
        problems = list(env.dataset.keys())  # all tasks

    num_workers = int(os.environ.get("DEBUG_GYM_WORKERS", 1))
    logger.warning(f"Running with {num_workers} workers")
    if args.debug:
        num_workers = 1

    tasks_done = 0
    mean_perf = 0
    tasks_succeeded = []

    if num_workers > 1:
        # Multi-thread
        with ThreadPoolExecutor(num_workers) as executor:
            futures = {
                executor.submit(run_agent, args, problem, config): problem
                for problem in problems
            }
            mean_perf_text = colored(f"{mean_perf}", "green")
            desc = f"Overall progress ({mean_perf_text} are successful)"
            pbar = tqdm(as_completed(futures), desc=desc, total=len(problems))
            for future in pbar:
                if future.cancelled():
                    continue

                try:
                    problem = futures[future]
                    success = future.result()
                    mean_perf += success
                    tasks_done += 1

                    if success:
                        tasks_succeeded.append(problem)

                    # update message on overall progress bar
                    mean_perf_text = colored(f"{mean_perf}", "green")
                    pbar.set_description(
                        f"Overall tasks done ({mean_perf_text} are successful)"
                    )
                except (KeyboardInterrupt, BreakTaskLoop) as e:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise e
                except Exception as e:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise e

    else:
        # Single thread
        mean_perf_text = colored(f"{mean_perf}", "green")
        desc = f"Overall tasks done ({mean_perf_text} are successful)"
        pbar = tqdm(problems, desc=desc, total=len(problems))
        for problem in pbar:
            try:
                success = run_agent(args, problem, config)
                mean_perf += success
                tasks_done += 1

                if success:
                    tasks_succeeded.append(problem)

                # update message on overall progress bar
                mean_perf_text = colored(f"{mean_perf}", "green")
                pbar.set_description(
                    f"Overall tasks done ({mean_perf_text} are successful)"
                )
            except (KeyboardInterrupt, BreakTaskLoop) as e:
                raise e
            except Exception as e:
                raise e

        logger.info(f"Tasks that succeeded: {tasks_succeeded}")
        logger.info(f"Tasks that failed: {set(problems) - set(tasks_succeeded)}")


if __name__ == "__main__":
    main()
