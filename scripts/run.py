import json
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rich.markup import escape

from debug_gym.agents.base_agent import AGENT_REGISTRY, create_agent
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
        task_logger.error(
            escape(
                f"Task Error: {problem} - {e!r}. Run with --very-verbose "
                "or check {task_logger.log_file} for more information."
            )
        )
        task_logger.debug(
            escape(f"Task {problem} generated an exception: {e!r}", exc_info=True)
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
        tool_instantiated = Toolbox.get_tool(tool)
        env.add_tool(tool_instantiated)
        logger.debug(f"Adding tool to toolbox: {tool_instantiated.__class__.__name__}")


def main():
    config, args = load_config()
    logger = DebugGymLogger("debug-gym", level=args.logging_level)

    config["uuid"] = config.get("uuid", str(uuid.uuid4()))
    logger.info(f"Experiment log path: {config['output_path']}/{config['uuid']}")

    # Figure out which problems to solve.
    problems = config.get("problems", ["custom"])
    if isinstance(problems, str) and "benchmark" in config:
        env = create_env(config, logger=logger)
        if problems == "all":
            problems = sorted(env.dataset.keys())  # all tasks
        else:
            problems = env.get_dataset_split(problems)

    if args.list:
        print(f"\n# Available problems in {config.get('benchmark', 'config')}:")
        for problem in problems:
            print(f" - {problem}")

        # list agent
        print("\n# Available agents:")
        for agent in AGENT_REGISTRY:
            print(f" - {agent}")

        return

    num_workers = args.num_workers or int(os.environ.get("DEBUG_GYM_WORKERS", 1))
    if args.debug:
        logger.warning("Running in debug mode, num_workers set to 1")
        num_workers = 1
    # make sure number of workers is in range [1, len(problems)]
    num_workers = min(max(1, num_workers), len(problems))
    logger.info(f"Running with {num_workers} workers")

    tasks_done = 0
    mean_perf = 0
    tasks_succeeded = []

    with logger.rich_progress(problems, max_display=20):
        if num_workers == 1:  # run sequentially for easier debugging
            for problem in problems:
                try:
                    success = run_agent(args, problem, config)
                    mean_perf += success
                    tasks_done += 1

                    if success:
                        tasks_succeeded.append(problem)

                    mean_perf_text = f"[green]{mean_perf}[/green]"
                    logger.info(f"Overall tasks done ({mean_perf_text} are successful)")
                except (KeyboardInterrupt, BreakTaskLoop) as e:
                    raise e
                except Exception as e:
                    raise e
        else:
            with ProcessPoolExecutor(
                num_workers,
                initializer=DebugGymLogger.set_as_worker,
            ) as executor:
                futures = {
                    executor.submit(run_agent, args, problem, config): problem
                    for problem in problems
                }
                for future in as_completed(futures):
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
                        mean_perf_text = f"[green]{mean_perf}[/green]"
                        logger.info(
                            f"Overall tasks done ({mean_perf_text} are successful)"
                        )
                    except (KeyboardInterrupt, BreakTaskLoop) as e:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise e
                    except Exception as e:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise e

    tasks_failed = list(set(problems) - set(tasks_succeeded))
    logger.info(f"[green]Tasks that succeeded:[/green] {tasks_succeeded}")
    logger.info(f"[red]Tasks that failed:[/red] {tasks_failed}")


if __name__ == "__main__":
    main()
