import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.live import Live
from termcolor import colored

from froggy.logger import FroggyLogger
from froggy.terminal import select_terminal
from froggy.tools.toolbox import Toolbox
from froggy.utils import load_config


class BreakTaskLoop(Exception):
    pass


def select_env(env_type: str = None):
    match env_type:
        case None:
            from froggy.envs.env import RepoEnv as env_class
        case "aider":
            from froggy.envs import AiderBenchmarkEnv as env_class
        case "swebench":
            from froggy.envs import SWEBenchEnv as env_class
        case "terminal_simulator":
            from froggy.envs import TerminalSimulatorEnv as env_class
        case _:
            raise ValueError(f"Unknown benchmark {env_type}")
    return env_class


def run_agent(args, problem, config):
    config["uuid"] = config.get("uuid", str(uuid.uuid4()))
    exp_path = Path(config["output_path"]) / config["uuid"] / problem

    task_logger = FroggyLogger(
        problem,
        is_task=True,
        log_dir=exp_path,
        level=args.logging_level,
        mode="w" if args.force_all else "a",
    )
    try:
        previous_run = exp_path / "froggy.jsonl"
        if not args.force_all and os.path.exists(previous_run):
            task_logger.debug(f"Previous run found: {previous_run}")
            with open(previous_run) as reader:
                success = json.load(reader)["success"]

            task_logger.debug(f"Previous run success: {success}")
            if not args.force_failed or success:
                task_logger.info("[bold gray]Skipped, already done.")
                task_logger.stop(remove=not args.keep_completed_tasks)
                return success

        env = create_env(args, config, task_logger)
        agent = create_agent(args.agent, config=config, env=env, logger=task_logger)
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

    task_logger.info("[bold green]Completed!")
    task_logger.stop(remove=not args.keep_completed_tasks)
    return success


def create_env(args, config: dict, logger: FroggyLogger):
    terminal = select_terminal(config.get("terminal"), logger)
    env_class = select_env(config.get("benchmark"))
    env = env_class(**config["env_kwargs"], terminal=terminal, logger=logger)

    # import tools to the environment
    for tool in config["tools"]:
        kwargs = {}

        if tool == "pdb":
            kwargs["persistent_breakpoints"] = config["persistent_breakpoints"]

        tool_instantiated = Toolbox.get_tool(tool, **kwargs)
        env.add_tool(tool_instantiated)
        logger.debug(f"Adding tool to toolbox: {tool_instantiated.__class__.__name__}")

    return env


def create_agent(agent_type, **kwargs):
    match agent_type:
        case "zero_shot":
            from froggy.agents import AgentZeroShot as agent_class
        case "cot":
            from froggy.agents import AgentCoT as agent_class
        case "tadpole":
            from froggy.agents import AgentTadpole as agent_class
        case "zero_shot_nopdb":
            from froggy.agents import AgentZeroShot_NoPDB as agent_class
        case "cot_nopdb":
            from froggy.agents import AgentCoT_NoPDB as agent_class
        case "zero_shot_pdb_after_rewrites":
            from froggy.agents import AgentZeroShot_PdbAfterRewrites as agent_class
        case "zero_shot_nopdb_whole":
            from froggy.agents import AgentZeroShot_NoPDB as agent_class
        case _:
            raise ValueError(f"Unknown agent {agent_type}")

    agent = agent_class(**kwargs)

    return agent


def main():
    config, args = load_config()
    logger = FroggyLogger("froggy", level=args.logging_level, is_task=False)

    available_agents = list(config.keys())
    if args.agent not in available_agents:
        raise ValueError(
            f"Invalid agent: {args.agent}. Available agents: {available_agents}"
        )

    config = config[args.agent]

    # Figure out which problems to solve.
    problems = config.get("problems", ["custom"])
    if problems == "all" and "benchmark" in config:
        env = create_env(args, config, logger=logger)
        problems = list(env.dataset.keys())  # all tasks

    with Live(logger.progress_group, refresh_per_second=20) as live:
        num_workers = int(os.environ.get("FROGGY_WORKERS", 0))
        if args.debug:
            num_workers = 0
            live.stop()  # Because it interferes with pdb.

        tasks_done = 0
        mean_perf = 0

        tasks_succeeded = []
        overall_task_id = logger.overall_progress.add_task(
            description=f"[bold #AAAAAA]({tasks_done} out of {len(problems)} tasks done)",
            total=len(problems),
        )

        if num_workers > 1:
            # Multi-thread
            with ThreadPoolExecutor(num_workers) as executor:
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
                        logger.overall_progress.update(
                            overall_task_id,
                            description=(
                                f"[bold #AAAAAA]({tasks_done} out of {len(problems)} tasks "
                                f"done - [bold green]{mean_perf}[bold #AAAAAA] are successful)"
                            ),
                            advance=1,
                        )
                    except (KeyboardInterrupt, BreakTaskLoop) as e:
                        live.stop()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise e
                    except Exception as e:
                        live.stop()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise e

        else:
            # Single thread
            for problem in problems:
                try:
                    success = run_agent(args, problem, config)
                    mean_perf += success
                    tasks_done += 1

                    if success:
                        tasks_succeeded.append(problem)

                    # update message on overall progress bar
                    logger.overall_progress.update(
                        overall_task_id,
                        description=(
                            f"[bold #AAAAAA]({tasks_done} out of {len(problems)} tasks "
                            f"done - [bold green]{mean_perf}[bold #AAAAAA] are successful)"
                        ),
                        advance=1,
                    )
                except (KeyboardInterrupt, BreakTaskLoop) as e:
                    live.stop()
                    raise e
                except Exception as e:
                    live.stop()
                    raise e

        # final update for message on overall progress bar
        logger.overall_progress.update(
            overall_task_id,
            description=f"[bold green]{mean_perf}/{tasks_done} success!",
        )

        logger.info(f"Tasks that succeeded: {tasks_succeeded}")
        logger.info(f"Tasks that failed: {set(problems) - set(tasks_succeeded)}")


if __name__ == "__main__":
    main()
