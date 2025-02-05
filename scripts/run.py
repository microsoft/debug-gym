import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.live import Live
from termcolor import colored

from froggy.envs import select_env
from froggy.logger import FroggyLogger
from froggy.terminal import select_terminal
from froggy.tools.toolbox import Toolbox
from froggy.utils import load_config


class BreakTaskLoop(Exception):
    pass


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
        case "pdb_agent":
            from agent import PdbAgent as agent_class
        case "rewrite_only":
            from agent import RewriteOnly as agent_class
        case "pdb_after_rewrites":
            from agent import PdbAfterRewrites as agent_class
        case _:
            raise ValueError(f"Unknown agent {agent_type}")

    agent = agent_class(**kwargs)

    return agent


def main():
    config, args = load_config()
    logger = FroggyLogger("froggy", level=args.logging_level, is_task=False)

    available_agents = list(config.keys())
    assert (
        args.agent in available_agents
    ), f"Invalid agent. Available agents: {available_agents}"

    config = config[args.agent]

    with Live(logger.progress_group, refresh_per_second=20) as live:
        if args.debug:
            live.stop()  # Because it interferes with pdb.
        # run agent, loop over the tasks
        if "benchmark" in config and "problems" in config:
            if "all" == config["problems"]:
                env = create_env(args, config, logger=logger)
                problem_list = env.dataset.keys()  # all tasks
            else:
                assert isinstance(config["problems"], list)
                problem_list = config["problems"]

            num_workers = int(os.environ.get("FROGGY_WORKERS", 1))
            tasks_done = 0
            mean_perf = 0

            tasks_succeeded = []

            overall_task_id = logger.overall_progress.add_task(
                "", total=len(problem_list)
            )
            top_descr = "[bold #AAAAAA](%d out of %d tasks done)" % (
                tasks_done,
                len(problem_list),
            )
            logger.overall_progress.update(
                overall_task_id, description=top_descr, advance=0
            )

            with ThreadPoolExecutor(num_workers) as executor:
                futures = {
                    executor.submit(run_agent, args, problem, config): problem
                    for problem in problem_list
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
                        top_descr = (
                            f"[bold #AAAAAA]({tasks_done} out of {len(problem_list)} tasks "
                            f"done - [bold green]{mean_perf}[bold #AAAAAA] are successful)"
                        )
                        logger.overall_progress.update(
                            overall_task_id, description=top_descr, advance=1
                        )
                    except (KeyboardInterrupt, BreakTaskLoop) as e:
                        live.stop()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise e
                    except Exception as e:
                        live.stop()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise e

            # final update for message on overall progress bar
            logger.overall_progress.update(
                overall_task_id,
                description=f"[bold green]{mean_perf}/{tasks_done} success!",
            )

            logger.info(f"Tasks that succeeded: {tasks_succeeded}")
        else:
            # custom repo
            problem = "custom"
            config["uuid"] = config.get("uuid", str(uuid.uuid4()))
            exp_path = Path(config["output_path"]) / config["uuid"] / problem
            task_logger = FroggyLogger(
                problem,
                is_task=True,
                log_dir=exp_path,
                level=args.logging_level,
                mode="w" if args.force_all else "a",
            )

            env = create_env(args, config, logger=task_logger)
            agent = create_agent(args.agent, config=config, env=env, logger=task_logger)
            print(colored(f"Running agent {agent.name}", "green"))
            agent.run(debug=args.debug)

            # optionally apply patch
            if config["save_patch"]:
                agent.save_patch()

            # save log
            agent.log()


if __name__ == "__main__":
    main()
