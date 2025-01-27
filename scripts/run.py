import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Column
from termcolor import colored

from froggy.terminal import select_terminal
from froggy.tools.toolbox import Toolbox
from froggy.utils import load_config, setup_logger


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


def run_agent(args, problem, config, current_app_progress: Progress, live):
    exp_path = Path(config["output_path"]) / config["uuid"] / problem
    # add progress bar for steps of this app, and run the steps
    current_task_id = current_app_progress.add_task(
        f"\\[{problem}]:", log="Starting task..."
    )

    task_logger = setup_logger(
        problem,
        log_dir=exp_path,
        verbose=args.very_verbose,
        mode="w" if args.force_all else "a",
        progress=current_app_progress,
        task_id=current_task_id,
    )
    task_logger.live = live
    try:
        previous_run = exp_path / "froggy.jsonl"
        if not args.force_all and os.path.exists(previous_run):
            task_logger.debug(f"Previous run found: {previous_run}")
            with open(previous_run) as reader:
                success = json.load(reader)["success"]

            task_logger.debug(f"Previous run success: {success}")
            if not args.force_failed or success:
                task_logger.info(f"Skipping {problem}, already done.")
                current_app_progress.stop_task(current_task_id)
                current_app_progress.update(
                    current_task_id,
                    completed=True,
                    description=f"[bold gray]\\[{problem}]:",
                    log="[bold gray]Skipped!",
                )
                current_app_progress.remove_task(current_task_id)
                return success

        agent = create_agent(args, config, logger=task_logger)
        agent.env = create_env(args, config, task_logger)
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

    current_app_progress.stop_task(current_task_id)
    current_app_progress.update(
        current_task_id,
        completed=True,
        description=f"[bold green]\\[{problem}]:",
        log="[bold green]Completed!",
    )
    current_app_progress.remove_task(current_task_id)
    return success


def create_env(args, config, logger):
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


def create_agent(args, config, logger):
    agent_config = dict(
        config_dict=config,
        env=None,  # Will be set once after we determine if we skip the task.
        logger=logger,
    )

    # instantiate agent
    match args.agent:
        case "solution":
            from froggy.agents import AgentSolution as agent_class
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
            raise ValueError(f"Unknown agent {args.agent}")

    agent = agent_class(**agent_config)

    if args.verbose:
        agent.llm.verbose = True

    return agent


def main():
    config, args = load_config()
    if args.very_verbose:
        args.verbose = True

    logger = setup_logger("froggy", verbose=args.very_verbose)

    available_agents = list(config.keys())
    assert (
        args.agent in available_agents
    ), f"Invalid agent. Available agents: {available_agents}"

    config = config[args.agent]

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

        # progress bar for current task(s)
        task_progress = Progress(
            TimeElapsedColumn(),
            TextColumn("{task.description}"),
            TextColumn(
                "{task.fields[log]}"
            ),  # , table_column=Column(no_wrap=True, width=80)),
        )

        tasks_succeeded = []
        overall_progress = Progress(
            TextColumn("🐸"),
            TimeElapsedColumn(),
            BarColumn(),
            TextColumn("{task.description}"),
        )

        progress_group = Group(
            Panel(task_progress, title="Workers"),
            overall_progress,
        )

        overall_task_id = overall_progress.add_task("", total=len(problem_list))
        top_descr = "[bold #AAAAAA](%d out of %d tasks done)" % (
            tasks_done,
            len(problem_list),
        )
        overall_progress.update(overall_task_id, description=top_descr, advance=0)

        # from rich.console import Console
        # from rich.layout import Layout
        # Console(st)
        # layout = Layout()
        # layout.split_row(
        #     Layout(name="top"),
        #     Layout(progress_group, name="bottom"),
        # )

        # use own live instance as context manager with group of progress bars,
        # which allows for running multiple different progress bars in parallel,
        # and dynamically showing/hiding them
        import sys

        # Redirect stdout and stderr to avoid printing progress bars to stdout
        # sys.stdout = open(os.devnull, "w")
        # sys.stderr = open(os.devnull, "w")

        with Live(progress_group) as live:
            if args.debug:
                live.stop()

            with ThreadPoolExecutor(num_workers) as executor:
                futures = {
                    executor.submit(
                        run_agent, args, problem, config, task_progress, live
                    ): problem
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
                        overall_progress.update(
                            overall_task_id, description=top_descr, advance=1
                        )
                    except (KeyboardInterrupt, BreakTaskLoop):
                        live.stop()
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception as e:
                        live.stop()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise e

            # final update for message on overall progress bar
            overall_progress.update(
                overall_task_id,
                description=f"[bold green]{mean_perf}/{tasks_done} success!",
            )

        logger.info(f"Tasks that succeeded: {tasks_succeeded}")
    else:
        # custom repo
        print(colored(f"Running agent {agent.name}", "green"))
        agent = create_agent(args, config)
        agent.env = create_env(args, config)
        agent.run(debug=args.debug)

        # optionally apply patch
        if config["save_patch"]:
            agent.save_patch()

        # save log
        agent.log()


if __name__ == "__main__":
    main()
