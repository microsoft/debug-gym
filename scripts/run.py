import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from termcolor import colored
from rich.console import Group
from rich.panel import Panel
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from froggy.terminal import select_terminal
from froggy.tools.toolbox import Toolbox
from froggy.utils import load_config, setup_logger



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


def run_agent(args, problem, config, current_app_progress: Progress):
    # add progress bar for steps of this app, and run the steps
    current_task_id = current_app_progress.add_task(f"Running task {problem}")
    # app_steps_task_id = app_steps_progress.add_task("", total=len(step_times), name=problem)

    task_logger = setup_logger(
        problem, log_dir=config["output_path"], verbose=args.very_verbose, mode="w" if args.force_all else "a"
    )
    try:
        previous_run = Path(config["output_path"]) / config["uuid"] / problem / "froggy.jsonl"

        if not args.force_all and os.path.exists(previous_run):
            task_logger.debug(f"Previous run found: {previous_run}")
            with open(previous_run) as reader:
                success = json.load(reader)["success"]

            task_logger.debug(f"Previous run success: {success}")
            if not args.force_failed or success:
                task_logger.info(f"Skipping {problem}, already done.")
                current_app_progress.stop_task(current_task_id)
                current_app_progress.update(current_task_id, description=f"[bold gray]Task {problem} skipped!")
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
    except Exception as e:
        task_logger.warning(
            f"Task Error: {problem} - {e!r}. Run with --very-verbose or check {task_logger.log_file} for more information."
        )
        task_logger.debug(
            f"Task {problem} generated an exception: {e!r}", exc_info=True
        )
        if args.debug:
            breakpoint()
            raise e

        success = False

    # stop and hide steps progress bar for this specific app
    # app_steps_progress.update(app_steps_task_id, visible=False)
    current_app_progress.stop_task(current_task_id)
    current_app_progress.update(current_task_id, description=f"[bold green]Task {problem} completed!")
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
        verbose=args.verbose,
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

        num_workers = int(
            os.environ.get("FROGGY_WORKERS", 1)
        )  # 1 if args.verbose else int(os.environ.get("FROGGY_WORKERS", 1))
        tasks_done = 0
        mean_perf = 0

        # progress bar for current app showing only elapsed time,
        # which will stay visible when app is installed
        current_app_progress = Progress(
            TimeElapsedColumn(),
            TextColumn("{task.description}"),
        )

        # progress bars for single app steps (will be hidden when step is done)
        step_progress = Progress(
            TextColumn("  "),
            TimeElapsedColumn(),
            TextColumn("[bold purple]{task.fields[action]}"),
            SpinnerColumn("simpleDots"),
        )
        # progress bar for current app (progress in steps)
        app_steps_progress = Progress(
            TextColumn(
                "[bold blue]Progress for app {task.fields[name]}: {task.percentage:.0f}%"
            ),
            BarColumn(),
            TextColumn("({task.completed} of {task.total} steps done)"),
        )
        # overall progress bar
        overall_progress = Progress(
            TimeElapsedColumn(), BarColumn(), TextColumn("{task.description}")
        )
        # group of progress bars;
        # some are always visible, others will disappear when progress is complete
        progress_group = Group(
            Panel(Group(current_app_progress, step_progress, app_steps_progress), title="Workers"),
            overall_progress,
        )

        overall_task_id = overall_progress.add_task("", total=len(problem_list))
        top_descr = "[bold #AAAAAA](%d out of %d tasks done)" % (tasks_done, len(problem_list))
        overall_progress.update(overall_task_id, description=top_descr, advance=0)

        # use own live instance as context manager with group of progress bars,
        # which allows for running multiple different progress bars in parallel,
        # and dynamically showing/hiding them
        with Live(progress_group):
            with ThreadPoolExecutor(num_workers) as executor:
                futures = [
                    executor.submit(run_agent, args, problem, config, current_app_progress)
                    for problem in problem_list
                ]
                for future in as_completed(futures):
                    result = future.result()
                    mean_perf += result
                    tasks_done += 1

                    # update message on overall progress bar
                    top_descr = "[bold #AAAAAA](%d out of %d tasks done)" % (tasks_done, len(problem_list))
                    overall_progress.update(overall_task_id, description=top_descr, advance=1)

            # final update for message on overall progress bar
            overall_progress.update(
                overall_task_id, description=f"[bold green]{mean_perf}/{tasks_done} success!"
            )
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
