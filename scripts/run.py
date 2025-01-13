import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from os.path import join as pjoin

from rich.progress import BarColumn, Progress, TextColumn
from termcolor import colored
from tqdm import tqdm

from froggy.terminal import select_terminal
from froggy.tools.toolbox import Toolbox
from froggy.utils import TaskLogger, load_config

logger = logging.getLogger("froggy")


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


def run_agent_wrapper(payload):
    args, problem, config = payload
    return run_agent(args, problem, config)


def run_agent(args, problem, config):
    task_logger = TaskLogger(logger, {"task": problem})
    try:
        agent = create_agent(args, config, logger=task_logger)

        if os.path.exists(pjoin(agent._output_path, problem, "froggy.jsonl")):
            print(colored(f"Skipping {problem}, already done.", "yellow"))
            return

        done = agent.run(task_name=problem, debug=args.debug)

        # optionally apply patch
        if config["save_patch"]:
            agent.save_patch(task_name=problem)

        # save log
        agent.log(task_name=problem)
    except Exception as e:
        logger.warning(
            f"Task Error: {problem} - {e!r}. Run with --verbose for more information."
        )
        logger.debug(f"Task {problem} generated an exception: {e!r}", exc_info=True)
        raise e

    return done


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
    env = create_env(args, config, logger)

    agent_config = dict(
        config_dict=config,
        env=env,
        verbose=args.verbose,
        logger=logger,
    )

    # instantiate agent
    match args.agent:
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
    logger.setLevel(logging.DEBUG)
    config, args = load_config()
    if args.very_verbose:
        args.verbose = True
        logger.setLevel(logging.DEBUG)

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

        num_workers = 1 if args.verbose else int(os.environ.get("FROGGY_WORKERS", 1))
        tasks_done = 0
        mean_perf = 0

        # with Pool(num_workers) as pool:
        #     results = []
        #     jobs = ((args, problem, config) for problem in problem_list)
        #     for result in tqdm(pool.map(run_agent_wrapper, jobs), total=len(problem_list)):
        #         try:
        #             if result is not None:
        #                 mean_perf += result
        #                 tasks_done += 1

        #             # pbar.set_description_str(
        #             #     f"Avg. Score so far: {mean_perf / tasks_done:.2f}"
        #             # )
        #             #pbar.update()
        #         except Exception as e:
        #             logger.warning(f"Task Error: {e!r}. Run with --verbose for more information.")
        #             logger.debug(f"Task generated an exception: {e!r}", exc_info=True)
        #             if args.debug:
        #                 raise e

        pbar = tqdm(range(len(problem_list)))
        with ThreadPoolExecutor(num_workers) as executor:
            futures = [
                executor.submit(run_agent, args, problem, config)
                for problem in problem_list
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                except asyncio.CancelledError:
                    logger.warning("Task cancelled.")
                    break
                except KeyboardInterrupt:
                    logger.warning("Task interrupted by user.")
                    break
                except Exception as e:
                    logger.warning(
                        f"Task Error: {e!r}. Run with --verbose for more information."
                    )
                    logger.debug(f"Task generated an exception: {e!r}", exc_info=True)
                    if args.debug:
                        raise e

                    continue  # Skip to the next future if desired

                mean_perf += result
                tasks_done += 1

                pbar.set_description_str(
                    f"Avg. Score so far: {mean_perf / tasks_done:.2f}"
                )
                pbar.update()
    else:
        # custom repo
        print(colored(f"Running agent {agent.name}", "green"))
        agent = create_agent(args, config)
        agent.run(debug=args.debug)
        # optionally apply patch
        if config["save_patch"]:
            agent.save_patch()
        # save log
        agent.log()


if __name__ == "__main__":
    main()
