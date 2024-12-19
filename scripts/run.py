import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join as pjoin

from rich.progress import BarColumn, Progress, TextColumn
from termcolor import colored
from tqdm import tqdm

from froggy.tools.toolbox import Toolbox
from froggy.utils import load_config

logger = logging.getLogger("froggy")


def select_terminal(terminal_config: None):
    terminal_config = terminal_config or {"type": "local"}
    terminal_type = terminal_config["type"]
    match terminal_type:
        case "docker":
            from froggy.terminal import DockerTerminal as terminal_class
        case "local":
            from froggy.terminal import Terminal as terminal_class
        case _:
            raise ValueError(f"Unknown terminal {terminal_type}")

    return terminal_class(**terminal_config)


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
    agent = create_agent(args, config)

    if os.path.exists(pjoin(agent._output_path, problem, "froggy.jsonl")):
        print(colored(f"Skipping {problem}, already done.", "yellow"))
        return

    done = agent.run(task_name=problem, debug=args.debug)

    # optionally apply patch
    if config["save_patch"]:
        agent.save_patch(task_name=problem)

    # save log
    agent.log(task_name=problem)
    return done


def create_env(args, config):
    terminal = select_terminal(config.get("terminal"))
    env_class = select_env(config.get("benchmark"))
    env = env_class(**config["env_kwargs"], terminal=terminal)

    # import tools to the environment
    for tool in config["tools"]:
        kwargs = {}

        if tool == "pdb":
            kwargs["persistent_breakpoints"] = config["persistent_breakpoints"]

        tool_instantiated = Toolbox.get_tool(tool, **kwargs)
        env.add_tool(tool_instantiated)
        logger.debug(f"Adding tool to toolbox: {tool_instantiated.__class__.__name__}")
    return env


def create_agent(args, config):
    env = create_env(args, config)

    # instantiate agent
    match args.agent:
        case "zero_shot":
            from froggy.agents import AgentZeroShot
            agent = AgentZeroShot(config, env, verbose=args.verbose, _uuid=args.uuid)
        case "cot":
            from froggy.agents import AgentCoT

            agent = AgentCoT(config, env, verbose=args.verbose)
        case "tadpole":
            from froggy.agents import AgentTadpole

            agent = AgentTadpole(config, env, verbose=args.verbose)
        case "zero_shot_nopdb":
            from froggy.agents import AgentZeroShot_NoPDB

            agent = AgentZeroShot_NoPDB(config, env, verbose=args.verbose)
        case "cot_nopdb":
            from froggy.agents import AgentCoT_NoPDB

            agent = AgentCoT_NoPDB(config, env, verbose=args.verbose)
        case "zero_shot_pdb_after_rewrites":
            from froggy.agents import AgentZeroShot_PdbAfterRewrites

            agent = AgentZeroShot_PdbAfterRewrites(config, env, verbose=args.verbose)
        case "zero_shot_nopdb_whole":
            from froggy.agents import AgentZeroShot_NoPDB

            agent = AgentZeroShot_NoPDB(config, env, verbose=args.verbose)
        case _:
            raise ValueError(f"Unknown agent {args.agent}")

    if args.verbose:
        agent.llm.verbose = True
    return agent


def main():
    config, args = load_config()
    available_agents = list(config.keys())
    assert (
        args.agent in available_agents
    ), f"Invalid agent. Available agents: {available_agents}"

    config = config[args.agent]

    env = create_env(args, config)

    # run agent, loop over the tasks
    if "benchmark" in config and "problems" in config:
        if "all" == config["problems"]:
            problem_list = env.dataset.keys()  # all tasks
            print(problem_list)
        else:
            assert isinstance(config["problems"], list)
            problem_list = config["problems"]

        num_workers = 1 if args.verbose else int(os.environ.get("FROGGY_WORKERS", 1))
        tasks_done = 0
        mean_perf = 0

        pbar = tqdm(range(len(problem_list)))
        with ThreadPoolExecutor(num_workers) as executor:
            futures = [
                executor.submit(run_agent, args, problem, config)
                for problem in problem_list
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    print(f"Task generated an exception: {e}")
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
