import os
from os.path import join as pjoin

from termcolor import colored

from froggy.tools.toolbox import Toolbox
from froggy.utils import load_config


def select_terminal(terminal_config: None):
    terminal_config = terminal_config or {"type": "local"}
    terminal_type = terminal_config.pop("type")
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


def main():
    config, args = load_config()
    available_agents = list(config.keys())
    assert (
        args.agent in available_agents
    ), f"Invalid agent. Available agents: {available_agents}"
    config = config[args.agent]

    terminal = select_terminal(config.get("terminal"))

    env_class = select_env(config.get("benchmark"))

    env = env_class(**config["env_kwargs"], terminal=terminal)

    # import tools to the environment
    for tool in config["tools"]:
        kwargs = {}
        if tool == "pdb":
            kwargs["persistent_breakpoints"] = config["persistent_breakpoints"]
        tool_instantiated = Toolbox.get_tool(tool, **kwargs)
        print(f"Adding tool to toolbox: {tool_instantiated.__class__.__name__}")
        env.add_tool(tool_instantiated)

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
        case _:
            raise ValueError(f"Unknown agent {args.agent}")

    if args.verbose:
        agent.llm.verbose = True

    # run agent, loop over the tasks
    if "benchmark" in config and "problems" in config:
        if "all" == config["problems"]:
            problem_list = env.dataset.keys()  # all tasks
            print(problem_list)
        else:
            assert isinstance(config["problems"], list)
            problem_list = config["problems"]
        for problem in problem_list:
            if os.path.exists(pjoin(agent._output_path, problem, "froggy.jsonl")):
                print(colored(f"Skipping {problem}, already done.", "yellow"))
                continue
            print(
                colored(
                    f"Running agent {agent.name} on {
                        config["benchmark"]}.{problem}",
                    "green",
                )
            )
            agent.run(task_name=problem, debug=args.debug)
            # optionally apply patch
            if config["save_patch"]:
                agent.save_patch(task_name=problem)
            # save log
            agent.log(task_name=problem)
    else:
        # custom repo
        print(colored(f"Running agent {agent.name}", "green"))
        agent.run(debug=args.debug)
        # optionally apply patch
        if config["save_patch"]:
            agent.save_patch()
        # save log
        agent.log()


if __name__ == "__main__":
    main()
