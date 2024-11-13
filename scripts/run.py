import os
from os.path import join as pjoin

from termcolor import colored

from froggy.utils import load_config


def main():

    config, args = load_config()
    available_agents = list(config.keys())
    assert (
        args.agent in available_agents
    ), f"Invalid agent. Available agents: {available_agents}"
    config = config[args.agent]

    # create environment
    if "benchmark" in config:
        match config["benchmark"]:
            case "aider":
                from froggy.envs import AiderBenchmarkEnv
                env = AiderBenchmarkEnv(**config["env_kwargs"])
            case "swebench":
                from froggy.envs import SWEBenchEnv
                env = SWEBenchEnv(**config["env_kwargs"])
            case "terminal_simulator":
                from froggy.envs import TerminalSimulatorEnv
                env = TerminalSimulatorEnv(**config["env_kwargs"])
            case _:
                raise ValueError(f"Unknown benchmark {config['benchmark']}")
    else:
        # custom repo
        from froggy.envs import RepoEnv
        env = RepoEnv(**config["env_kwargs"])

    # import tools to the environment
    for tool in config["tools"]:
        match tool:
            case "view":
                from froggy.tools.view import ViewTool
                env.add_tool(ViewTool())
            case "eval":
                from froggy.tools.eval import EvalTool
                env.add_tool(EvalTool())
            case "listdir":
                from froggy.tools.listdir import ListdirTool
                env.add_tool(ListdirTool())
            case "pdb":
                from froggy.tools.pdb import PDBTool
                env.add_tool(
                    PDBTool(persistent_breakpoints=config["persistent_breakpoints"]))
            case "reasoning":
                from froggy.tools.reasoning import ReasoningTool
                env.add_tool(ReasoningTool())
            case _:
                if tool.startswith("patcher"):
                    from froggy.tools.patchers import CodePatcher
                    patcher_name = tool.split(":")[1]
                    env.add_tool(CodePatcher.get(patcher_name))
                else:
                    raise ValueError(f"Unknown tool {tool}")

    # instantiate agent
    match args.agent:
        case "zero_shot":
            from froggy.agents import AgentZeroShot
            agent = AgentZeroShot(config, env, verbose=args.verbose)
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
        case _:
            raise ValueError(f"Unknown agent {args.agent}")

    if args.verbose:
        agent.llm.verbose = True

    # run agent, loop over the tasks
    if "benchmark" in config and "problems" in config:
        if "all" == config["problems"]:
            problem_list = env.dataset.keys()  # all tasks
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
