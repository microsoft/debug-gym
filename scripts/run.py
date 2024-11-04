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
    if "benchmark" not in config:
        from froggy.envs import RepoEnv

        env = RepoEnv(**config["env_kwargs"])
    elif "aider" == config["benchmark"]:
        from froggy.envs import AiderBenchmarkEnv

        env = AiderBenchmarkEnv(**config["env_kwargs"])
    elif "swebench" == config["benchmark"]:
        from froggy.envs import SWEBenchEnv

        env = SWEBenchEnv(**config["env_kwargs"])
    elif "terminal_simulator" == config["benchmark"]:
        from froggy.envs import TerminalSimulatorEnv

        env = TerminalSimulatorEnv(**config["env_kwargs"])
    else:
        # TODO: add SWEBench and Pytorch
        raise ValueError(f"Unknown benchmark {config['benchmark']}")

    for tool in config["tools"]:
        if "view" == tool:
            from froggy.tools.view import ViewTool

            env.add_tool(ViewTool())
        elif "eval" == tool:
            from froggy.tools.eval import EvalTool

            env.add_tool(EvalTool())
        elif "listdir" == tool:
            from froggy.tools.listdir import ListdirTool

            env.add_tool(ListdirTool())
        elif "pdb" == tool:
            from froggy.tools.pdb import PDBTool

            env.add_tool(
                PDBTool(persistent_breakpoints=config["persistent_breakpoints"])
            )
        elif tool.startswith("patcher"):
            from froggy.tools.patchers import CodePatcher

            patcher_name = tool.split(":")[1]
            env.add_tool(CodePatcher.get(patcher_name))
        else:
            raise ValueError(f"Unknown tool {tool}")

    # instantiate agent
    if "zero_shot" == args.agent:
        from froggy.agents import AgentZeroShot

        agent = AgentZeroShot(config, env, verbose=args.verbose)
    elif "cot" == args.agent:
        from froggy.agents import AgentCoT

        agent = AgentCoT(config, env, verbose=args.verbose)
    elif "tadpole" == args.agent:
        from froggy.agents import AgentTadpole

        agent = AgentTadpole(config, env, verbose=args.verbose)
    elif "zero_shot_nopdb" == args.agent:
        from froggy.agents import AgentZeroShot_NoPDB

        agent = AgentZeroShot_NoPDB(config, env, verbose=args.verbose)
    elif "cot_nopdb" == args.agent:
        from froggy.agents import AgentCoT_NoPDB

        agent = AgentCoT_NoPDB(config, env, verbose=args.verbose)
    else:
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
                    f"Running agent {agent.name} on {config["benchmark"]}.{problem}",
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
