# Froggy

<img src="https://github.com/microsoft/Froggy/blob/main/media/froggy_logo.png" width=50% height=50%>

Froggy is an interactive debugging system for Python. This LLM-based agent can import tools such as `pdb` to interactively investigate the code and generate patches to fix it.

## Installation

    conda create -n froggy python=3.12
    conda activate froggy
    pip install -e .

To install the development dependencies:

    pip install -e '.[dev]'

### Set your API information in llm.cfg
First, make a copy of the template,

    cp llm.cfg.template llm.cfg

Then, edit `llm.cfg` with your endpoint and API key information. If using `az login` for authentication, remove the `api_key` and provide the `scope` instead.

> [!WARNING]
> When using open-sourced LLMs, e.g., via vLLM, you need to correctly setup `HF_TOKEN` required by the tokenizer.


## System Design

Our base environment, `RepoEnv`, is an interactive environment that follows the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) paradigm. Once the environment `env` instantiated, one can use `env.reset()` to start an episode and receives initial informations. Then, one can interact with the environment using `env.step(action)`, where `action` is one of the available tools (see below), doing so will return subsequent informations (e.g, error message, debugger stdout, etc.)

One of the core designs of Froggy is the notion of tools. Users can dynamically import tools, or develop customized tools and utilize them in the environment. Tools are modules that augment an agent's action space, observation space, or provide additonal functionalities to the agent. Below are the set of tools we have implemented so far.

| Tool name | Description |
| :-: | :----- |
| `listdir` | Listdir returns the directory tree at a given subdirectory. This is particularly useful when dealing with a repository with multiple files. |
| `view` | Viewing tool is used to change an agent's focus to a particular source code file. This is particularly useful when dealing with a repository with multiple files. |
| `eval` | Eval tool runs the current code repository using the provided entrypoint (e.g., pytest). |
| `pdb` | Interactive debugger wrapping the python pdb tool. In additon, users can choose to maintain a set of persistent breakpoints (as in some programming IDEs), which are not reset after every eval. With such feature, a new pdb debugging session is activated automatically, with all the breakpoints restored. Note such breakpoint can be cleared by pdb commands such as `cl`. |
| `patcher` | Patchers are modules that rewrite a certain piece of code to fix the bug. There can be many implementations, such as rewriting a entire file, rewriting a chunk of code in a file, or applying a diff patch to a file. Note that a patcher can only modify files that are editable, which is defined in a `.froggyignore` file in the working repository, sharing the same syntax as `.gitignore`. |
| `reasoning` | Reasoning tool enables the model to output explicit reasoning text. Unlike CoT, the reasoning tool maintains the reasoning text in the history as if it were any other tool/action. When initializing, passing ```allow_chain_action = True``` to allow the agent to output another action after the reasoning tokens, in the same step. |

Upon importing a tool, its action space and observation space will be automatically merged into the agent's action space and observation space; its instruction will also be merged into the overall instruction provided to the agent (e.g., as system prompt).

## Running Baselines

### Agents

We have the below LLM-based agents available:

| Agent name | Description |
| :-: | :----- |
| `zero_shot` | A minimal agent that takes all available information as part of the prompt and asks the LLM to generate a command. |
| `cot`| A two-step agent, it first asks the LLM to think step-by-step about the current debugging state, then based on this to generate a command. |
| `tadpole` | A hierarchical agent consisting a task decomposer and a command generator. The task decomposer determines to continue the current subgoal or to switch to a new one; based on the subgoal, the command generator generates a command. |
| `zero_shot_nopdb` | `zero_shot` agent, pdb tool is disabled (an agent keeps rewriting). |
| `cot_nopdb`| `cot` agent, pdb tool is disabled. |

### Benchmarks

| Benchmark name | Link |
| :-: | :----- |
| `aider` | [https://github.com/Aider-AI/aider](https://github.com/Aider-AI/aider) |
| `swebench`| [https://github.com/princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) |
| `terminal_simulator`| A dataset where bug are generated using LLMs, based on human-authored working code. |

### Run

    python scripts/run.py scripts/config_<benchmark name>.yaml --agent <agent name>

Add `-v`, `--debug` to be verbose, or to enter debug mode.
> [!WARNING]
> When using --debug, you will need to press `c` to continue after each reasoning step.

### Debugging Custom Repo

Modify `scripts/config.yaml`, especially the `env_kwargs` to set the path and entrypoint of the custom repository. We assume there is a `.froggyignore` file within the repository that labels files/folders that are not editable.

As an example, we provide a buggy pytorch code repository in `data/pytorch`.

    python scripts/run.py scripts/config.yaml --agent <agent name>


### Overriding values in config

`-p` is a handy way to override values defined in config. For example, the below command will run zero_shot agent on aider with human mode (while in config file it specifies llama)

    python scripts/run.py scripts/config_aider.yaml --agent zero_shot -v -p zero_shot.llm_name="human"

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
