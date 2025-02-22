# Froggy: an Interactive Debugging Framework

<img src="https://github.com/microsoft/Froggy/blob/main/media/froggy_logo.png" width=50% height=50%>

`froggy` is a text-based interactive debugging framework, designed for debugging Python programs. 

[[Technical Report](https://arxiv.org/)] [[Project Page](https://arxiv.org/)]

## 1. Installation

    conda create -n froggy python=3.12
    conda activate froggy
    pip install -e .

To install the development dependencies:

    pip install -e '.[dev]'

**Set your API information in llm.cfg**

First, make a copy of the template,

    cp llm.cfg.template llm.cfg

Then, edit llm.cfg with your endpoint and credentials. You can choose one of these authentication methods:
- For authenticating with an API key, provide `api_key`.
- For `az login` or Managed Identity authentication, remove `api_key` and include `scope` instead.

> [!WARNING]
> When using open-sourced LLMs, e.g., via vLLM, you need to correctly setup `HF_TOKEN` required by the tokenizer.

---

## 2. System Design

The structure of `froggy` is as below:
```bash
froggy
├── pond
│   ├── envs
│   ├── terminal
│   └── tools
└── agents
```

`froggy.pond` is a simulation environment. Given a code repository, an agent can iteratively interact with a set of tools, such as `pdb`, that are designed for investigate the code. Once gathered enough information, the agent can propose a patch that rewrites certain lines of the code. The terminal will subsequently execute the new code against a set of test cases.

`froggy.agents` are LLM-based debugging agents that use `froggy.pond` to interact with code repositories to seek necessary information and thus fix potential bugs. At an interaction step, the agent takes a text observation that describes the environment states and tool states as input, it is expected to generate a command, subsequently, the environment will provide a new text observation in response, describing the state change caused by that command. 
 
---

#### 2.1. Environment and Tools

Our base environment, `RepoEnv`, is an interactive environment that follows the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) paradigm. Once the environment `env` is instantiated, one can use `env.reset()` to start an episode and receives initial informations. Then, one can interact with the environment using `env.step(action)`, where `action` specifies one of the available tools (see below), doing so will return subsequent informations (e.g, error message, debugger stdout, etc.)

One of the core designs of `froggy` is the notion of tools. Users can dynamically import tools, or develop customized tools and utilize them in the environment. Tools are modules that augment an agent's action space, observation space, or provide additonal functionalities to the agent. Below are the set of tools we have implemented so far.

| Tool name | Description |
| :-: | :----- |
| `listdir` | It returns the directory tree at a given subdirectory. This is particularly useful when dealing with a repository with multiple files. |
| `view` | It is used to change an agent's focus to a particular source code file. This is particularly useful when dealing with a repository with multiple files. |
| `eval` | It runs the current code repository using the provided entrypoint (e.g., pytest), and returns the terminal's output (e.g., error message). |
| `pdb` | Interactive debugger wrapping the [Python pdb tool](https://docs.python.org/3/library/pdb.html). In additon, users can choose to maintain a set of persistent breakpoints (as in some programming IDEs), which are not reset after every eval. With such feature, a new pdb debugging session is activated automatically, with all the breakpoints restored. Note such breakpoint can be cleared by pdb commands such as `cl`. |
| `rewrite` | It can be used to rewrite a certain piece of code to fix the bug. The inputs of this tool call include the file path, the start and end line numbers, and the new code. |

Upon importing a tool, its action space and observation space will be automatically merged into `froggy`'s action space and observation space; its instruction will also be merged into the overall instruction provided to the agent (e.g., as system prompt).

Users can include a `.froggyignore` file in the repository to specify files and directories that are not visible to `froggy`, similarly, they can include a `.froggyreadonly` to specify files and directories that are read only by `froggy` (e.g., the test files). Both files share the same syntax as `.gitignore`.

---

#### 2.2. Agents

We provide the below LLM-based agents, they all have minimal design and serve the purpose of demonstrating the `froggy` APIs. 

| Agent name | Available Tools | Description |
| :-: | :-: | :----- |
| `pdb_agent` | `pdb`, `patcher`, `view`, `eval` | A minimal agent that dumps all available information into its prompt and queries the LLM to generate a command. |
| `rewrite_only` | `patcher`, `view`, `eval`  | A `pdb_agent` but `pdb` tool is disabled (an agent keeps rewriting). |
| `pdb_after_rewrite` | `pdb`, `patcher`, `view`, `eval`  | A `pdb_agent`, but `pdb` tool is only enabled after certain amount of rewrites. |

---

#### 2.3. Benchmarks

To demonstrate how to integrate `froggy` with coding tasks and repositories, we provide example code importing two widely used benchmarks, namely `aider` and `swebench`, and a small set of minimal buggy code snippets, namely `mini_nightmare`.

| Benchmark name | Link |
| :-: | :----- |
| `aider` | [https://github.com/Aider-AI/aider](https://github.com/Aider-AI/aider) |
| `swebench`| [https://github.com/princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) |
| `mini_nightmare` | A set of 10 hand-crafted minimal buggy code snippet where rewrite only agents have harder time to tackle. Read details [here](https://github.com/microsoft/Froggy/blob/main/data/mini_nightmare/mini_nightmare.md). |

---

## 3. Running Baselines
We use `.yaml` files to specify configurations. Example config files can be found in `scripts/`. To run an agent:

    python scripts/run.py scripts/config_<benchmark name>.yaml --agent <agent name>

Add `-v`, `--debug` to be verbose, or to enter debug mode.
> [!WARNING]
> When using --debug, you will need to press `c` to continue after each reasoning step.


#### 3.1. Overriding Values in Config

`-p` is a handy way to override values defined in config. For example, the below command will run rewrite_only agent on Aider with human mode (while in config file it specifies gpt-4o).

    python scripts/run.py scripts/config_aider.yaml --agent rewrite_only -v -p rewrite_only.llm_name="human"

#### 3.2. Debugging a Custom Repository

Modify `scripts/config.yaml`, especially the `env_kwargs` to set the path and entrypoint of the custom repository. We assume there is a `.froggyignore` file and a `.froggyreadonly` within the repository that labels files/folders that are not seen or not editable, respectively.

As an example, we provide a buggy pytorch code repository in `data/pytorch`.

    python scripts/run.py scripts/config.yaml --agent <agent name>

#### 3.3. Design Your Own Tool
`froggy`'s modular design makes it extensible. Users are encouraged to extend `froggy` to their specific usecases, for example by creating new tools that diversify an agent's action and observation spaces. For detailed instruction on designing new tools that are `froggy`-compatible, please refer to the [Technical Report](https://arxiv.org/). 

## Citation
```
tbd
```

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
