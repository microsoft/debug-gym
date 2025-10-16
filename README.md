# debug-gym: A Text-Based Environment for Interactive Debugging

`debug-gym` is a text-based interactive debugging framework, designed for debugging Python programs.

[[Technical Report](https://arxiv.org/abs/2503.21557)] [[Project Page](https://aka.ms/debug-gym/)]

The technical report corresponds to [version 1.0.0](https://github.com/microsoft/debug-gym/tree/1.0.0). Please see [CHANGELOG.md](https://github.com/microsoft/debug-gym/blob/main/CHANGELOG.md) for recent updates.

## 1. Installation

It's recommended to create and activate a conda or virtual environment. `debug-gym` requires `Python>=3.12`:

    conda create -n debug-gym python=3.12
    conda activate debug-gym

Then, install `debug-gym` directly from PyPI:

    pip install debug-gym

Alternatively, clone the repository and install locally:

    git clone https://github.com/microsoft/debug-gym
    cd debug-gym
    pip install -e .

To install development dependencies, run:

    pip install -e '.[dev]'


**Set your API information in llm.yaml**

First, create an LLM config template by running `python -m debug_gym.llms.configure`:

    python -m debug_gym.llms.configure

> [!TIP]
> Run `python -m debug_gym.llms.configure --help` for more options. By default, the template is created at `$HOME/.config/debug_gym/llm.yaml`, but you can specify any directory.

Then, edit this file with your endpoint and credentials. You can choose one of these authentication methods:
- For authenticating with an API key, provide `api_key`.
- For `az login` or Managed Identity authentication on Azure, remove `api_key` and include `scope` instead.

> [!WARNING]
> When using open-sourced LLMs, e.g., via vLLM, you need to correctly setup `HF_TOKEN` required by the tokenizer. You can also provide `tokenizer_kwargs` in your `llm.yaml` entry (for example `trust_remote_code: true`) to control how the Hugging Face tokenizer is instantiated.

By default, `debug-gym` looks for the LLM config file at `$HOME/.config/debug_gym/llm.yaml`. You can change this behavior by exporting the environment variable `LLM_CONFIG_FILE_PATH` or by setting `llm_config_file_path` in your script config file (see [Running Baselines](#3-running-baselines)).

---

## 2. System Design

The structure of `debug-gym` is as below:
```bash
debug_gym
├── gym
│   ├── envs
│   ├── terminals
│   └── tools
├── agents
└── llms
```

`debug_gym.gym` is a simulation environment. Given a code repository, an agent can iteratively interact with a set of tools, such as `pdb`, that are designed for investigate the code. Once gathered enough information, the agent can propose a patch that rewrites certain lines of the code. The terminal will subsequently execute the new code against a set of test cases.

`debug_gym.agents` are LLM-based debugging agents that use `debug_gym.gym` to interact with code repositories to seek necessary information and thus fix potential bugs. At an interaction step, the agent takes a text observation that describes the environment states and tool states as input, it is expected to generate a command, subsequently, the environment will provide a new text observation in response, describing the state change caused by that command.

`debug_gym.llms` are the different LLM backends that can be used to instantiate agents. Currently, we support OpenAI, Azure OpenAI, Hugging Face/vLLM deployments (via an OpenAI-compatible endpoint), and Anthropic. For Hugging Face models served through vLLM, the tokenizer's chat template is applied automatically to ensure token counting and truncation match the hosted model.

> [!WARNING]
> `debug-gym` has limited support on non-Linux platforms. Interactive terminal sessions using PTY (pseudo-terminal) in Docker are not fully supported on macOS or Windows. As a result, the `pdb` tool (see [2.1. Environment and Tools](#21-environment-and-tools)) only works on Linux.

---

#### 2.1. Environment and Tools

Our base environment, `RepoEnv`, is an interactive environment that follows the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) paradigm. Once the environment `env` is instantiated, one can use `env.reset()` to start an episode and receives initial informations. Then, one can interact with the environment using `env.step(action)`, where `action` specifies one of the available tools (see below), doing so will return subsequent informations (e.g, error message, debugger stdout, etc.)

One of the core designs of `debug-gym` is the notion of tools. Users can dynamically import tools, or develop customized tools and utilize them in the environment. Tools are modules that augment an agent's action space, observation space, or provide additonal functionalities to the agent. Below are the set of tools we have implemented so far.

| Tool name | Description |
| :-: | :----- |
| `listdir` | It returns the directory tree at a given subdirectory. This is particularly useful when dealing with a repository with multiple files. |
| `view` | It is used to change an agent's focus to a particular source code file. This is particularly useful when dealing with a repository with multiple files. |
| `eval` | It runs the current code repository using the provided entrypoint (e.g., pytest), and returns the terminal's output (e.g., error message). |
| `pdb` | Interactive debugger wrapping the [Python pdb tool](https://docs.python.org/3/library/pdb.html). In additon, users can choose to maintain a set of persistent breakpoints (as in some programming IDEs), which are not reset after every eval. With such feature, a new pdb debugging session is activated automatically, with all the breakpoints restored. Note such breakpoint can be cleared by pdb commands such as `cl`. |
| `grep` | Search for patterns in files within the repository. Supports both literal string matching and regular expressions. Can search in specific files, directories, or the entire repository. Useful for finding code patterns, function definitions, variable usage, or identifying files containing specific text. |
| `rewrite` | It can be used to rewrite a certain piece of code to fix the bug. The inputs of this tool call include the file path, the start and end line numbers, and the new code. |

Upon importing a tool, its action space and observation space will be automatically merged into `debug-gym`'s action space and observation space; its instruction will also be merged into the overall instruction provided to the agent (e.g., as system prompt).

Users can include a `.debugignore` file in the repository to specify files and directories that are not visible to `debug-gym`, similarly, they can include a `.debugreadonly` to specify files and directories that are read only by the agents (e.g., the test files). Both files share the same syntax as `.gitignore`.

---

#### 2.2. Agents

We provide the below LLM-based agents, they all have minimal design and serve the purpose of demonstrating the `debug-gym` APIs.

| Agent name | Available Tools | Description |
| :-: | :-: | :----- |
| `debug_agent` | `pdb`, `rewrite`, `view`, `eval` | A minimal agent that dumps all available information into its prompt and queries the LLM to generate a command. |
| `rewrite_agent` | `rewrite`, `view`, `eval`  | A `debug_agent` but `pdb` tool is disabled (an agent keeps rewriting). |
| `debug_5_agent` | `pdb`, `rewrite`, `view`, `eval`  | A `debug_agent`, but `pdb` tool is only enabled after certain amount of rewrites. |
| `grep_agent` | `grep`, `rewrite`, `view`, `eval`  | A variant of `rewrite_agent` that includes the `grep` tool for searching patterns in the codebase before making changes. |
| `solution_agent` | `pdb`, `eval`  | An oracle agent that applies a gold patch (only works with `swebench` and `swesmith` benchmarks for now). The agent checks that tests are failing before applying the patch, and passing after. It also checks that `pdb` tool can be used as expected. |

---

#### 2.3. Benchmarks

To demonstrate how to integrate `debug-gym` with coding tasks and repositories, we provide example code importing two widely used benchmarks, namely `aider` and `swebench`, and a small set of minimal buggy code snippets, namely `mini_nightmare`.

| Benchmark name | Link |
| :-: | :----- |
| `aider` | [https://github.com/Aider-AI/aider](https://github.com/Aider-AI/aider) |
| `swebench`| [https://github.com/princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) |
| `swesmith`| [https://github.com/SWE-bench/SWE-smith](https://github.com/SWE-bench/SWE-smith) |
| `r2egym`| [https://github.com/R2E-Gym/R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) |
| `mini_nightmare` | A set of 10 hand-crafted minimal buggy code snippet where rewrite only agents have harder time to tackle. Read details [here](https://github.com/microsoft/debug-gym/blob/main/data/mini_nightmare/mini_nightmare.md). |


---

#### 2.4. Terminals

`debug-gym` supports multiple terminal backends to accommodate different execution environments and deployment scenarios. Each terminal type provides a consistent interface while handling the underlying infrastructure differently.

| Terminal Type | Description |
| :-: | :----- |
| `LocalTerminal` | Executes commands directly on the local machine using bash. Ideal for development and testing on local systems. |
| `DockerTerminal` | Executes commands inside Docker containers running on your machine. Provides isolated execution environments. (Recommended) |
| `KubernetesTerminal` | Executes commands in Kubernetes pods for scalable deployments. Provides isolated execution environments. Suitable when dealing with large benchmarks like `swebench`, `swesmith`, and `r2egym`. |

All terminals support:
- Specify custom working directories and session commands
- Environment variable configuration
- Command execution with timeout handling
- Output capturing and error reporting
- Automatic cleanup and resource management
- Retry mechanisms for transient errors
- Provide a way to create persistent interactive shell sessions using pseudo-terminals (PTY). Used internally by interactive debugging tools such as `pdb`.
> [!WARNING]
> Interactive shell sessions are not fully compatible with macOS due to their reliance on pty.

Terminal selection is configured through the `terminal_config` in your script configuration file. The framework automatically handles terminal initialization, command execution, and cleanup based on the specified type.

---

## 3. Running Baselines
We use `.yaml` files to specify configurations. Example config files can be found in `scripts/`. To run an agent:

    python scripts/run.py scripts/config_<benchmark name>.yaml --agent <agent name>

Add `-v`, `--debug` to be verbose, or to enter debug mode.
> [!WARNING]
> When using --debug, you will need to press `c` to continue after each reasoning step.

#### 3.1 Sanity Checks

We can use the `solution_agent` to validate that your `swebench` and `swesmith` instances work as expected. This agent will apply a gold patch to the buggy code and check that the tests are failing before applying the patch, and passing after. It also checks that `pdb` tool can be used as expected.

    python scripts/run.py scripts/config_swebench.yaml --agent solution_agent
    python scripts/run.py scripts/config_swesmith.yaml --agent solution_agent

#### 3.2 Human Mode

We provide a human mode that enables developers to manually interact with `debug-gym`. To activate this mode, change the `llm_name` field in the `config_*.yaml` to be `"human"`. Once activated, at every step, the environment will expect a command input (in tool calling format). One can use the `Tab` key to get a list of tool calling templates and fill in any necessary arguments.

#### 3.3. Overriding Values in Config

The `-p` flag is a handy way to override values defined in the config file. For example, the command below will run the rewrite_agent agent on Aider with human mode (even if the config file specifies gpt-4o). The command also overrides the default system prompt (see below for more information).

    python scripts/run.py scripts/config_aider.yaml \
        --agent debug_agent \
        -v \
        -p debug_agent.llm_name="human" \
        -p debug_agent.system_prompt_template_file="scripts/templates/human_friendly_system_prompt.jinja"


#### 3.4. Customizing the System Prompt with Jinja Templates

`debug-gym` allows you to fully customize the system prompt by providing a [Jinja](https://jinja.palletsprojects.com/) template file. This enables you to control the format and content of the prompt sent to the LLM, making it easier to adapt the environment to your specific needs or research experiments.

To use a custom system prompt template, specify the path to your Jinja template file in your agent's configuration under `system_prompt_template_file`. For example:

```yaml
debug_agent:
  system_prompt_template_file: scripts/templates/custom_system_prompt.jinja
```

Alternatively, you can provide a custom template from the command line with `-p <agent>.system_prompt_template_file="<path/to/template.jinja>"` (see above).

Within your Jinja template, you have access to the `agent` and `info` objects, which provide all relevant context about the current environment and agent state.

#### Custom Jinja Filters

In addition to all [built-in Jinja filters](https://jinja.palletsprojects.com/en/stable/templates/#list-of-builtin-filters), two custom filters are available for use in your template:

- **`to_pretty_json`**: Converts a Python object to a pretty-printed JSON string. Useful for displaying structured data in a readable format.
    ```jinja
    {{ info.tools | to_pretty_json }}
    ```

- **`trim_message`**: Trims a string to fit within a token or character limit, also filtering out non-UTF8 characters. This is helpful for ensuring that large outputs (such as directory trees or evaluation results) do not exceed the LLM's context window. The `trim_message` filter accepts the following arguments to control how messages are trimmed:
    - **`max_length`**: The maximum number of tokens to keep in the message. If the message exceeds this length, it will be trimmed.
    - **`max_length_percentage`**: Instead of specifying an absolute number, you can provide a percentage (e.g., `0.1` for 10%) of the LLM's context window. The message will be trimmed to fit within this percentage of the model's maximum context length.
    - **`where`**: Specifies where to trim the message if it exceeds the limit. The default is `"middle"`, which trims from the middle of the message. Other options are `start` or `end`.

    ```jinja
    {{ info.dir_tree | trim_message(max_length_percentage=0.1, where="end") }}
    ```

#### Example Template

```jinja
System Prompt for Debug-Gym

Task: {{ agent.system_prompt }}

Instructions:
{{ info.instructions }}

Directory Tree:
{{ info.dir_tree | trim_message(max_length=1000) }}

Current Breakpoints:
{{ info.current_breakpoints | to_pretty_json }}

{% if agent.shortcut_features() %}
Shortcut Features:
{{ agent.shortcut_features() | to_pretty_json }}
{% endif %}
```


#### 3.5. Debugging a Custom Repository

Modify `scripts/config.yaml`, especially the `env_kwargs` to set the path and entrypoint of the custom repository. We assume there is a `.debugignore` file and a `.debugreadonly` within the repository that labels files/folders that are not seen or not editable, respectively.

As an example, we provide a buggy pytorch code repository in `data/pytorch`.

    python scripts/run.py scripts/config.yaml --agent <agent name>

#### 3.6. Debugging a Custom SWE-Smith Instance

[SWE-Smith](https://github.com/SWE-bench/SWE-smith) allows to generate new buggy code instances. Give a custom HuggingFace dataset (either local or remote) that has a similar structure as [SWE-bench/SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith), one can override the `-p base.env_kwargs.dataset_id=<dataset_id>` in the command line to run the agent on that dataset. For example, to run on a local dataset:

    python scripts/run.py scripts/config_swesmith.yaml --agent <agent name> -p base.env_kwargs.dataset_id="path/to/local/dataset"

#### 3.7. Design Your Own Tool
`debug-gym`'s modular design makes it extensible. Users are encouraged to extend `debug-gym` to their specific usecases, for example by creating new tools that diversify an agent's action and observation spaces. For detailed instruction on designing new tools that are `debug-gym`-compatible, please refer to the [Technical Report](https://arxiv.org/abs/2503.21557).

#### 3.8. Analysis and Visualization

We provide a set of scripts to help analyze the log files (e.g., the `.jsonl` files) generated by the agent.
- In the `analysis` folder, we provide scripts that used to generate the corresponding figures in our technical report.
- In the `analysis/json_log_viewer` folder, we provide a Flask app to view a `.jsonl` log file in the browser.

## Citation
```
@article{yuan2025debuggym,
  title={debug-gym: A Text-Based Environment for Interactive Debugging},
  author={Xingdi Yuan, Morgane M Moss, Charbel El Feghali, Chinmay Singh, Darya Moldavskaya, Drew MacPhee, Lucas Caccia, Matheus Pereira, Minseon Kim, Alessandro Sordoni, Marc-Alexandre C\^ot\'e},
  journal={arXiv preprint arXiv:2503.21557},
  year={2025},
  url={https://arxiv.org/abs/2503.21557}
}
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

## Privacy
This framework does not collect user's personal data. For more information about Microsoft's privacy policies. Please see [Microsoft Privacy Statement](https://www.microsoft.com/en-ca/privacy/privacystatement).

## Responsible AI
Please see our [Responsible AI Statement](https://github.com/microsoft/debug-gym/blob/main/RESPONSIBLE_AI.md).
