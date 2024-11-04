# Froggy

## Installation

    conda create -n froggy python=3.12
    conda activate froggy
    pip install -e .

### Set your API information in llm.cfg

    cp llm.cfg.template llm.cfg

Then edit `llm.cfg` with your endpoint and API key information.

## Running Baseline

### Agents

We have the below LLM-based agents available:

| Agent name | Description |
| :-: | :----- |
| `zero_shot` | A minimal agent that takes all available information as part of the prompt and asks the LLM to generate a command. |
| `cot`| A two-step agent, it first asks the LLM to think step-by-step about the current debugging state, then based on this to generate a command. |
| `tadpole` | A hierarchical agent consisting a task decomposer and a command generator. The task decomposer determines to continue the current subgoal or to switch to a new one; based on the subgoal, the command generator generates a command. |
| `zero_shot_nopdb` | `zero_shot` agent, pdb tool is disabled. |
| `cot_nopdb`| `cot` agent, pdb tool is disabled. |

### Benchmarks

| Benchmark name | Link |
| :-: | :----- |
| `aider` | [https://github.com/Aider-AI/aider](https://github.com/Aider-AI/aider) |
| `swebench`| [https://github.com/princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) |
| `terminal_simulator`| A dataset where bug are generated using LLMs, based on human-authored working code. |

### Run

    python examples/run.py scripts/config_<benchmark name>.yaml --agent <agent name>

Add `-v`, `--debug` to be verbose, or to enter debug mode.
> [!WARNING]
> When using --debug, you will need to press `c` to continue after each reasoning step.

### Debugging Custom Repo

Modify `scripts/config.yaml`, especially the `env_kwargs` to set the path and entrypoint of the custom repository. We assume there is a `.pdbignore` file within the repository that labels files/folders that are not editable.

As an example, we provide a buggy pytorch code repository in `data/pytorch`.

    python examples/run.py scripts/config.yaml --agent <agent name>


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
