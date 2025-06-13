# Changelog

### 2025-06-11

Added support to [SWE-smith](https://swesmith.com/). Users can use the tasks shipped with the official SWE-smith package, or customized tasks generated using SWE-smith.

### 2025-06-03
* Refactored `debug_gym/agents/llm_api.py` into separate modules in `debug_gym/llms/` for OpenAI, AzureOpenAI, Anthropic APIs, and human mode, allowing for easier extension to other LLM providers in the future.
* Improved the Human mode to support better prompt completion and error handling.

### 2025-05-28

Improved the View tool, added the `start` and `end` arguments so the agent can specify a particular chunk of code to view.

### 2025-05-22

Added in the [analysis](https://github.com/microsoft/debug-gym/tree/main/analysis/json_log_viewer) folder a Flask app to view `.jsonl` log files in the browser.

### 2025-05-20

Changed the tool-calling syntax to be compatible with the [OpenAI](https://platform.openai.com/docs/guides/function-calling) and [Anthropic](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use) function-calling formats.

* Switched tools (view, rewrite, pdb, listdir, eval) to a function-call API with explicit arguments and environment injection.
* Overhauled LLM interfaces to define, parse, and format function calls, and updated agents to consume `ToolCall` objects.
* Removed the old conversational-prompt flag from configs.
