# Changelog


### 2025-05-20

Changed the tool-calling syntax to be compatible with the [OpenAI](https://platform.openai.com/docs/guides/function-calling) and [Anthropic](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use) function-calling formats.

* Switched tools (view, rewrite, pdb, listdir, eval) to a function-call API with explicit arguments and environment injection.
* Overhauled LLM interfaces to define, parse, and format function calls, and updated agents to consume `ToolCall` objects.
* Removed the old conversational-prompt flag from configs.

### 2024-05-22

Added in the [analysis](https://github.com/microsoft/debug-gym/tree/main/analysis/json_log_viewer) folder a Flask app to view `.jsonl` log files in the browser.