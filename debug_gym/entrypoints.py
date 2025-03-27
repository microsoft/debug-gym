import argparse
import os
from pathlib import Path

LLM_CONFIG_TEMPLATE = """# Please edit this file replacing the placeholders with your own values.
gpt-4o:
  model: gpt-4o
  tokenizer: gpt-4o
  endpoint: "[YOUR_ENDPOINT]"
  api_key: "[YOUR_API_KEY]"
  tags: [gpt-4o, azure openai, GCR]
  api_version: "2024-09-01-preview"
  context_limit: 128
  generate_kwargs:
    temperature: 0.5

o1-mini:
  model: o1-mini
  tokenizer: gpt-4o
  endpoint: "[YOUR_ENDPOINT]"
  api_key: "[YOUR_API_KEY]"
  tags: [gpt-4o, azure openai, GCR]
  api_version: "2024-09-01-preview"
  context_limit: 128
  system_prompt_support: false
  ignore_kwargs: [temperature, top_p, presence_penalty, frequency_penalty, logprobs, top_logprobs, logit_bias, max_tokens]

gpt-4o-az-login:
  model: gpt-4o
  tokenizer: gpt-4o
  endpoint: "[YOUR_ENDPOINT]"
  scope: "[YOUR_SCOPE]"
  tags: [gpt-4o, azure openai, GCR]
  api_version: "2024-09-01-preview"
  context_limit: 128
  generate_kwargs:
    temperature: 0.5

deepseek-r1-distill-qwen-32b:
  model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
  tokenizer: Qwen/Qwen2.5-32B
  endpoint: "[YOUR_ENDPOINT]"
  api_key: "[YOUR_API_KEY]"
  tags: [DeepSeek-R1-Distill-Qwen-32B, H100]
  system_prompt_support: false
  context_limit: 128
  reasoning_end_token: "</think>"
  generate_kwargs:
    temperature: 0.5

claude-3.7:
  model: claude-3-7-sonnet-20250219
  tokenizer: claude-3-7-sonnet-20250219
  tags: [anthropic, claude, claude-3.7]
  context_limit: 100
  api_key: "[YOUR_API_KEY]"
  generate_kwargs:
    max_tokens: 8192
    temperature: 0.5

claude-3.7-thinking:
  model: claude-3-7-sonnet-20250219
  tokenizer: claude-3-7-sonnet-20250219
  tags: [anthropic, claude, claude-3.7]
  context_limit: 100
  api_key: "[YOUR_API_KEY]"
  generate_kwargs:
    max_tokens: 20000
    temperature: 1
    thinking:
      type: enabled
      budget_tokens: 16000
"""


def copy_llm_config_template(dest_dir: str = None):
    """Copy the llm config template to the specified
    directory or the user's home directory."""

    parser = argparse.ArgumentParser(
        description="Create an LLM config template in the specified directory or `~/.config/debug_gym`."
    )
    parser.add_argument(
        "destination", nargs="?", type=str, help="Destination directory (positional)"
    )
    parser.add_argument("-d", "--dest", type=str, help="Destination directory")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Override the file if it already exists",
    )

    args = parser.parse_args()
    force = args.force

    if args.destination is not None:
        dest_dir = Path(args.destination)
    elif args.dest is not None:
        dest_dir = Path(args.dest)
    else:
        dest_dir = Path.joinpath(Path.home(), ".config", "debug_gym")

    os.makedirs(dest_dir, exist_ok=True)

    destination = dest_dir / "llm.yaml"
    if not os.path.exists(destination):
        with open(destination, "w") as f:
            f.write(LLM_CONFIG_TEMPLATE)
        print(f"LLM config template created at `{destination}`.")
    elif force:
        with open(destination, "w") as f:
            f.write(LLM_CONFIG_TEMPLATE)
        print(f"LLM config template overridden at `{destination}`.")
    else:
        print(f"LLM config template already exists at `{destination}`.")

    print("Please edit the file to configure your LLM settings.")
