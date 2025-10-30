from debug_gym.constants import DEBUG_GYM_CONFIG_DIR

DEFAULT_LLM_CONFIG = DEBUG_GYM_CONFIG_DIR / "llm.yaml"
LLM_API_KEY_PLACEHOLDER = "[YOUR_API_KEY]"
LLM_ENDPOINT_PLACEHOLDER = "[YOUR_ENDPOINT]"
LLM_SCOPE_PLACEHOLDER = "[YOUR_SCOPE]"

LLM_CONFIG_TEMPLATE = f"""# Please edit this file replacing the placeholders with your own values.
gpt-4o:
  model: gpt-4o
  tokenizer: gpt-4o
  endpoint: "{LLM_ENDPOINT_PLACEHOLDER}"
  api_key: "{LLM_API_KEY_PLACEHOLDER}"
  tags: [gpt-4o, azure openai, GCR]
  api_version: "2024-09-01-preview"
  context_limit: 128
  generate_kwargs:
    temperature: 0.5

o1-mini:
  model: o1-mini
  tokenizer: gpt-4o
  endpoint: "{LLM_ENDPOINT_PLACEHOLDER}"
  api_key: "{LLM_API_KEY_PLACEHOLDER}"
  tags: [gpt-4o, azure openai, GCR]
  api_version: "2025-04-01-preview"
  context_limit: 128
  ignore_kwargs: [temperature, top_p, presence_penalty, frequency_penalty, logprobs, top_logprobs, logit_bias, max_tokens]

gpt-4o-az-login:
  model: gpt-4o
  tokenizer: gpt-4o
  endpoint: "{LLM_ENDPOINT_PLACEHOLDER}"
  scope: "{LLM_SCOPE_PLACEHOLDER}"
  tags: [gpt-4o, azure openai, GCR]
  api_version: "2024-09-01-preview"
  context_limit: 128
  generate_kwargs:
    temperature: 0.5

deepseek-r1-distill-qwen-32b:
  model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
  tokenizer: Qwen/Qwen2.5-32B
  endpoint: "{LLM_ENDPOINT_PLACEHOLDER}"
  api_key: "{LLM_API_KEY_PLACEHOLDER}"
  tags: [DeepSeek-R1-Distill-Qwen-32B, vllm]
  system_prompt_support: false
  context_limit: 128
  reasoning_end_token: "</think>"
  generate_kwargs:
    temperature: 0.5

qwen3-8b-vllm:
  model: Qwen/Qwen3-8b
  tokenizer: Qwen/Qwen3-8b
  endpoint: "{LLM_ENDPOINT_PLACEHOLDER}"
  api_key: "{LLM_API_KEY_PLACEHOLDER}"
  tags: [qwen3-8b, vllm]
  context_limit: 128
  generate_kwargs:
    temperature: 0.5
    max_tokens: 8192

claude-3.7:
  model: claude-3-7-sonnet-20250219
  tokenizer: claude-3-7-sonnet-20250219
  tags: [anthropic, claude, claude-3.7]
  context_limit: 100
  api_key: "{LLM_API_KEY_PLACEHOLDER}"
  generate_kwargs:
    max_tokens: 8192
    temperature: 0.5

claude-3.7-thinking:
  model: claude-3-7-sonnet-20250219
  tokenizer: claude-3-7-sonnet-20250219
  tags: [anthropic, claude, claude-3.7]
  context_limit: 100
  api_key: "{LLM_API_KEY_PLACEHOLDER}"
  generate_kwargs:
    max_tokens: 20000
    temperature: 1
    thinking:
      type: enabled
      budget_tokens: 16000
"""
