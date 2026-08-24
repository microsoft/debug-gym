from importlib import import_module

__all__ = [
    "AnthropicLLM",
    "AzureOpenAILLM",
    "LLM",
    "HuggingFaceLLM",
    "Human",
    "OpenAILLM",
]

_LLM_CLASSES = {
    "AnthropicLLM": "debug_gym.llms.anthropic",
    "AzureOpenAILLM": "debug_gym.llms.azure_openai",
    "LLM": "debug_gym.llms.base",
    "HuggingFaceLLM": "debug_gym.llms.huggingface",
    "Human": "debug_gym.llms.human",
    "OpenAILLM": "debug_gym.llms.openai",
}


def __getattr__(name: str):
    if module_name := _LLM_CLASSES.get(name):
        return getattr(import_module(module_name), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
