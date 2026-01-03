from functools import lru_cache

from transformers import AutoTokenizer

from debug_gym.llms.openai import OpenAILLM


@lru_cache(maxsize=5)
def _get_hf_tokenizer(tokenizer_name: str, tokenizer_kwargs_tuple: tuple):
    """Cache HuggingFace tokenizers to limit memory usage (max 5 different tokenizers).

    Note: tokenizer_kwargs is converted to a tuple of items for hashability.
    """
    tokenizer_kwargs = dict(tokenizer_kwargs_tuple) if tokenizer_kwargs_tuple else {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    # Ensure we have a pad token to avoid downstream warnings
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class HuggingFaceLLM(OpenAILLM):
    """LLM implementation backed by a Hugging Face tokenizer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hf_tokenizer = None

    def _load_tokenizer(self):
        if self._hf_tokenizer is None:
            tokenizer_kwargs = getattr(self.config, "tokenizer_kwargs", None) or {}
            # Convert dict to tuple of items for hashability in lru_cache
            tokenizer_kwargs_tuple = tuple(sorted(tokenizer_kwargs.items()))
            try:
                self._hf_tokenizer = _get_hf_tokenizer(
                    self.tokenizer_name, tokenizer_kwargs_tuple
                )
            except OSError:
                raise ValueError(
                    f"Tokenizer `{self.tokenizer_name}` not found for model "
                    f"{self.model_name}, make sure you have access to "
                    "the model (e.g., HuggingFace API key is correctly set)."
                )
        return self._hf_tokenizer

    def tokenize(self, messages: list[dict]) -> list[list[str]]:
        tokenizer = self._load_tokenizer()

        if self.apply_chat_template:
            # When applying chat template, tokenize all messages together
            # then return as a single list
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            tokens = tokenizer.tokenize(text)
            # Return as list with single element (all tokens together)
            return [tokens]
        else:
            # Tokenize each message individually
            result = []
            for msg in messages:
                content = str(msg["content"])
                tokens = tokenizer.tokenize(content)
                result.append(tokens)
            return result
