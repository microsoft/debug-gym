import json
from typing import Iterable

from transformers import AutoTokenizer

from debug_gym.llms.openai import OpenAILLM


class HuggingFaceLLM(OpenAILLM):
    """LLM implementation backed by a Hugging Face tokenizer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hf_tokenizer = None

    def _load_tokenizer(self):
        if self._hf_tokenizer is None:
            tokenizer_kwargs = getattr(self.config, "tokenizer_kwargs", None) or {}
            try:
                self._hf_tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name, **tokenizer_kwargs
                )
            except OSError:
                raise ValueError(
                    f"Tokenizer `{self.tokenizer_name}` not found for model "
                    f"{self.model_name}, make sure you have access to "
                    "the model (e.g., HuggingFace API key is correctly set)."
                )

            # Ensure we have a pad token to avoid downstream warnings when invoking
            # the tokenizer in encode mode.
            if (
                getattr(self._hf_tokenizer, "pad_token", None) is None
                and getattr(self._hf_tokenizer, "eos_token", None) is not None
            ):
                self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
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
