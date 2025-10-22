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

        normalized = self._normalize_messages_for_template(messages)

        if self.apply_chat_template:
            # When applying chat template, tokenize all messages together
            # then return as a single list
            text = tokenizer.apply_chat_template(
                normalized,
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
            for msg in normalized:
                content = str(msg["content"])
                tokens = tokenizer.tokenize(content)
                result.append(tokens)
            return result

    # --- chat template helpers -------------------------------------------------

    def _normalize_messages_for_template(self, messages: Iterable[dict]) -> list[dict]:
        """Normalize messages to be a list of dicts with 'role' and 'content' keys.
        Roles are mapped to 'system', 'user', 'assistant'. Content is converted to string.
        """
        normalized = []
        for message in messages:
            role = message.get("role", "user")
            if role not in {"system", "user", "assistant"}:
                role = "user"

            content = message.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    else:
                        parts.append(str(item))
                content = "\n".join(parts)
            elif content is None and message.get("tool_calls"):
                content = json.dumps(message.get("tool_calls"))
            else:
                content = "" if content is None else str(content)

            normalized.append({"role": role, "content": content})
        return normalized
