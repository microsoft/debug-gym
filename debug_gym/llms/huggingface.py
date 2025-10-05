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
            except OSError as exc:
                raise ValueError(
                    "Failed to load Hugging Face tokenizer "
                    f"`{self.tokenizer_name}` for model {self.model_name}."
                ) from exc

            # Ensure we have a pad token to avoid downstream warnings when invoking
            # the tokenizer in encode mode.
            if (
                getattr(self._hf_tokenizer, "pad_token", None) is None
                and getattr(self._hf_tokenizer, "eos_token", None) is not None
            ):
                self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
        return self._hf_tokenizer

    def tokenize(self, text: str) -> list[str]:
        tokenizer = self._load_tokenizer()
        token_ids = tokenizer.encode(str(text), add_special_tokens=False)

        if hasattr(tokenizer, "convert_ids_to_tokens"):
            try:
                return tokenizer.convert_ids_to_tokens(token_ids)
            except Exception:  # pragma: no cover
                pass

        return [str(t) for t in token_ids]

    def count_tokens(self, text) -> int:
        tokenizer = self._load_tokenizer()
        token_ids = tokenizer.encode(str(text), add_special_tokens=False)
        return len(token_ids)

    # --- chat template helpers -------------------------------------------------

    def _get_message_token_counts(self, messages: list[dict]) -> list[int]:
        if not self._supports_chat_template():
            return super()._get_message_token_counts(messages)

        tokenizer = self._load_tokenizer()
        normalized = self._normalize_messages_for_template(messages)
        counts: list[int] = []
        prev_len = 0

        for idx in range(1, len(normalized) + 1):
            try:
                tokenized = tokenizer.apply_chat_template(
                    normalized[:idx],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            except TypeError:
                tokenized = tokenizer.apply_chat_template(
                    normalized[:idx], tokenize=True
                )
            except ValueError:
                return super()._get_message_token_counts(messages)

            token_ids = (
                tokenized.get("input_ids") if isinstance(tokenized, dict) else tokenized
            )
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]

            if token_ids is None:
                return super()._get_message_token_counts(messages)

            current_len = len(token_ids)
            if current_len == 0 and idx == len(normalized):
                return super()._get_message_token_counts(messages)

            counts.append(max(current_len - prev_len, 0))
            prev_len = current_len

        return counts

    def _supports_chat_template(self) -> bool:
        tokenizer = self._load_tokenizer()
        return hasattr(tokenizer, "apply_chat_template")

    def _normalize_messages_for_template(self, messages: Iterable[dict]) -> list[dict]:
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
