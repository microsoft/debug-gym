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

        counts = self._compute_message_token_counts(messages)
        if counts is None:
            return super()._get_message_token_counts(messages)
        return counts

    def _supports_chat_template(self) -> bool:
        tokenizer = self._load_tokenizer()
        return hasattr(tokenizer, "apply_chat_template")

    def _normalize_messages_for_template(self, messages: Iterable[dict]) -> list[dict]:
        normalized = []
        for message in messages:
            role = message.get("role", "user")
            if role == "tool":
                role = "assistant"
            elif role not in {"system", "user", "assistant"}:
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

    def _chat_template_token_ids(self, messages: list[dict]):
        tokenizer = self._load_tokenizer()
        if not hasattr(tokenizer, "apply_chat_template"):
            return None

        normalized = self._normalize_messages_for_template(messages)
        try:
            tokenized = tokenizer.apply_chat_template(
                normalized,
                tokenize=True,
                add_generation_prompt=False,
            )
        except TypeError:
            tokenized = tokenizer.apply_chat_template(normalized, tokenize=True)
        except ValueError:
            return None

        if isinstance(tokenized, dict):
            token_ids = tokenized.get("input_ids", [])
        else:
            token_ids = tokenized

        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        return token_ids

    def _compute_message_token_counts(self, messages: list[dict]) -> list[int] | None:
        normalized = self._normalize_messages_for_template(messages)
        counts = []
        prev_len = 0

        for idx in range(len(normalized)):
            partial_ids = self._chat_template_token_ids(normalized[: idx + 1])
            if partial_ids is None:
                return None
            current_len = len(partial_ids)
            counts.append(max(current_len - prev_len, 0))
            prev_len = current_len

        # Fallback in case template produced zero tokens
        if not any(counts):
            tokenizer = self._load_tokenizer()
            return [
                len(
                    tokenizer.encode(
                        str(m.get("content", "")), add_special_tokens=False
                    )
                )
                for m in messages
            ]

        return counts

    def _trim_messages_to_context(
        self,
        messages: list[dict],
        message_token_counts: list[int] | None = None,
    ) -> list[dict]:
        if not self._supports_chat_template():
            return super()._trim_messages_to_context(messages, message_token_counts)

        if message_token_counts is None:
            message_token_counts = self._compute_message_token_counts(messages)
            if message_token_counts is None:
                return super()._trim_messages_to_context(messages, None)

        if len(messages) != len(message_token_counts):
            return super()._trim_messages_to_context(messages, None)

        context_limit = self.context_length
        total_tokens = sum(message_token_counts)
        if total_tokens <= context_limit:
            return messages

        assert messages, "messages should not be empty"

        result = []
        remaining_tokens = context_limit

        # Handle system message if present
        system_idx = 0 if messages[0].get("role") == "system" else None
        if system_idx is not None:
            system_tokens = message_token_counts[0]
            assert (
                system_tokens <= context_limit
            ), f"System message tokens exceed context length: {system_tokens} > {context_limit}!"
            result.append(messages[0])
            remaining_tokens -= system_tokens

        # Locate the first user message
        user_msg_idx = None
        for idx, msg in enumerate(messages):
            if msg.get("role") == "user":
                user_msg_idx = idx
                break

        # Collect assistant/tool pairs starting from the end
        assistant_tool_pairs: list[tuple[int, int]] = []
        i = len(messages) - 1
        while i >= 0:
            if (
                messages[i].get("role") == "tool"
                and i > 0
                and messages[i - 1].get("role") == "assistant"
            ):
                assistant_tool_pairs.append((i - 1, i))
                i -= 2
            else:
                i -= 1

        included_pairs: list[tuple[int, int]] = []
        for assistant_idx, tool_idx in assistant_tool_pairs:
            pair_tokens = (
                message_token_counts[assistant_idx] + message_token_counts[tool_idx]
            )
            if pair_tokens <= remaining_tokens:
                included_pairs.append((assistant_idx, tool_idx))
                remaining_tokens -= pair_tokens
            else:
                break

        include_user = False
        if (
            user_msg_idx is not None
            and len(included_pairs) == len(assistant_tool_pairs)
            and message_token_counts[user_msg_idx] <= remaining_tokens
        ):
            include_user = True

        if include_user:
            result.append(messages[user_msg_idx])

        included_pairs.sort(key=lambda pair: pair[0])
        for assistant_idx, tool_idx in included_pairs:
            result.append(messages[assistant_idx])
            result.append(messages[tool_idx])

        assert (
            result
        ), f"After trimming, no messages fit within context length: {context_limit}!"

        return result
