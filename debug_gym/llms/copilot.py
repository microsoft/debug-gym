import hashlib
import hmac
import json
import logging
import os
import subprocess
import time
import uuid

import openai
import tiktoken
from openai import NOT_GIVEN, OpenAI

from debug_gym.llms.base import (
    ContextLengthExceededError,
    LLMResponse,
    retry_on_exception,
)
from debug_gym.llms.openai import OpenAILLM

# Set logging level down to WARNING for endpoint queries.
logging.getLogger("openai").setLevel(logging.WARNING)


class CopilotLLM(OpenAILLM):
    CLIENT_MAX_AGE_SECONDS = 1200  # 20 minutes
    AUTH_RETRY_DELAY_SECONDS = 5

    def __init__(
        self,
        model_name,
        llm_config,
        logger=None,
        runtime_generate_kwargs=None,
    ):
        super().__init__(
            model_name,
            llm_config=llm_config,
            logger=logger,
            runtime_generate_kwargs=runtime_generate_kwargs,
        )
        self._client = None
        self._token_cache = None
        self._token_expires_at = 0
        self._client_created_at = 0

    def create_request_hmac(self, hmac_secret):
        """Create HMAC for request authentication"""
        if not hmac_secret:
            return None
        current = str(int(time.time()))
        signature = hmac.new(
            hmac_secret.encode("utf-8"), current.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"{current}.{signature}"

    def _resolve_vscode_copilot_dir(self) -> str:
        """Resolve the path to the vscode-copilot repository."""

        vscode_copilot_dir = os.environ.get(
            "VSCODE_COPILOT_DIR", os.path.expanduser("~/vscode-copilot")
        )
        return os.path.expanduser(vscode_copilot_dir)

    def _parse_env_value(self, value: str) -> str:
        """Parse a .env file value, handling quotes properly.

        Supports:
        - Unquoted values: HMAC_SECRET=abc123
        - Single quoted: HMAC_SECRET='abc123'
        - Double quoted: HMAC_SECRET="abc123"
        """
        value = value.strip()
        if not value:
            return value

        # Handle quoted values - must start and end with same quote type
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]

        return value

    def _get_hmac_secret(self, vscode_copilot_dir: str) -> str:
        """Load the HMAC secret from environment variables or .env file."""

        hmac_secret = os.environ.get("HMAC_SECRET")
        if not hmac_secret:
            env_file_path = os.path.join(vscode_copilot_dir, ".env")
            if os.path.exists(env_file_path):
                try:
                    with open(env_file_path, "r", encoding="utf-8") as env_file:
                        for line in env_file:
                            line = line.strip()
                            # Skip empty lines and comments
                            if not line or line.startswith("#"):
                                continue
                            if line.startswith("HMAC_SECRET="):
                                raw_value = line.split("=", 1)[1]
                                hmac_secret = self._parse_env_value(raw_value)
                                break
                except Exception as exc:
                    self.logger.warning(
                        "Failed to read .env file at %s: %s", env_file_path, exc
                    )

        if not hmac_secret:
            raise ValueError(
                "HMAC_SECRET not found in environment variables or .env file in vscode-copilot directory"
            )

        return hmac_secret

    def fetch_token(self):
        """Fetch GitHub Copilot token using Node.js script"""
        # Cache token for 30 minutes to avoid frequent fetches
        if self._token_cache and time.time() < self._token_expires_at:
            return self._token_cache

        try:
            vscode_copilot_dir = self._resolve_vscode_copilot_dir()
            if not os.path.exists(vscode_copilot_dir):
                raise ValueError(
                    f"vscode-copilot directory not found at: {vscode_copilot_dir}. "
                    "Set VSCODE_COPILOT_DIR environment variable to the correct path."
                )

            script_path = os.path.join(
                vscode_copilot_dir, "src", "util", "node", "fetch-token-standalone.js"
            )
            result = subprocess.run(
                [
                    "node",
                    script_path,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,  # 30 second timeout to prevent indefinite hangs
            )

            if result.returncode != 0:
                error_msg = f"Command failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nSTDOUT: {result.stdout}"
                raise ValueError(error_msg)

            token = result.stdout.strip()
            if not token:
                raise ValueError("fetch-token-standalone.js returned empty output")

            # Cache the token for 30 minutes
            self._token_cache = token
            self._token_expires_at = time.time() + 1800  # 30 minutes
            return token
        except Exception as exc:
            raise ValueError(f"Failed to get Copilot token: {exc}") from exc

    @property
    def client(self):
        now = time.time()
        reason = None
        if self._client is None:
            reason = "initialize"
        elif now - self._client_created_at >= self.CLIENT_MAX_AGE_SECONDS:
            reason = f"age>{self.CLIENT_MAX_AGE_SECONDS}s"

        if reason:
            self.logger.debug("Creating Copilot client (%s)", reason)
            self._client = self._create_copilot_client()
            self._client_created_at = time.time()

        return self._client

    def _create_copilot_client(self) -> OpenAI:
        vscode_copilot_dir = self._resolve_vscode_copilot_dir()
        hmac_secret = self._get_hmac_secret(vscode_copilot_dir)

        bearer_token = self.fetch_token()
        hmac_value = self.create_request_hmac(hmac_secret)

        if not hmac_value or not bearer_token:
            raise ValueError(
                "Missing HMAC or Bearer token for GitHub Copilot Claude API"
            )

        timestamp = hmac_value.split(".")[0]
        self.logger.debug(
            "Creating client with HMAC timestamp: %s (current time: %s)",
            timestamp,
            int(time.time()),
        )

        return OpenAI(
            api_key=bearer_token,
            base_url=self.config.endpoint or "https://api.enterprise.githubcopilot.com",
            default_headers={
                "X-Interaction-Type": "conversation-agent",
                "OpenAI-Intent": "conversation-agent",
                "X-GitHub-Api-Version": self.config.api_version or "2025-05-01",
                "Copilot-Integration-Id": "vscode-chat-dev",
                "VScode-SessionId": "debug-gym-session",
                "VScode-MachineId": "debug-gym-machine",
                "X-Interaction-Id": str(uuid.uuid4()),
                "X-Initiator": "agent",
                "Editor-Version": "debug-gym/1.0",
                "Editor-Plugin-Version": "debug-gym/1.0",
                "Request-Hmac": hmac_value,
            },
            timeout=300.0,  # 5 minute timeout to prevent indefinite hangs
        )

    def tokenize(self, messages: list[dict]) -> list[list[str]]:
        if getattr(self, "_tk_func", None) is None:
            try:
                encoder = tiktoken.encoding_for_model("gpt-4o")
                # For tiktoken, encode returns list of ints, convert to strings
                self._tk_func = lambda text: [str(t) for t in encoder.encode(text)]
            except KeyError:
                # Simple word-based tokenization as fallback
                self._tk_func = lambda x: x.split()

        # Tokenize each message individually
        result = []
        for msg in messages:
            content = str(msg.get("content", msg.get("tool_calls", msg)))
            tokens = self._tk_func(content)
            result.append(tokens)
        return result

    def need_to_be_retried(self, exception) -> bool:
        # re-use the need_to_be_retried function from the parent class
        need_to_retry = super().need_to_be_retried(exception)
        exception_full_name = (
            f"{exception.__class__.__module__}.{exception.__class__.__name__}"
        )
        logger = self.logger.debug
        if exception_full_name == "openai.AuthenticationError":
            error_message = getattr(exception, "message", str(exception))
            if "HMAC timestamp out of range" in error_message:
                self.logger.info(
                    "HMAC timestamp out of range, regenerating client with fresh timestamp"
                )
                self._invalidate_client_cache()
                need_to_retry = True
                time.sleep(self.AUTH_RETRY_DELAY_SECONDS)
            elif "unauthorized" in error_message.lower():
                self.logger.info("Authentication failure, refreshing token and client")
                self._invalidate_client_cache()
                need_to_retry = True
                time.sleep(self.AUTH_RETRY_DELAY_SECONDS)
        logger(
            f"Error calling {self.model_name}: {exception_full_name!r} {
                getattr(exception, 'message', str(exception))
            }"
        )
        return need_to_retry

    def _invalidate_client_cache(self):
        """Invalidate both token and client cache to force regeneration"""
        self._client = None
        self._token_cache = None
        self._token_expires_at = 0
        self._client_created_at = 0

    def close(self):
        """Clean up HTTP client resources and caches."""
        super().close()  # Clean up the HTTP client
        self._invalidate_client_cache()  # Also clear token caches


class CopilotOpenAILLM(CopilotLLM):
    """GitHub Copilot Claude API backend for debug-gym"""


class CopilotClaudeLLM(CopilotLLM):
    """GitHub Copilot Claude API backend for debug-gym
    This set of endpoints are special, they take list of messages in OpenAI format as output, and return in the Anthropic format.
    """

    def generate(self, messages, tools, **kwargs) -> LLMResponse:
        # claude cannot handle messages with only system prompt
        if messages and len(messages) == 1:
            if messages[0]["role"] == "system":
                # Convert system message to user message for Claude compatibility
                messages = messages + [{"role": "user", "content": "Your response is:"}]

        # oai way of request
        kwargs["max_tokens"] = kwargs.get("max_tokens", NOT_GIVEN)
        try:
            response = retry_on_exception(
                self._perform_chat_completion, self.need_to_be_retried
            )(
                model=self.config.model,
                messages=messages,
                tools=self.define_tools(tools),
                tool_choice="auto",
                **kwargs,
            )
        except openai.BadRequestError as e:
            # Handle specific error for context length exceeded, otherwise just propagate the error
            if self.is_context_length_error(e):
                raise ContextLengthExceededError
            raise
        except openai.APIStatusError as e:
            if "Request Entity Too Large" in e.message:
                raise ContextLengthExceededError
            raise

        # the response is in OpenAI format
        # e.g.,
        # {
        # "choices": [
        #     {
        #     "finish_reason": "tool_calls",
        #     "message": {
        #         "content": "I'll help you get the weather information for Paris. Let me think through this step by step:\n\n**Step 1: Analyze the request**\n- The user is asking for weather information\n- The location 'cified is 'Paris'\n- I have access to a `get_weather` function that can provide this information\n\n**Step 2: Check the function requirements**\n- The `get_weather` function requires one parameter: `location` (string)\n- The user has provided 'Paris' as the location\n- All required parameters are available\n\n**Step 3: Execute the function call**\nI'll now call the weather function with 'Paris' as the location parameter.",
        #         "role": "assistant"
        #     }
        #     },
        #     {
        #     "finish_reason": "tool_calls",
        #     "message": {
        #         "role": "assistant",
        #         "tool_calls": [
        #         {
        #             "function": {
        #             "arguments": "{'location':'Paris'}",
        #             "name": "get_weather"
        #             },
        #             "id": "toolu_vrtx_012pL1KsHJWs6V9g8CMrYAft",
        #             "type": "function"
        #         }
        #         ]
        #     }
        #     }
        # ],
        # "created": 1751829973,
        # "id": "msg_vrtx_01EaXusudrdwnEuTYpx62dSa",
        # "usage": {
        #     "completion_tokens": 198,
        #     "prompt_tokens": 420,
        #     "prompt_tokens_details": {
        #     "cached_tokens": 0
        #     },
        #     "total_tokens": 618
        # },
        # "model": "claude-sonnet-4"
        # }

        text_messages = [
            r.message.content for r in response.choices if r.message.content
        ]
        text_message = text_messages[0] if text_messages else None
        # find the first tool call in the response
        tool_calls = [
            r.message.tool_calls[0] for r in response.choices if r.message.tool_calls
        ]
        tool_call = tool_calls[0] if tool_calls else None
        if tool_call:
            assert tool_call.type == "function"

        thinking_messages = [
            r.message.thinking_content
            for r in response.choices
            if "thinking_content" in r.message and r.message.thinking_content
        ]
        thinking_message = thinking_messages[0] if thinking_messages else None

        llm_response = LLMResponse(
            prompt=messages,
            response=text_message,
            reasoning_response=thinking_message,
            tool=self.parse_tool_call_response(tool_call),
            prompt_token_count=response.usage.prompt_tokens,
            response_token_count=response.usage.completion_tokens,
        )
        return llm_response
