import hashlib
import hmac
import json
import logging
import subprocess
import time
import uuid
from pathlib import Path

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
    def __init__(self, model_name, logger=None, llm_config=None, llm_config_file=None):
        super().__init__(model_name, logger, llm_config, llm_config_file)
        self._client = None
        self._token_cache = None
        self._token_expires_at = 0

    def create_request_hmac(self, hmac_secret):
        """Create HMAC for request authentication"""
        if not hmac_secret:
            return None
        current = str(int(time.time()))
        signature = hmac.new(
            hmac_secret.encode("utf-8"), current.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return f"{current}.{signature}"

    def fetch_token(self):
        """Fetch GitHub Copilot token using Node.js script"""
        # Cache token for 30 minutes to avoid frequent fetches
        if self._token_cache and time.time() < self._token_expires_at:
            return self._token_cache

        try:
            # Get the vscode-copilot directory path
            import os

            vscode_copilot_dir = os.environ.get(
                "VSCODE_COPILOT_DIR", os.path.expanduser("~/vscode-copilot")
            )
            if not os.path.exists(vscode_copilot_dir):
                raise ValueError(
                    f"vscode-copilot directory not found at: {vscode_copilot_dir}. "
                    "Set VSCODE_COPILOT_DIR environment variable to the correct path."
                )

            result = subprocess.run(
                [
                    "node",
                    f"{vscode_copilot_dir}/src/util/node/fetch-token-standalone.js",
                ],
                capture_output=True,
                text=True,
                check=True,
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
        except Exception as e:
            raise ValueError(f"Failed to get Copilot token: {e}")

    @property
    def client(self):
        if self._client is None:
            # Get HMAC secret from environment or .env file in vscode-copilot directory
            import os

            # Get the vscode-copilot directory path
            vscode_copilot_dir = os.environ.get(
                "VSCODE_COPILOT_DIR", os.path.expanduser("~/vscode-copilot")
            )

            # Try to load HMAC_SECRET from .env file in vscode-copilot directory
            hmac_secret = os.environ.get("HMAC_SECRET")
            if not hmac_secret:
                env_file_path = os.path.join(vscode_copilot_dir, ".env")
                if os.path.exists(env_file_path):
                    try:
                        with open(env_file_path, "r") as f:
                            for line in f:
                                line = line.strip()
                                if line.startswith("HMAC_SECRET="):
                                    hmac_secret = line.split("=", 1)[1].strip("\"'")
                                    break
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to read .env file at {env_file_path}: {e}"
                        )

            if not hmac_secret:
                raise ValueError(
                    "HMAC_SECRET not found in environment variables or .env file in vscode-copilot directory"
                )

            bearer_token = self.fetch_token()
            hmac_value = self.create_request_hmac(hmac_secret)

            if not hmac_value or not bearer_token:
                raise ValueError(
                    "Missing HMAC or Bearer token for GitHub Copilot Claude API"
                )

            # Create OpenAI client with GitHub Copilot endpoint and custom headers
            self._client = OpenAI(
                api_key=bearer_token,
                base_url=self.config.endpoint
                or "https://api.enterprise.githubcopilot.com",
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
                timeout=None,
            )
        return self._client

    def tokenize(self, text: str) -> list[str]:
        if getattr(self, "_tk_func", None) is None:
            try:
                self._tk_func = tiktoken.encoding_for_model("gpt-4o").encode
            except KeyError:
                # Simple word-based tokenization as fallback
                # For Claude, you might want to use tiktoken or another tokenizer
                self._tk_func = lambda x: x.split()
        return self._tk_func(text)

    def need_to_be_retried(self, exception) -> bool:
        # re-use the need_to_be_retried function from the parent class
        need_to_retry = super().need_to_be_retried(exception)
        exception_full_name = (
            f"{exception.__class__.__module__}.{exception.__class__.__name__}"
        )
        logger = self.logger.debug
        if exception_full_name == "openai.AuthenticationError":
            if "HMAC timestamp out of range" in exception.message:
                # This error indicates that the HMAC timestamp is out of range,
                # which can happen if the system clock is not synchronized.
                # We should retry after a short delay to allow for clock synchronization.
                need_to_retry = True
                time.sleep(5)
        logger(
            f"Error calling {self.model_name}: {exception_full_name!r} {
                exception.message if hasattr(exception, 'message') else exception
            }"
        )
        return need_to_retry


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
                self.client.chat.completions.create, self.need_to_be_retried
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
