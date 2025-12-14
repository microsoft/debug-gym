import re
from dataclasses import dataclass
from typing import List, Tuple

from debug_gym.agents.base_agent import (
    AgentArgs,
    BaseAgent,
    LLMResponse,
    register_agent,
)
from debug_gym.gym.envs.env import EnvInfo, RepoEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms.base import LLM


@dataclass
class SimpleAgentArgs(AgentArgs):
    system_prompt: str = """You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>

You have access to the following functions:

---- BEGIN FUNCTION #1: bash ----
Description: Execute a bash command in the terminal.

Parameters:
(1) command (string, required): The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.
---- END FUNCTION #1 ----

---- BEGIN FUNCTION #2: submit ----
Description: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
No parameters are required for this function.
---- END FUNCTION #2 ----

If you choose to call a function ONLY reply in the following format with NO suffix:

Provide any reasoning for the function call here.
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Always provide reasoning for your function call in natural language BEFORE the function call (not after)
</IMPORTANT>
"""
    instance_prompt: str = """
I have uploaded a python code repository in the /testbed directory.

Now consider the following instructions:

\n\n{info.instructions}\n\n

Can you help me solve the issue?
"""


@register_agent
class SimpleAgent(BaseAgent):
    name: str = "simple_agent"

    def parse_tool_call(self, tool_call: str) -> List[ToolCall]:
        """
        Parses a string of the form:

          <function=FUNCTION_NAME>
            <parameter=KEY>VALUE</parameter>
            ...
          </function>

        and returns a ToolCall object.

        For example:
          <function=file_editor>
            <parameter=command>view</parameter>
            <parameter=path>./sympy/tensor/array/dense_ndim_array.py</parameter>
            <parameter=concise>True</parameter>
          </function>
        """
        tool_calls = []
        func_pattern = r"<function=([^>]+)>(.*?)</function>"
        for func_match in re.finditer(func_pattern, tool_call, re.DOTALL):
            function_name = func_match.group(1)
            function_content = func_match.group(2)

            pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
            param_matches = re.findall(pattern, function_content, flags=re.DOTALL)

            params = {}
            for param_key, param_value in param_matches:
                param_key = param_key.strip()
                param_value = param_value.strip()
                params[param_key] = param_value

            tool_calls.append(ToolCall(id="None", name=function_name, arguments=params))
        return tool_calls

    def parse_response(self, response_text: str) -> Tuple[str, List[ToolCall]]:
        """
        Extracts:
        - thought: everything before the first <function=...> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<function=` up to the first `</function>`
        pattern = re.compile(r"(?s)(<function=.*?</function>)")
        match = pattern.search(response_text)

        if match:
            action = match.group(1)  # The entire <function=...></function> block
            thought = response_text[: match.start()]  # Everything before the block
        else:
            # If no match, treat entire text as "thought"
            thought = response_text
            action = ""

        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        tool_calls = self.parse_tool_call(action)
        return thought, tool_calls

    def step(self, info: EnvInfo) -> LLMResponse | List[LLMResponse]:
        """Execute a single agent step (LLM decision only).

        Args:
            info: Current environment info.

        Returns:
            LLMResponse with the agent's decision.
        """
        messages = self.build_prompt(info)
        response = self.llm(messages, tools=None)
        thought, tool_calls = self.parse_response(response.response)
        if tool_calls and len(tool_calls) > 1:
            self.logger.info(
                f"Multiple tool calls detected ({len(tool_calls)}), using the first one."
            )
        response.response = thought
        response.tool = tool_calls[0] if tool_calls else None
        return response
