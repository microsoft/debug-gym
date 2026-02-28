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
from debug_gym.gym.tools.tool import EnvironmentTool, ToolCall
from debug_gym.llms.base import LLM

# Templates for generating tool descriptions dynamically
TOOL_TEMPLATE = """---- BEGIN FUNCTION #{index}: {tool_name} ----
Description: {description}

Parameters:
{parameters}
---- END FUNCTION #{index} ----"""

PARAMETER_TEMPLATE = (
    """({index}) {param_name} ({param_type}, {required}): {param_description}"""
)


def describe_tools(tools: list[EnvironmentTool]) -> str:
    """Generates a description of available tools for the system prompt."""
    descriptions = []
    for i, tool in enumerate(tools, start=1):
        if tool.arguments:
            param_descriptions = [
                PARAMETER_TEMPLATE.format(
                    index=j + 1,
                    param_name=param_name,
                    param_type=", ".join(param["type"]),
                    required="optional" if "null" in param["type"] else "required",
                    param_description=param["description"],
                )
                for j, (param_name, param) in enumerate(tool.arguments.items())
            ]
            parameters = "\n".join(param_descriptions)
        else:
            parameters = "No parameters are required for this function."

        tool_description = TOOL_TEMPLATE.format(
            index=i,
            tool_name=tool.name,
            description=tool.description,
            parameters=parameters,
        )
        descriptions.append(tool_description)
    return "\n\n".join(descriptions)


@dataclass
class SimpleAgentArgs(AgentArgs):
    system_prompt: str = """You are a helpful assistant that can interact with a computer to solve tasks.
<IMPORTANT>
* If user provides a path, you should NOT assume it's relative to the current working directory. Instead, you should explore the file system to find the file before working on it.
</IMPORTANT>

You have access to the following functions:

{tools_description}

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
    args_class = SimpleAgentArgs
    _system_prompt_generated: bool = False

    def build_prompt(self, info: EnvInfo) -> list:
        """Build the prompt with dynamically generated tool descriptions."""
        # Generate system prompt with tools on first call (lazy initialization)
        if not self._system_prompt_generated and self.env is not None:
            tools_desc = describe_tools(self.env.tools)
            # Replace placeholder with actual tool descriptions
            self.system_prompt = self.args.system_prompt.format(
                tools_description=tools_desc
            )
            self._system_prompt_generated = True

        return super().build_prompt(info)

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

        # Get valid tool names from environment if available
        valid_tool_names = None
        if self.env is not None and hasattr(self.env, "tools"):
            valid_tool_names = {tool.name for tool in self.env.tools}

        for func_match in re.finditer(func_pattern, tool_call, re.DOTALL):
            function_name = func_match.group(1).strip()
            function_content = func_match.group(2)

            # Validate tool name if we have access to valid tools
            if valid_tool_names is not None and function_name not in valid_tool_names:
                self.logger.warning(
                    f"Unknown tool '{function_name}' requested. "
                    f"Valid tools are: {sorted(valid_tool_names)}"
                )
                continue  # Skip invalid tools

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
