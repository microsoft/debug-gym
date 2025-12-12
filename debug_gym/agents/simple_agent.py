import re

from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.gym.envs.env import EnvInfo, RepoEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms.base import LLM


@register_agent
class SimpleAgent(BaseAgent):
    name: str = "simple_agent"

    def _parse_tool_call(self, response: str) -> list[ToolCall]:
        # Extract tool calls from LLM response.
        # Supports multiple tool calls in a single response.
        # Assume the following format.
        # <function=example_function_name>
        # <parameter=example_parameter_1>value_1</parameter>
        # <parameter=example_parameter_2>
        # This is the value for the second parameter
        # that can span
        # multiple lines
        # </parameter>
        # </function>
        try:
            tool_calls = []

            # Extract all function blocks with their content
            func_pattern = r"<function=([^>]+)>(.*?)</function>"
            for func_match in re.finditer(func_pattern, response, re.DOTALL):
                function_name = func_match.group(1)
                function_content = func_match.group(2)

                # Extract all parameters within this function block
                arguments = {}
                param_pattern = r"<parameter=([^>]+)>(.*?)</parameter>"
                for param_match in re.finditer(
                    param_pattern, function_content, re.DOTALL
                ):
                    param_name = param_match.group(1)
                    param_value = param_match.group(2).rstrip()
                    arguments[param_name] = param_value

                tool_calls.append(
                    ToolCall(
                        id="None",
                        name=function_name,
                        arguments=arguments,
                    )
                )

            # Return list with unknown_function if no tool calls found
            if not tool_calls:
                tool_calls.append(
                    ToolCall(id="None", name="unknown_function", arguments={})
                )

            return tool_calls
        except Exception as e:
            self.logger.warning(
                f"Failed to parse tool call from LLM response: {e!r}. "
                f"LLM response was: {response}. "
                "The agent will stop execution."
            )
            return None

    def run(self, env: RepoEnv, llm: LLM, debug=False):
        self.env = env
        self.llm = llm
        info = None
        step = 0

        try:
            info = self.env.reset()
            self.history.init(
                self.build_system_prompt(info), self.build_instance_prompt(info), info
            )

            if info.resolved:
                self.logger.report_progress(
                    problem_id=env.task_name,
                    step=0,
                    total_steps=self.args.max_steps,
                    score=info.score,
                    max_score=info.max_score,
                    status="resolved",
                )
                return self._build_trajectory()

            highscore = info.score
            should_stop = False
            step = 1

            while not should_stop:
                self.logger.info(f"\n{'='*20} STEP {step} {'='*20}\n")

                messages = self.build_prompt(info)
                llm_response = self.llm(messages, tools=None)
                tool_calls = self._parse_tool_call(llm_response.response)

                # Handle multiple tool calls - use the first one
                if tool_calls and len(tool_calls) > 1:
                    self.logger.info(
                        f"Multiple tool calls detected ({len(tool_calls)}), using the first one."
                    )

                # TODO: deal with multiple tool calls.
                llm_response.tool_call = tool_calls[0] if tool_calls else None

                if debug:
                    breakpoint()

                info = self.env.step(
                    llm_response.tool,
                    llm_response.response,
                    llm_response.reasoning_response,
                )
                self.history.step(info, llm_response)
                should_stop, reason = self.should_stop(step + 1, info)
                status = (
                    "resolved"
                    if info.resolved
                    else ("unresolved" if should_stop else "running")
                )

                highscore = max(highscore, info.score)
                msg = f"[{env.task_name[:10]:<10}] Step {step} | Score: {info.score}/{info.max_score or '-'} [Best: {highscore}]"
                if should_stop:
                    msg += f" | Stopping Reason: {reason}"
                self.logger.info(msg)
                step += 1

                # keep progress bar running until max_steps is reached
                self.logger.report_progress(
                    problem_id=env.task_name,
                    step=step,
                    total_steps=self.args.max_steps,
                    score=info.score,
                    max_score=info.max_score,
                    status=status,
                )
            return self._build_trajectory()
        except Exception as e:
            # report any error that happens during the run
            self.logger.report_progress(
                problem_id=env.task_name,
                step=step,
                total_steps=step,
                score=getattr(info, "score", 0),
                max_score=getattr(info, "max_score", None),
                status="error",
            )
            raise e
