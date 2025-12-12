from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.gym.envs.env import EnvInfo, RepoEnv
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.llms.base import LLM


@register_agent
class SimpleAgent(BaseAgent):
    name: str = "simple_agent"

    def _parse_tool_call(self, response: str) -> ToolCall:
        # Extract tool call from LLM response.
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
            tool_call = {}
            lines = response.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("<function=") and line.endswith(">"):
                    tool_call["name"] = line[len("<function=") : -1]
                    tool_call["arguments"] = {}
                    i += 1
                    while i < len(lines):
                        line = lines[i].strip()
                        if line.startswith("<parameter=") and line.endswith(">"):
                            param_name = line[len("<parameter=") : -1]
                            param_value_lines = []
                            i += 1
                            while i < len(lines):
                                line = lines[i].strip()
                                if line == f"</parameter>":
                                    break
                                param_value_lines.append(lines[i])
                                i += 1
                            param_value = "\n".join(param_value_lines).rstrip()
                            tool_call["arguments"][param_name] = param_value
                        elif line == "</function>":
                            break
                        i += 1
                i += 1

            return ToolCall(
                id="None",
                name=tool_call.get("name", "unknown_function"),
                arguments=tool_call.get("arguments", {}),
            )
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
                llm_response.tool_call = self._parse_tool_call(llm_response.response)

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
