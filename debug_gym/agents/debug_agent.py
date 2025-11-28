from dataclasses import dataclass

from debug_gym.agents.base_agent import register_agent
from debug_gym.agents.froggy_agent import FroggyAgent, FroggyAgentArgs
from debug_gym.gym.envs.env import RepoEnv


@register_agent
class DebugAgent(FroggyAgent):
    name = "debug_agent"
    system_prompt = "You are a debugging agent specialized in fixing Python programs. Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools including the pdb debugger to help you investigate the code before proposing a patch. While the code may seem familiar to you from your training, you should not assume you know the code. Instead, you must use the pdb debugger to investigate the code and understand the potential bugs. A common debugging workflow is to 1) find suspicious files and lines (from error messages or test failures); 2) set breakpoints at suspicious places; 3) continue execution so the frame is at the breakpoint you set; 4) then print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. You must make tool calls to interact with the environment, but you can only call one tool at a time. Do not repeat your previous action, especially if it returned tool calling errors or it resulted in information that you already know. You can spend some time thinking to help you make the decision when you are stuck, but you must be concise and avoid overthinking. If you already had a plan in the previous steps, you can just follow it without repeating the thinking process. If you are confident that you have enough information, propose a patch to fix the bugs by calling the rewrite tool. If you are not sure, continue using the pdb tool to gather more information before proposing a patch. After every rewrite, it's always a good idea to call the eval tool to execute the new code and check if it passes the tests; if it does not, the tool will return the error messages, which you can use to continue debugging. Output both your thinking process (if any) and the tool call (must) in the response. "


@dataclass
class Debug5AgentArgs(FroggyAgentArgs):
    n_rewrites_before_pdb: int = 0


@register_agent
class Debug_5_Agent(DebugAgent):
    name: str = "debug_5_agent"

    def run(self, env: RepoEnv, debug=False):
        self.env = env
        step = 0
        max_steps = self.args.max_steps
        try:
            # remove the pdb tool from the environment
            pdb_tool = self.env.remove_tool("pdb")

            self.history.reset()
            info = self.env.reset()
            # initial state does not have prompt and response
            self.history.step(info, None)

            if info.resolved is True:
                # msg = "Environment started with entrypoint passing without errors."
                self.logger.report_progress(
                    problem_id=env.task_name,
                    step=1,
                    total_steps=1,
                    score=info.score,
                    max_score=info.max_score,
                    status="resolved",
                )
                return True

            highscore = info.score
            for step in range(max_steps):
                self.logger.info(f"\n{'='*20} STEP {step+1} {'='*20}\n")
                highscore = max(highscore, info.score)
                msg = f"[{env.task_name[:10]:<10}] Step {step} | Score: {info.score}/{info.max_score or '-'} [Best: {highscore}]"
                self.logger.info(msg)

                messages = self.build_prompt(info)
                llm_response = self.llm(messages, info.tools)

                if debug:
                    breakpoint()

                info = self.env.step(
                    llm_response.tool,
                    llm_response.response,
                    llm_response.reasoning_response,
                )

                # re-introduce pdb tool at the right time
                if (
                    info.rewrite_counter >= self.args.n_rewrites_before_pdb
                    and pdb_tool.name not in self.env.tools
                ):
                    self.env.add_tool(pdb_tool)
                    pdb_tool.start_pdb()
                    # update info tools related fields after adding pdb so it's included when building the next prompt
                    info.instructions = self.env.instructions
                    info.tools = self.env.tools

                self.history.step(info, llm_response)

                if (
                    info.terminated
                    or info.rewrite_counter >= self.args.max_rewrite_steps
                ):
                    reason = (
                        "terminated" if info.resolved else "max_rewrite_steps reached"
                    )
                    self.logger.info(
                        f"Step: {step} | Score: {info.score}/{info.max_score if info.max_score else '-'} | Reason: {reason}"
                    )
                    # early stop, set current step and total steps to be the same
                    self.logger.report_progress(
                        problem_id=env.task_name,
                        step=step + 1,
                        total_steps=step + 1,
                        score=info.score,
                        max_score=info.max_score,
                        status="resolved" if info.resolved else "unresolved",
                    )
                    break
                # keep progress bar running until max_steps is reached
                self.logger.report_progress(
                    problem_id=env.task_name,
                    step=step + 1,
                    total_steps=max_steps + 1,
                    score=info.score,
                    max_score=info.max_score,
                    status="running",
                )
            # max_steps was reached, task was either resolved or unresolved
            self.logger.report_progress(
                problem_id=env.task_name,
                step=step + 1,
                total_steps=step + 1,
                score=info.score,
                max_score=info.max_score,
                status="resolved" if info.resolved else "unresolved",
            )
            return info.resolved
        except Exception:
            # report any error that happens during the run
            self.logger.report_progress(
                problem_id=env.task_name,
                step=step + 1,
                total_steps=step + 1,
                score=info.score if info else 0,
                max_score=info.max_score if info else None,
                status="error",
            )
            raise
