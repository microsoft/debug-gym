from debug_gym.agents.base_agent import BaseAgent, register_agent


@register_agent
class DebugAgent(BaseAgent):
    name = "debug_agent"
    system_prompt = "You are a debugging agent specialized in fixing Python programs. Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools including the pdb debugger to help you investigate the code before proposing a patch. While the code may seem familiar to you from your training, you should not assume you know the code. Instead, you must use the pdb debugger to investigate the code and understand the potential bugs. A common debugging workflow is to 1) find suspicious files and lines (from error messages or test failures); 2) set breakpoints at suspicious places; 3) continue execution so the frame is at the breakpoint you set; 4) then print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. "
    action_prompt = "Based on the information you have collected, continue your debugging process using the provided tools. You can only call one tool at a time. Do not repeat your previous action, especially if it returned tool calling errors or it resulted in information that you already know. You must be concise and avoid overthinking. If you are confident that you have enough information, propose a patch to fix the bugs by calling the rewrite tool. If you are not sure, continue using the pdb tool to gather more information before proposing a patch. After every rewrite, it's always a good idea to call the eval tool to execute the new code and check if it passes the tests; if it does not, the tool will return the error messages, which you can use to continue debugging. "


@register_agent
class Debug_5_Agent(DebugAgent):
    name: str = "debug_5_agent"

    def run(self, task_name=None, debug=False):
        # remove the pdb tool from the environment
        pdb_tool = self.env.remove_tool("pdb")

        self.history.reset()
        info = self.env.reset(options={"task_name": task_name})
        # initial state does not have prompt and response
        self.history.step(info, None)

        if info.done is True:
            # msg = "Environment started with entrypoint passing without errors."
            return True

        highscore = info.score

        for step in self.logger.tqdm(range(self.config["max_steps"])):
            highscore = max(highscore, info.score)
            self.logger.info(
                f"Step: {step} | Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%}) [Best: {highscore}]"
            )

            messages = self.build_prompt(info)
            llm_response = self.llm(messages, info.tools)

            if debug:
                breakpoint()

            info = self.env.step(llm_response.tool)

            # re-introduce pdb tool at the right time
            if (
                info.rewrite_counter >= self.config["n_rewrites_before_pdb"]
                and pdb_tool.name not in self.env.tools
            ):
                self.env.add_tool(pdb_tool)
                pdb_tool.start_pdb()
                # update info tools related fields after adding pdb so it's included when building the next prompt
                info.instructions = self.env.instructions
                info.tools = self.env.tools

            self.history.step(info, llm_response)

            if info.done or info.rewrite_counter >= self.config["max_rewrite_steps"]:
                reason = "done" if info.done else "max_rewrite_steps reached"
                self.logger.info(
                    f"Step: {step} | Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%}) | Reason: {reason}"
                )
                break

        return info.done
