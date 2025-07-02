from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.agents.history_tracker import build_history_prompt
from debug_gym.gym.tools.context import ContextTool


@register_agent
class RewriteAgent(BaseAgent):
    name: str = "rewrite_agent"
    system_prompt: str = (
        "Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools, you can use them to investigate the code and propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. At every step, you have to use one of the tools via function calling. "
    )
    action_prompt: str = (
        "Based on the instruction, the current code, the last execution output, and the history information, continue your debugging process to propose a patch using the rewrite tool. You can only call one tool at a time. Do not repeat your previous action unless they can provide more information. You must be concise and avoid overthinking."
    )


@register_agent
class RewriteContextAgent(BaseAgent):
    name: str = "rewrite_context_agent"
    system_prompt: str = (
        "Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools, you can use them to investigate the code, modify your own history of the interactions. At every step, you have to use one of the tools via function calling (view, rewrite, modify_history, eval). "
    )
    action_prompt: str = (
        "Based on the instruction, the current code, the last execution output, and the history information, continue your debugging process. You can only call one tool at a time. Do not repeat your previous action unless they can provide more information. You must be concise and avoid overthinking. You can modify your own past history by removing useless information or rewriting some past messages with more concise content."
    )

    def run(self, task_name=None, debug=False):
        # add the context tool to the environment
        self.env.add_tool(ContextTool(self.history))

        return super().run(task_name, debug)

    def build_history_prompt(self):
        messages = build_history_prompt(
            self.history.filter_out(actions=[None]),
            self.llm,
            self.config["reset_prompt_history_after_rewrite"],
        )
        i = 0
        for message in messages:
            if 'content' in message and isinstance(message['content'], str):
                message["content"] = f"Message idx = {i}\n\n" + message["content"]
                i += 1
        return messages
