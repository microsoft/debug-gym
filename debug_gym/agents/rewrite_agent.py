from debug_gym.agents.base_agent import BaseAgent, register_agent


@register_agent
class RewriteAgent(BaseAgent):
    name: str = "rewrite_agent"
    system_prompt: str = (
        "Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools, you can use them to investigate the code and propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. At every step, you have to use one of the tools via function calling. "
    )
    action_prompt: str = (
        "Based on the instruction, the current code, the last execution output, and the history information, continue your debugging process to propose a patch using the rewrite tool. You can only call one tool at a time. Do not repeat your previous action unless they can provide more information. You must be concise and avoid overthinking."
    )
