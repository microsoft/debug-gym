from debug_gym.agents.base_agent import BaseAgent, register_agent


@register_agent
class RewriteAgent(BaseAgent):
    name: str = "rewrite_agent"
    system_prompt: str = (
        "Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools, you can use them to investigate the code and propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. At every step, you have to use one of the tools via function calling. You can only call one tool at a time. Do not repeat your previous action unless they can provide more information. You can think step by step to help you make the decision at every step, but you must be concise and avoid overthinking. Output both your thinking process (if any) and the tool call in the response. "
    )
