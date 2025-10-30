from debug_gym.agents.base_agent import BaseAgent, register_agent


@register_agent
class RewriteAgent(BaseAgent):
    name: str = "rewrite_agent"
    system_prompt: str = (
        "Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools, you can use them to investigate the code and propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. You must make tool calls to interact with the environment, but you can only call one tool at a time. Do not repeat your previous action unless they can provide more information. You can spend some time thinking to help you make the decision when you are stuck, but you must be concise and avoid overthinking. If you already had a plan in the previous steps, you can just follow it without repeating the thinking process. Output both your thinking process (if any) and the tool call (must) in the response. "
    )
