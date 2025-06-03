from debug_gym.agents.base_agent import BaseAgent, register_agent


@register_agent
class Agent1(BaseAgent):
    name = "agent1"
    system_prompt = "You are a debugging agent specialized in fixing Python programs. Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools including the pdb debugger to help you investigate the code before proposing a patch. While the code may seem familiar to you from your training, you should not assume you know the code. Instead, you must use the pdb debugger to investigate the code and understand the potential bugs. A common debugging workflow is to find suspicious files and lines (from error messages or test failures); set breakpoints; continue execution so the frame is at the breakpoint you set; then print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. "

    action_prompt = "Based on the information you have collected, continue your debugging process using the provided tools. You can only call one tool at a time. Do not repeat your previous action, especially if it returned tool calling errors or it resulted in information that you already know. You must be concise and avoid overthinking. If you are confident that you have enough information, propose a patch to fix the bugs by calling the rewrite tool. If you are not sure, continue using the pdb tool to gather more information before proposing a patch."


@register_agent
class Agent2(BaseAgent):
    name = "agent2"
    system_prompt = "You are a debugging agent specialized in fixing Python programs. Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools including the pdb debugger to help you investigate the code before proposing a patch. While the code may seem familiar to you from your training, you should not assume you know the code. Instead, you must use the pdb debugger to investigate the code and understand the potential bugs. A common debugging workflow is to find suspicious files and lines (from error messages or test failures); set breakpoints; continue execution so the frame is at the breakpoint you set; then print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. "

    action_prompt = "Based on the information you have collected, continue your debugging process using the provided tools. You can only call one tool at a time. Do not repeat your previous action, especially if it returned tool calling errors or it resulted in information that you already know. You must be concise and avoid overthinking. If you are confident that you have enough information, propose a patch to fix the bugs by calling the rewrite tool. If you are not sure, continue using the pdb tool to gather more information before proposing a patch."


@register_agent
class Agent3(BaseAgent):
    name = "agent3"
    system_prompt = "You are a debugging agent specialized in fixing Python programs. Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools including the pdb debugger to help you investigate the code before proposing a patch. While the code may seem familiar to you from your training, you should not assume you know the code. Instead, you must use the pdb debugger to investigate the code and understand the potential bugs. A common debugging workflow is to find suspicious files and lines (from error messages or test failures); set breakpoints; continue execution so the frame is at the breakpoint you set; then print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. "

    action_prompt = "Based on the information you have collected, continue your debugging process using the provided tools. You can only call one tool at a time. Do not repeat your previous action, especially if it returned tool calling errors or it resulted in information that you already know. You must be concise and avoid overthinking. If you are confident that you have enough information, propose a patch to fix the bugs by calling the rewrite tool. If you are not sure, continue using the pdb tool to gather more information before proposing a patch."


@register_agent
class Agent4(BaseAgent):
    name = "agent4"
    system_prompt = "Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools including the pdb debugger, you can use them to investigate the code, set breakpoints, and print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. At every step, you have to use one of the tools via function calling. "

    action_prompt = "Based on the instruction, the current code, the last execution output, and the history information, continue your debugging process by calling the pdb tool or to propose a patch by calling the rewrite tool. You can only call one tool at a time. Do not repeat your previous action unless they can provide more information. You must be concise and avoid overthinking."
