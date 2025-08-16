from debug_gym.llms.base import LLM
from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.agents.history_tracker import HistoryTracker
from debug_gym.gym.envs.env import EnvInfo
from debug_gym.gym.entities import Observation
import json

@register_agent
class DebugAgent(BaseAgent):
    name = "debug_agent"
    system_prompt = "You are a debugging agent specialized in fixing Python programs. Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools including the pdb debugger to help you investigate the code before proposing a patch. While the code may seem familiar to you from your training, you should not assume you know the code. Instead, you must use the pdb debugger to investigate the code and understand the potential bugs. A common debugging workflow is to 1) find suspicious files and lines (from error messages or test failures); 2) set breakpoints at suspicious places; 3) continue execution so the frame is at the breakpoint you set; 4) then print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. You can only call one tool at a time. Do not repeat your previous action, especially if it returned tool calling errors or it resulted in information that you already know. You can think step by step to help you make the decision at every step, but you must be concise and avoid overthinking. If you are confident that you have enough information, propose a patch to fix the bugs by calling the rewrite tool. If you are not sure, continue using the pdb tool to gather more information before proposing a patch. After every rewrite, it's always a good idea to call the eval tool to execute the new code and check if it passes the tests; if it does not, the tool will return the error messages, which you can use to continue debugging. Output both your thinking process (if any) and the tool call in the response. "


@register_agent
class Debug_5_Agent(DebugAgent):
    name: str = "debug_5_agent"

    def run(self, task_name=None, debug=False):
        step = 0
        max_steps = self.config["max_steps"]
        try:
            # remove the pdb tool from the environment
            pdb_tool = self.env.remove_tool("pdb")

            self.history.reset()
            info = self.env.reset(options={"task_name": task_name})
            # initial state does not have prompt and response
            self.history.step(info, None)

            if info.done is True:
                # msg = "Environment started with entrypoint passing without errors."
                self.logger.report_progress(
                    problem_id=task_name,
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
                self.logger.info(
                    f"Step: {step} | Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%}) [Best: {highscore}]"
                )

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
                    info.rewrite_counter >= self.config["n_rewrites_before_pdb"]
                    and pdb_tool.name not in self.env.tools
                ):
                    self.env.add_tool(pdb_tool)
                    pdb_tool.start_pdb()
                    # update info tools related fields after adding pdb so it's included when building the next prompt
                    info.instructions = self.env.instructions
                    info.tools = self.env.tools

                self.history.step(info, llm_response)

                if (
                    info.done
                    or info.rewrite_counter >= self.config["max_rewrite_steps"]
                ):
                    reason = "done" if info.done else "max_rewrite_steps reached"
                    self.logger.info(
                        f"Step: {step} | Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%}) | Reason: {reason}"
                    )
                    # early stop, set current step and total steps to be the same
                    self.logger.report_progress(
                        problem_id=task_name,
                        step=step + 1,
                        total_steps=step + 1,
                        score=info.score,
                        max_score=info.max_score,
                        status="resolved" if info.done else "unresolved",
                    )
                    break
                # keep progress bar running until max_steps is reached
                self.logger.report_progress(
                    problem_id=task_name,
                    step=step + 1,
                    total_steps=max_steps + 1,
                    score=info.score,
                    max_score=info.max_score,
                    status="running",
                )
            # max_steps was reached, task was either resolved or unresolved
            self.logger.report_progress(
                problem_id=task_name,
                step=step + 1,
                total_steps=step + 1,
                score=info.score,
                max_score=info.max_score,
                status="resolved" if info.done else "unresolved",
            )
            return info.done
        except Exception:
            # report any error that happens during the run
            self.logger.report_progress(
                problem_id=task_name,
                step=step + 1,
                total_steps=step + 1,
                score=info.score if info else 0,
                max_score=info.max_score if info else 1,
                status="error",
            )
            raise

@register_agent
class ReplayAgent(DebugAgent):
    name: str = "replay_agent"
    
    def run(self, task_name=None, debug=False):
        print("running!")
        step = 0
        info = None
        max_steps = self.config["max_steps"]
        try:
            self.history.reset()
            info = self.env.reset(options={"task_name": task_name})
            # initial state does not have prompt and response
            self.history.step(info, None)

            if info.done is True:
                self.logger.report_progress(
                    problem_id=task_name,
                    step=1,
                    total_steps=1,
                    score=info.score,
                    max_score=info.max_score,
                    status="resolved",
                )
                return True

            self.logger.info(
                "Available tools (in LLM's tool calling format):\n"
                f"{json.dumps(self.llm.define_tools(info.tools), indent=4)}\n"
            )

            highscore = info.score
            for step in range(max_steps):
                print(step)
                critique_step = self.config.get("replay_from", None)
                critique = self.config.get("critique", "")
                if critique_step is not None and step == critique_step: 
                    new_observation = Observation(
                        source="user",
                        observation=critique
                    )
                    info = EnvInfo(
                        step_observation=new_observation,
                        all_observations=info.all_observations + [new_observation],
                        action_tool_call=None,
                        action_content=None,
                        action_reasoning=None,
                        rewrite_counter=info.rewrite_counter,
                        score=info.score,
                        max_score=info.max_score,
                        done=info.done,
                        instructions=info.instructions,
                        tools=info.tools,
                        dir_tree=info.dir_tree,
                        current_breakpoints=info.current_breakpoints,
                        eval_observation=info.eval_observation,
                    )
                    self.history.step(info, None)
                
                self.logger.info(f"\n{'='*20} STEP {step+1} {'='*20}\n")
                highscore = max(highscore, info.score)
                self.logger.info(
                    f"[{task_name[:10]:<10}] | Step: {step:<4} | Score: {info.score:>4}/{info.max_score:<4} ({info.score/info.max_score:.1%}) [Best: {highscore}]"
                )

                messages = self.build_prompt(info)
                llm_response = self.llm(messages, info.tools)
                
                if debug:
                    breakpoint()

                info = self.env.step(
                    llm_response.tool,
                    llm_response.response,
                    llm_response.reasoning_response,
                )
                self.history.step(info, llm_response)
                
                

                if (
                    info.done
                    or info.rewrite_counter >= self.config["max_rewrite_steps"]
                ):
                    reason = "done" if info.done else "max_rewrite_steps reached"
                    self.logger.info(
                        f"Step: {step} | Score: {info.score}/{info.max_score} ({info.score/info.max_score:.1%}) | Reason: {reason}"
                    )
                    # early stop, set current step and total steps to be the same
                    self.logger.report_progress(
                        problem_id=task_name,
                        step=step + 1,
                        total_steps=step + 1,
                        score=info.score,
                        max_score=info.max_score,
                        status="resolved" if info.done else "unresolved",
                    )
                    break
                # keep progress bar running until max_steps is reached
                self.logger.report_progress(
                    problem_id=task_name,
                    step=step + 1,
                    total_steps=max_steps + 1,
                    score=info.score,
                    max_score=info.max_score,
                    status="running",
                )
            # max_steps was reached, task was either resolved or unresolved
            self.logger.report_progress(
                problem_id=task_name,
                step=step + 1,
                total_steps=step + 1,
                score=info.score,
                max_score=info.max_score,
                status="resolved" if info.done else "unresolved",
            )
            return info.done
        except Exception:
            # report any error that happens during the run
            self.logger.report_progress(
                problem_id=task_name,
                step=step + 1,
                total_steps=step + 1,
                score=info.score if info else 0,
                max_score=info.max_score if info else 1,
                status="error",
            )
            raise

def build_history_prompt(
    history: HistoryTracker, llm: LLM, reset_prompt_history_after_rewrite: bool = False
):
    _history, _prompt_response_pairs = history.get()
    latest_rewrite_step = 0
    # Find the latest rewrite step if reset_prompt_history_after_rewrite
    if reset_prompt_history_after_rewrite:
        for i in range(len(_history)):
            if _history[i].rewrite_counter == _history[-1].rewrite_counter:
                latest_rewrite_step = i
                break
    _messages = []
    for history_info, response in zip(
        _history[latest_rewrite_step:], _prompt_response_pairs[latest_rewrite_step:]
    ):
        _messages.extend(llm.format_tool_call_history(history_info, response))
    return _messages

def insert_critique_into_history(
    history: HistoryTracker, critique: str, step_id: int = None
):
    """
    Insert a critique into the history at the specified step.
    If step_id is None, insert at the last step.
    If step_id is longer than the history, return the history unchanged.
    """
    new_history = history.clone()
    if step_id is None:
        step_id = len(history) - 1
    if step_id < 0 or step_id >= len(history):
        return new_history
    
    before_step = new_history.memory[:step_id]
    after_step = new_history.memory[step_id + 1:]
    new_step = new_history.memory[step_id].clone()
    new_step.action_reasoning = None
    new_step.action_content = None
    new_step.action_tool_call = None
    new_step.step_observation = critique
    

    new_history.memory = before_step + [new_step] + after_step
    return new_history