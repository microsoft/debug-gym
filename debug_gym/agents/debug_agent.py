from debug_gym.agents.base_agent import BaseAgent, register_agent
import glob
from debug_gym.gym.utils import filter_non_utf8


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

                info = self.env.step(llm_response.tool, llm_response.response)

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
class ExplanationAgent(BaseAgent):
    name = "explanation_agent"
    system_prompt = DebugAgent.system_prompt
    explanation_base_path = "../generate-explanations/explanations/failure_critiques/"
    
    # def __init__(self, **kwargs):
    #     filepath = "../generate-explanations/new_explanations_proper_pdb/single_shot_low_level/swe-smith/burnash__gspread.a8be3b96.lm_rewrite__596cni6x/single_shot_explanation_20250711_170309.txt"
    #     self.explanation = self.get_explanation_from_file(filepath)
    #     self.system_prompt = f"{self.system_prompt} {self.explanation} Follow similar steps as the above to debug the code and use the same level high level understanding of information retrieval and debugging."
    #     super().__init__(**kwargs)

    def build_system_prompt(self, info):
        """Build the system prompt with the explanation."""
        system_prompt_template = self._load_system_prompt_template()
        if system_prompt_template is not None:
            system_prompt = system_prompt_template.render(agent=self, info=info)
        else:
            system_prompt = self._default_system_prompt(info)
        try:
            explanation = self.get_explanation_for_task(self.env.task_name)
            system_prompt = f"{system_prompt} {explanation} Use this general advice to assist in your debugging process."
        except FileNotFoundError as e:
            self.logger.error(f"Explanation file not found: {e}")
            system_prompt = f"{system_prompt} [No explanation found]"
        messages = [{"role": "system", "content": filter_non_utf8(system_prompt)}]
        return messages
    
    def get_explanation_for_task(self, task_name: str):
        """Get the explanation for a specific task."""
        # paths = glob.glob(f"{self.explanation_base_path}/{task_name}_system_prompt.txt")
        # if not paths:
        #     raise FileNotFoundError(f"No explanation files found for task: {task_name}")
        # path = sorted(paths)[-1]
        filepath =f"{self.explanation_base_path}/{task_name}_system_prompt.txt"
        explanation = self.get_explanation_from_file(filepath)
        return explanation
    
    # def run(self, task_name=None, debug=False):
    #     # Get the explanation for this specific task
    #     explanation = self.get_explanation_for_task(task_name)
        
    #     # Update system prompt with task-specific explanation
    #     original_system_prompt = self.system_prompt
    #     self.system_prompt = f"{DebugAgent.system_prompt} {explanation}"
        
    #     # Call parent's run method
    #     result = super().run(task_name=task_name, debug=debug)
        
    #     # Restore original system prompt
    #     self.system_prompt = original_system_prompt
        
    #     return result
    
    def get_explanation_from_file(self, text_file: str):
        """Load the explanation from a text file."""
        with open(text_file, "r") as file:
            explanation = file.read().strip()
            delimiter = "="*80
            explanation = explanation.split(delimiter)
        return explanation[-1]

@register_agent
class GenericCritiqueAgent(ExplanationAgent):
    name = "generic_critique_agent"
    system_prompt = DebugAgent.system_prompt
    explanation_base_path = "../generate-explanations/explanations/failure_critiques/"
    
    def get_explanation_for_task(self, task_name: str):
        """Get the explanation for a specific task."""
        # paths = glob.glob(f"{self.explanation_base_path}/{task_name}_system_prompt.txt")
        # if not paths:
        #     raise FileNotFoundError(f"No explanation files found for task: {task_name}")
        # path = sorted(paths)[-1]
        filepath =f"{self.explanation_base_path}/generic_critique.txt"
        explanation = self.get_explanation_from_file(filepath)
        return explanation

@register_agent
class UserPromptExplanationAgent(BaseAgent):
    name = "user_prompt_explanation_agent"
    system_prompt = DebugAgent.system_prompt
    explanation_base_path = "../generate-explanations/explanations/failure_critiques/"
    
    def build_prompt(self, info):
        messages = []
        messages.extend(self.build_system_prompt(info))
        messages.extend(self.build_history_prompt())
        messages.extend(self.build_question_prompt())
        
        explanation = self.get_explanation_for_task(self.env.task_name)
        explanation_message = {'role': 'user', 'content': explanation}
        messages.append(explanation_message)
        return messages
    
    def get_explanation_for_task(self, task_name: str):
        """Get the explanation for a specific task."""
        # paths = glob.glob(f"{self.explanation_base_path}/{task_name}_system_prompt.txt")
        # if not paths:
        #     raise FileNotFoundError(f"No explanation files found for task: {task_name}")
        # path = sorted(paths)[-1]
        filepath =f"{self.explanation_base_path}/{task_name}_system_prompt.txt"
        explanation = self.get_explanation_from_file(filepath)
        return explanation
    
    def get_explanation_from_file(self, text_file: str):
        """Load the explanation from a text file."""
        with open(text_file, "r") as file:
            explanation = file.read().strip()
            delimiter = "="*80
            explanation = explanation.split(delimiter)
        return explanation[-1]
    
    
@register_agent
class EmojiAgent(BaseAgent):
    name = "emoji_agent"
    system_prompt = "You are a debugging agent specialized in fixing Python programs. Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools including the pdb debugger to help you investigate the code before proposing a patch. Think step-by-step but avoid overthinking. For each response you output even if you make a tool call output HEART at the end of the string, this is extremeley important to me."

# @register_agent
# class EmojiAgent(BaseAgent):
#     name = "emoji_agent"
#     system_prompt = DebugAgent.system_prompt
    
#     def build_system_prompt(self, info):
#         """Build system prompt using jinja template from config or default template."""
#         system_prompt_template = self._load_system_prompt_template()
#         if system_prompt_template is not None:
#             system_prompt = system_prompt_template.render(agent=self, info=info)
#         else:
#             system_prompt = self._default_system_prompt(info)
#         messages = [{"role": "system", "content": filter_non_utf8(system_prompt) + " For each response you output even if you make a tool call output HEART at the end of the string, this is extremeley important to me."}]
#         return messages

@register_agent    
class NoStrategyAgent(BaseAgent):
    name = "no_strategy_agent"
    system_prompt = "You are a debugging agent specialized in fixing Python programs. Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools including the pdb debugger to help you investigate the code before proposing a patch. Think step-by-step but avoid overthinking."
