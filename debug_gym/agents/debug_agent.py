from debug_gym.agents.base_agent import BaseAgent, register_agent
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.llms.base import LLM
from debug_gym.llms import OpenAILLM
from debug_gym.logger import DebugGymLogger
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
class SampleAgent(BaseAgent):
    name = "sample_agent"
    system_prompt = "You are a debugging agent specialized in fixing Python programs. Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to a set of tools including the pdb debugger to help you investigate the code before proposing a patch. While the code may seem familiar to you from your training, you should not assume you know the code. Instead, you must use the pdb debugger to investigate the code and understand the potential bugs. A common debugging workflow is to 1) find suspicious files and lines (from error messages or test failures); 2) set breakpoints at suspicious places; 3) continue execution so the frame is at the breakpoint you set; 4) then print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only. You can only call one tool at a time. Do not repeat your previous action, especially if it returned tool calling errors or it resulted in information that you already know. You can think step by step to help you make the decision at every step, but you must be concise and avoid overthinking. If you are confident that you have enough information, propose a patch to fix the bugs by calling the rewrite tool. If you are not sure, continue using the pdb tool to gather more information before proposing a patch. After every rewrite, it's always a good idea to call the eval tool to execute the new code and check if it passes the tests; if it does not, the tool will return the error messages, which you can use to continue debugging. Output both your thinking process (if any) and the tool call in the response."

    def __init__(
        self,
        config: dict,
        env: RepoEnv,
        llm: LLM | None = None,
        logger: DebugGymLogger | None = None,
    ):
        super().__init__(config, env, llm, logger)
        self.judge_llm = LLM.instantiate(
            llm_name="o3",
            llm_config_file_path=config.get("llm_config_file_path"),
            logger=logger,
        )
        with open("/home/t-iwhite/debug-gym/rubrics.txt", "r") as f:
            self.rubrics = f.read().strip().split("\n")
        
        self.rubrics = [r for r in self.rubrics if r.strip()]
        
    def prepare_history_tracker_for_analysis(self):
        data = []
        for step, info in enumerate(self.history.get_all()):
            if info.action_tool_call is not None:
                tool_calls = {
                    "name": info.action_tool_call.name,
                    "arguments": info.action_tool_call.arguments,
                }
            else:
                tool_calls = None
            
            data.append({
                "step": step, 
                "role": "user" if step % 2 == 0 else "assistant", 
                "tool_calls": tool_calls,
                "content": info.action_content,
                "reasoning": info.action_reasoning,
                "observation": info.step_observation.observation,
            })
        return data

    def judge_to_rank_response(self, responses):
        """
        Takes in a set of responses and gives each response a score based on the rubrics.
        The responses are then returned in descending order of their scores.
        """
        response_scores = {}
        for response in responses:
            score = self.score_response(response)
            response_scores[response] = score

        # sort indices of response_scores by
        sorted_responses = sorted(response_scores.items(), key=lambda x: x[1], reverse=True)
        return [resp for resp, _ in sorted_responses]

    def score_response(self, response):
        """
        Takes in a response and a set of rubrics and returns a score for the response based on the rubrics.
        """
        prompt = self.make_rubric_scoring_prompt(response)
        messages = [{"role": "user", "content": prompt}]
        judge_scoring = self.judge_llm(messages, self.env.tools)
        
        # parse the response to extract the scores
        try:
            fixed_scoring = judge_scoring.replace("'", '"').replace("True", "true").replace("False", "false")
            scores = json.loads(fixed_scoring)
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse judge response: {judge_scoring}")
            scores = []
        
        #TODO: check each piece of evidence against the ground truth??
        score = sum(1 for item in scores if item.get("applies", False))
        
        return score
    
    def make_rubric_scoring_prompt(self, response):
        """
        Takes in a set of rubrics and a response and returns a prompt that can be used to score the response based on the rubrics.
        """
        
        processed_history = self.prepare_history_tracker_for_analysis()
        
        decoded_response = {
            "step": len(processed_history),
            "tool_calls": {
                "name": response.tool.name,
                "arguments": response.tool.arguments,
            },
            "content": response.response,
            "reasoning": response.reasoning_response,
        }
        processed_history.append(decoded_response)
        prompt = (
            f"Please score the following response based on the following rubrics:\n"
            f"Rubrics: {self.rubrics}\n"
            f"Response: {json.dumps(processed_history, indent=2)}\n"
            f"For each rubric item, provide a response of true or false as well as evidence in terms of the steps, for example:\n"
            "Return your answer in this format: [{'rubric': 'Use the pdb tool efficiently', 'applies': true, 'examples': [{'step': '7', 'tool_call': 'pdb'}, {'step': '8', 'tool_call': 'pdb'}]}, {'applies': false, 'evidence': []}, ...] "
            "where applies is a boolean indicating whether the critique applies to the trajectory, step is the step number that the critique refers to, and tool_call is the name of the tool that was called at that step."
        )
        return prompt 
        
    def run(self, task_name=None, debug=False):
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
                self.logger.info(f"\n{'='*20} STEP {step+1} {'='*20}\n")
                highscore = max(highscore, info.score)
                self.logger.info(
                    f"[{task_name[:10]:<10}] | Step: {step:<4} | Score: {info.score:>4}/{info.max_score:<4} ({info.score/info.max_score:.1%}) [Best: {highscore}]"
                )

                messages = self.build_prompt(info)
                # Sample 5 responses from the LLM
                sampled_responses = []
                for _ in range(5):
                    response = self.llm(messages, info.tools)
                    sampled_responses.append(response)
                
                # Rank the responses using the judge
                ranked_responses = self.judge_to_rank_response(sampled_responses)
                
                # Use the top-ranked response
                llm_response = ranked_responses[0]

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