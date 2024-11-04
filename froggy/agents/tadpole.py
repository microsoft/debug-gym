import json

from tqdm import tqdm

from froggy.agents import AgentBase
from froggy.utils import HistoryTracker, unescape


class AgentTadpole(AgentBase):
    """
    Tadpole is a simple hierarchical agent that consists of a task decomposition module and a command generation module.
    The task decomposition module is responsible for determining the current subgoal (e.g., what are we trying to investigate now?).
    Guided by the subgoal, the command generation module generates a command towards achieving the subgoal.
    In such way, the command generation module can focus on generating a command that is relevant to the current subgoal.
    """

    name: str = "tadpole"

    def __init__(self, config_dict, env, verbose=False, _uuid=None):
        super().__init__(config_dict, env, verbose, _uuid)
        self.history = HistoryTracker(self.config["memory_size"])
        self.current_subgoal = None

    def _build_history_conversation(self):
        _history = self.history.get()
        _messages = []
        for history_info in _history:
            if history_info["action"] is not None:
                _messages.append(
                    {"role": "assistant", "content": f"{history_info["action"]}"}
                )
            _messages.append({"role": "user", "content": f"{history_info["obs"]}"})
        return _messages

    def _build_history_non_conversation(self):
        _history = self.history.get()
        _history_prompt = []
        for _i, history_info in enumerate(_history):
            _m = {
                "step": _i,
                "command": (
                    None if history_info["action"] is None else history_info["action"]
                ),
                "stdout": history_info["obs"],
            }
            _history_prompt.append(_m)
        return _history_prompt

    def build_system_prompt(self, info):
        system_prompt = {}
        system_prompt["Overall task"] = (
            "Your goal is to debug a Python program to make sure it can pass a set of test functions. You have access to the pdb debugger tools, you can use them to investigate the code, set breakpoints, and print necessary values to identify the bugs. Once you have gained enough information, propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only."
        )
        system_prompt["Instructions"] = info["instructions"]
        system_prompt["Repo directory tree"] = info["dir_tree"]
        system_prompt["Editable files"] = info["editable_files"]
        system_prompt["Current code in view"] = info["current_code_with_line_number"]
        system_prompt["Current breakpoints"] = info["current_breakpoints"]
        system_prompt["Last execution output"] = info["last_run_obs"]
        system_prompt = unescape(json.dumps(system_prompt, indent=4))
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        return messages

    def build_history_prompt(self):
        messages = []
        if self.config["use_conversational_prompt"] is True:
            conversation_history = self._build_history_conversation()
            messages.append(
                {
                    "role": "user",
                    "content": f"History of command and terminal outputs (the last {(len(conversation_history) + 1) // 2} steps):",
                }
            )
            messages.extend(conversation_history)
        else:
            history_prompt = self._build_history_non_conversation()
            prompt = [
                f"History of command and terminal outputs (the last {len(history_prompt)} steps):"
            ]
            prompt += ["\n" + unescape(json.dumps(history_prompt, indent=4)) + "\n"]
            messages.append({"role": "user", "content": "\n".join(prompt)})
        return messages

    def build_task_decomposer_prompt(self):
        what_is_a_subgoal = []
        what_is_a_subgoal += [
            "A subgoal is a smaller task that helps you focus on a specific part of the debugging process, such as identifying the bug in a specific function, understanding the control flow of the code, or verifying the correctness of a specific logic. "
        ]
        what_is_a_subgoal += [
            "A subgoal should be specific, actionable, and achievable within a few steps of actions. "
        ]
        what_is_a_subgoal += [
            "However, the subgoal should not be too low-level, such as printing a single variable or setting a single breakpoint. "
        ]
        what_is_a_subgoal += [
            "Let's identify the current subgoal based on the information we have. "
        ]
        what_is_a_subgoal += [
            "Generate the most immediate subgoal that you think is necessary to identify and fix the bug. "
        ]
        what_is_a_subgoal += [
            "The subgoal should be a short paragraph answering the following questions: "
        ]
        what_is_a_subgoal += [
            "1. What specific part of the code are you focusing on? Be specific about the function, class, or logic. "
        ]
        what_is_a_subgoal += [
            "2. What information do you need to gather during this subgoal? Be specific about the variables, values, or control flow. "
        ]
        what_is_a_subgoal += [
            "3. What actions do you plan to take to achieve this subgoal? Here you don't need to be accurate, but you should have a rough plan. "
        ]
        what_is_a_subgoal = "\n".join(what_is_a_subgoal)

        messages = []
        prompt = [
            "Based on the instruction, the current code, the last execution output, and the history information, continue your debugging process. "
        ]
        if self.current_subgoal is None:
            prompt += [
                "To help you think step by step, let's decompose the task into smaller subgoals. "
            ]
            prompt += [what_is_a_subgoal]
            prompt += ["Output a single paragraph as the subgoal, nothing else. "]
        else:
            prompt += ["In the previous steps, you identified the following subgoal: "]
            prompt += [self.current_subgoal]
            prompt += [
                "Based on the information you gathered so far, do you want to continue with the same subgoal or identify a new subgoal? "
            ]
            prompt += [
                "If you think the current subgoal is not completed yet, generate a single word 'continue' (and nothing else) to proceed with the same subgoal, so you can continue to investigate. "
            ]
            prompt += [
                "If you think the current subgoal is completed or you want to change the focus, generate a new subgoal following the same structure as before. "
            ]

        messages.append({"role": "user", "content": "\n".join(prompt)})
        return messages

    def build_prompt_step_1(self, info):
        messages = self.build_system_prompt(info)
        messages.extend(self.build_history_prompt())
        messages.extend(self.build_task_decomposer_prompt())
        return messages

    def build_question_prompt(self):
        messages = []
        question_prompt = [
            "To help you think step by step, we have decomposed the task into smaller subgoals. "
        ]
        question_prompt += ["Let's focus on the current subgoal only:"]
        question_prompt += [self.current_subgoal]
        question_prompt += [
            "Note that the subgoal could contain some rough plans, you can use them as a reference but you don't need to follow them strictly, you need to adapt based on the information you gather. "
        ]
        question_prompt += [
            "Based on the information above, what is the best next command?"
        ]
        question_prompt += [
            "Output a single command, nothing else. Do not repeat your previous commands unless they can provide more information. "
        ]
        messages.append({"role": "user", "content": "\n".join(question_prompt)})
        return messages

    def build_prompt_step_2(self, info):
        messages = self.build_system_prompt(info)
        messages.extend(self.build_history_prompt())
        messages.extend(self.build_question_prompt())
        return messages

    def run(self, task_name=None, debug=False):

        self.history.reset()
        _, info = self.env.reset(options={"task_name": task_name})
        self.history.step(info)

        if info["done"] is True:
            # msg = "Environment started with entrypoint passing without errors."
            return True

        done = False
        highscore = info["score"]
        pbar = tqdm(
            total=self.config["max_steps"],
            desc=f"Debugging inside {self.env.working_dir}",
            leave=True,
        )
        for step in range(self.config["max_steps"]):
            highscore = max(highscore, info["score"])
            pbar.set_postfix_str(
                f"Score: {info['score']}/{info['max_score']} ({info['score']/info['max_score']:.1%}) [Best: {highscore}]".format(
                    info["score"]
                )
            )
            prompt_response_pairs = []
            token_usage_list = []
            for _reasoning_step in range(2):
                if _reasoning_step == 0:
                    prompt = self.build_prompt_step_1(info)
                    answer, token_usage = self.llm(
                        prompt,
                        info,
                        temperature=self.config["llm_temperature"][_reasoning_step],
                    )
                    answer = answer.strip()
                    if self.current_subgoal is None:
                        self.current_subgoal = answer
                    else:
                        if (
                            answer.lower() == "continue"
                            or answer.lower() == "'continue'"
                        ):
                            pass
                        else:
                            self.current_subgoal = answer
                else:
                    prompt = self.build_prompt_step_2(info)
                    answer, token_usage = self.llm(
                        prompt,
                        info,
                        temperature=self.config["llm_temperature"][_reasoning_step],
                    )
                prompt_response_pairs.append((prompt, answer))
                token_usage_list.append(token_usage)

                if debug:
                    breakpoint()

            _, _, done, info = self.env.step(answer)
            info["token_usage"] = token_usage_list
            self.history.step(info)
            self.history.save_prompt_response_pairs(
                prompt_response_pairs=prompt_response_pairs
            )

            pbar.update()
            if done or info["rewrite_counter"] >= self.config["max_rewrite_steps"]:
                pbar.set_postfix_str(
                    f"Score: {info['score']}/{info['max_score']} ({info['score']/info['max_score']:.1%})".format(
                        info["score"]
                    )
                )
                break
        return done
