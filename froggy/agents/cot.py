import json

from tqdm import tqdm

from froggy.agents import AgentBase
from froggy.utils import HistoryTracker, unescape


class AgentCoT(AgentBase):
    name: str = "cot"

    def __init__(self, config_dict, env, verbose=False, _uuid=None):
        super().__init__(config_dict, env, verbose, _uuid)
        self.history = HistoryTracker(self.config["memory_size"])

    def build_cot_prompt(self):
        messages = []
        cot_prompt = [
            "Based on the instruction, the current code, the last execution output, and the history information, continue your debugging process. "
        ]
        cot_prompt += ["Let's think step by step using the following questions: "]
        cot_prompt += [
            "1. What information did we get from the last execution output? "
        ]
        cot_prompt += [
            "2. What did we decide to investigate based on the last execution output? "
        ]
        cot_prompt += ["3. What did we find so far from the investigation? "]
        cot_prompt += ["4. What is remaining unclear about the code? "]
        cot_prompt += [
            "5. What is our plan to investigate the remaining unclear part? "
        ]

        messages.append({"role": "user", "content": "\n".join(cot_prompt)})
        return messages

    def build_prompt_step_1(self, info):
        messages = self.build_system_prompt(info)
        messages.extend(self.build_history_prompt())
        messages.extend(self.build_cot_prompt())
        return messages

    def fill_in_cot_response(self, response):
        if self.config["use_conversational_prompt"] is True:
            return [{"role": "assistant", "content": response}]
        else:
            return [
                {"role": "user", "content": "\n".join(["Your response: ", response])}
            ]

    def build_question_prompt(self):
        messages = []
        question_prompt = [
            "Based on our retrospective above, what is the best next command?"
        ]
        question_prompt += [
            "Output a single command, nothing else. Do not repeat your previous commands unless they can provide more information. "
        ]
        messages.append({"role": "user", "content": "\n".join(question_prompt)})
        return messages

    def build_prompt_step_2(self, info, response):
        messages = self.build_system_prompt(info)
        messages.extend(self.build_history_prompt())
        messages.extend(self.build_cot_prompt())
        messages.extend(self.fill_in_cot_response(response))
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
                else:
                    prompt = self.build_prompt_step_2(info, answer)
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


class AgentCoT_NoPDB(AgentCoT):
    name: str = "cot no pdb"

    def __init__(self, config_dict, env, verbose=False, _uuid=None):
        super().__init__(config_dict, env, verbose, _uuid)
        self.history = HistoryTracker(self.config["memory_size"])

    def build_system_prompt(self, info):
        system_prompt = {}
        system_prompt["Overall task"] = (
            "Your goal is to debug a Python program to make sure it can pass a set of test functions. You need to propose a rewriting patch to fix the bugs. Avoid rewriting the entire code, focus on the bugs only."
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

    def build_cot_prompt(self):
        messages = []
        cot_prompt = [
            "Based on the instruction, the current code, the last execution output, and the history information, continue your debugging process. "
        ]
        cot_prompt += ["Let's think step by step using the following questions: "]
        cot_prompt += [
            "1. What information did we get from the last execution output? "
        ]
        cot_prompt += [
            "2. Which lines of the code might have caused the errors in the last execution output? "
        ]
        cot_prompt += ["3. How should we fix the errors in the code? "]

        messages.append({"role": "user", "content": "\n".join(cot_prompt)})
        return messages
