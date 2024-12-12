import json

from tqdm import tqdm

from froggy.agents import AgentBase
from froggy.agents.utils import HistoryTracker
from froggy.utils import unescape


class AgentCoT(AgentBase):
    # Prompt style is inspired by https://medium.com/@ickman/instruct-making-llms-do-anything-you-want-ff4259d4b91
    name: str = "cot"

    def __init__(self, config_dict, env, verbose=False):
        super().__init__(config_dict, env, verbose)
        self.command_special_tokens = ["<NEXT_COMMAND>", "</NEXT_COMMAND>"]

    def build_cot_prompt(self):
        messages = []
        cot_prompt = [
            "Based on the instruction, the current code, the last execution output, and the history information, continue your debugging process. "
        ]
        cot_prompt += ["Let's think step by step: "]
        cot_prompt += [
            "Step 1. What information did we get from the last execution output? "
        ]
        cot_prompt += [
            "Step 2. What did we decide to investigate based on the last execution output? "
        ]
        cot_prompt += ["Step 3. What did we find so far from the investigation? "]
        cot_prompt += ["Step 4. What is remaining unclear about the code? "]
        cot_prompt += [
            "Step 5. What is our plan to investigate the remaining unclear part? "
        ]
        cot_prompt += [
            f"Step 6. Concretely, what is the next command to execute? For this step, put your answer inside special tokens {self.command_special_tokens[0]} and {self.command_special_tokens[1]}. Make sure to follow the commands' syntax such as ``` delimiters. Make sure the answer is a single command, nothing else. Do not repeat your previous commands unless they can provide more information. "
        ]
        cot_prompt += [
            "Now, state each step above and show your work for performing that step. "
        ]

        messages.append({"role": "user", "content": "\n".join(cot_prompt)})
        return messages

    def build_prompt(self, info):
        messages = self.build_system_prompt(info)
        messages.extend(self.build_history_prompt())
        messages.extend(self.build_cot_prompt())
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

            prompt = self.build_prompt(info)
            llm_output, token_usage = self.llm(
                prompt,
                info,
                temperature=self.config["llm_temperature"][0],
            )
            # parse the answer to get the command
            if (
                self.command_special_tokens[0] in llm_output
                and self.command_special_tokens[1] in llm_output
            ):
                answer = (
                    llm_output.split(self.command_special_tokens[0])[1]
                    .split(self.command_special_tokens[1])[0]
                    .strip()
                )
            else:
                answer = llm_output  # if the special tokens are not used, just use the entire output as the answer

            if debug:
                breakpoint()

            _, _, done, info = self.env.step(answer)
            info["token_usage"] = [
                token_usage
            ]  # in some other agents this is a list because of multi-step llm calls
            self.history.step(info)
            self.history.save_prompt_response_pairs(
                prompt_response_pairs=[(prompt, llm_output, answer)]
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

    def __init__(self, config_dict, env, verbose=False):
        super().__init__(config_dict, env, verbose)
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
        cot_prompt += ["Let's think step by step: "]
        cot_prompt += [
            "Step 1. What information did we get from the last execution output? "
        ]
        cot_prompt += [
            "Step 2. Which lines of the code might have caused the errors in the last execution output? "
        ]
        cot_prompt += ["Step 3. How should we fix the errors in the code? "]
        cot_prompt += [
            f"Step 4. Concretely, what is the next command to execute? For this step, put your answer inside special tokens {self.command_special_tokens[0]} and {self.command_special_tokens[1]}. Make sure to follow the commands' syntax such as ``` delimiters. Make sure the answer is a single command, nothing else. Do not repeat your previous commands unless they can provide more information. "
        ]
        cot_prompt += [
            "Now, state each step above and show your work for performing that step. "
        ]

        messages.append({"role": "user", "content": "\n".join(cot_prompt)})
        return messages
