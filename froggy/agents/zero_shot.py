import json

from tqdm import tqdm

from froggy.agents import AgentBase
from froggy.utils import HistoryTracker, unescape


class AgentZeroShot(AgentBase):
    name: str = "zero shot"

    def __init__(self, config_dict, env, verbose=False, _uuid=None):
        super().__init__(config_dict, env, verbose, _uuid)
        self.history = HistoryTracker(self.config["memory_size"])

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

    def build_question_prompt(self):
        messages = []
        question = "Based on the instruction, the current code, the last execution output, and the history information, "
        question += "continue your debugging process using pdb commands or to propose a patch using rewrite command. "
        question += "Output a single command, nothing else. Do not repeat your previous commands unless they can provide more information."
        messages.append({"role": "user", "content": question})
        return messages

    def build_prompt(self, info):
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

            prompt = self.build_prompt(info)
            answer, token_usage = self.llm(
                prompt, info, temperature=self.config["llm_temperature"][0]
            )

            if debug:
                breakpoint()

            _, _, done, info = self.env.step(answer)
            info["token_usage"] = [
                token_usage
            ]  # in some other agents this is a list because of multi-step llm calls
            self.history.step(info)
            self.history.save_prompt_response_pairs(
                prompt_response_pairs=[(prompt, answer)]
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


class AgentZeroShot_NoPDB(AgentZeroShot):
    name: str = "zero shot no pdb"

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

    def build_question_prompt(self):
        messages = []
        question = "Based on the instruction, the current code, the last execution output, and the history information, "
        question += (
            "continue your debugging process to propose a patch using rewrite command. "
        )
        question += "Output a single command, nothing else. Do not repeat your previous commands unless they can provide more information."
        messages.append({"role": "user", "content": question})
        return messages
