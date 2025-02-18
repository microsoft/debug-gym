import copy

from froggy.tools.tool import EnvironmentTool
from froggy.tools.toolbox import Toolbox


@Toolbox.register()
class ReasoningTool(EnvironmentTool):
    name: str = "reasoning"
    action: str = "```reasoning"

    @property
    def instructions(self):
        assert hasattr(self, "environment")
        instruction = {
            "template": self.template,
            "description": "\n".join(
                ["Preface any action with explicit reasoning tokens.", self.description]
            ),
            "examples": self.examples,
        }
        return instruction

    @property
    def template(self):
        if self.allow_chain_action:
            return "```reasoning ... </reasoning> <next action> ```"
        else:
            return "```reasoning ... ```"

    @property
    def examples(self):
        if self.allow_chain_action:
            ex = [
                "```reasoning The execution trace points to line 43 in main.py, so I'll place a breakpoint there.</reasoning> ```pdb b 43``` ```",
                "```reasoning There's a shape mismatch that corresponds to a matrix transpose, so I'll rewrite the function to account for the transpose. </reasoning> ```rewrite 10 <c>    m = m.transpose()</c>``` ```",
            ]
        else:
            ex = [
                "```reasoning The execution trace points to line 43 in main.py, so I'll place a breakpoint there.```",
                "```reasoning There's a shape mismatch that corresponds to a matrix transpose, so I'll rewrite the function to account for the transpose.```",
            ]
        return ex

    @property
    def description(self):
        if self.allow_chain_action:
            desc = f"""You may explicitly reason about the current state and the best course of action before executing. You follow a particular reasoning style:
You break down complex problems into smaller parts and reason through them step by step, arriving at the best next action before then executing it. You should follow your reasoning with your next action. The next action should be a valid action, which follows the syntax rules defined by the tools available in the current environment. The next action cannot be another reasoning action."""
        else:
            desc = f"""You may explicitly reason about the current state and the best course of action. You follow a particular reasoning style:
You break down complex problems into smaller parts and reason through them step by step, arriving at the best next action(s). """
        return desc

    def __init__(self, allow_chain_action: bool = False):
        super().__init__()
        self.allow_chain_action = allow_chain_action
        self.success_chain_action = False

        from froggy.envs.env import EnvInfo

        self.infos_cache: EnvInfo = None

    def use(self, action):
        self.success_chain_action = False
        if self.allow_chain_action:
            obs = self.use_with_chaining(action)
        else:
            obs = self.use_without_chaining()
        return obs, [{self.name: obs}]

    def use_with_chaining(self, action):
        """Reasoning tokens are only to benefit the model, so we strip them and then pass the remainder of the action
        as a free next action.
        """
        try:
            reasoning_text, next_action = self.split_reasoning(action)
        except:
            return "SyntaxError: invalid syntax."
        if next_action.startswith(self.action):
            return "SyntaxError: invalid syntax. You cannot chain reasoning actions."
        # now execute the next action
        try:
            next_infos = self.environment.step(next_action)
            next_obs = next_infos.last_obs
        except:
            return "\n".join(
                [
                    "Error while executing the action after reasoning.",
                    "SyntaxError: invalid syntax.",
                ]
            )
        if next_obs == f"Invalid action: {action}." or next_obs.startswith(
            "Error while using tool"
        ):
            return "\n".join(
                ["Error while executing the action after reasoning.", next_obs]
            )
        self.success_chain_action = True
        self.infos_cache = copy.deepcopy(next_infos)
        return "\n".join(
            [
                "Reasoning:",
                reasoning_text,
                "Executing action:",
                next_action,
                "Next observation:",
                next_obs,
            ]
        )

    def use_without_chaining(self, action):
        try:
            reasoning_text = action.split(self.action)[1].split("```")[0].strip()
        except:
            return "SyntaxError: invalid syntax."
        return "\n".join(["Reasoning:", reasoning_text])

    def split_reasoning(self, action):
        content = action.split(self.action)[1].rsplit("```", 1)[0].strip()
        reasoning, next_action = content.split("</reasoning>", 1)
        reasoning, next_action = reasoning.strip(), next_action.strip()
        return reasoning, next_action
