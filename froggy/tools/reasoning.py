import copy

from froggy.tools.tool import EnvironmentTool
from froggy.tools.toolbox import Toolbox


@Toolbox.register()
class ReasoningTool(EnvironmentTool):
    name: str = "reasoning"

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
            return "reasoning(reasoning_text: str, next_action: str)"
        else:
            return "reasoning(reasoning_text: str)"

    @property
    def examples(self):
        if self.allow_chain_action:
            ex = [
                """reasoning(reasoning_text="The execution trace points to line 43 in main.py, so I'll place a breakpoint there.", next_action="pdb(b 43)")""",
                """reasoning(reasoning_text="There's a shape mismatch that corresponds to a matrix transpose, so I'll rewrite the function to account for the transpose.", next_action="rewrite(start=10, new_code='    m = m.transpose()')")""",
            ]
        else:
            ex = [
                """reasoning(reasoning_text="The execution trace points to line 43 in main.py, so I'll place a breakpoint there.")""",
                """reasoning(reasoning_text="There's a shape mismatch that corresponds to a matrix transpose, so I'll rewrite the function to account for the transpose.")""",
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

    def use(self, *args, **kwargs):
        self.success_chain_action = False
        if self.allow_chain_action:
            return self.use_with_chaining(*args, **kwargs)
        else:
            return self.use_without_chaining(*args, **kwargs)

    def use_with_chaining(self, reasoning_text: str, next_action: str):
        """Reasoning tokens are only to benefit the model, so we strip them and then pass the remainder of the action
        as a free next action.
        """
        if next_action.startswith(self.name):
            return "SyntaxError: invalid syntax. You cannot chain reasoning actions."
        # now execute the next action
        try:
            next_infos = self.environment.step(next_action)
            next_obs = next_infos.obs
        except:
            return "\n".join(
                [
                    "Error while executing the action after reasoning.",
                    "SyntaxError: invalid syntax.",
                ]
            )
        if next_obs.startswith("Invalid action:") or next_obs.startswith(
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

    def use_without_chaining(self, reasoning_text: str):
        return "\n".join(["Reasoning:", reasoning_text])
