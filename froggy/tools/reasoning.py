from froggy.tools import EnvironmentTool


class ReasoningTool(EnvironmentTool):
    name: str = "reasoning"
    action: str = "<reasoning>"
    description: str = "Preface any action with explicit reasoning tokens."

    @property
    def instructions(self):
        assert hasattr(self, "environment")
        instruction = {
            "template": "<reasoning> ... </reasoning>",
            "description": """You may explicitly reason about the current state and the best next action before taking it. You follow a particular reasoning style: 
You break down complex problems into smaller parts and reason through them step by step, arriving at the best next action before then executing it. You should follow your reasoning 
with your next action.""",
            "examples": [
                "<reasoning> The execution trace points to line 43 in main.py, so I'll place a breakpoint there.</reasoning> ```pdb b 43",
                "<reasoning> There's a shape mismatch that corresponds to a matrix transpose, so I'll rewrite the function to account for the transpose. </reasoning> ```rewrite ....",
            ],
        }
        return instruction

    def register(self, environment):
        from autopdb.envs import RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        self.environment = environment

    def is_triggered(self, action):
        return action.startswith(self.action)

    def use(self, action):
        """Reasoning tokens are only to benefit the model, so we strip them and then pass the remainder of the action
        as a free next action.
        """
        remaining_action = self.remove_reasoning(action)
        # now execute the next action
        next_action_obs = self.environment.step(remaining_action)
        if next_action_obs == f"Invalid action: {action}.":
            next_action_obs == f"You must provide a valid action after your reasoning. Found invalid action: {action}."
        return next_action_obs[0]

    def remove_reasoning(self, action):
        items = action.split("</reasoning>")
        return " ".join(items[1:]).lstrip()