"""Simple agent example for interacting with FreeEnv."""

from debug_gym.agents.base_agent import BaseAgent, register_agent


@register_agent
class FreeAgent(BaseAgent):
    """Minimal reasoning agent tailored for FreeEnv sessions."""

    name = "free_agent"
    system_prompt = (
        "You are assisting in an exploratory debugging session. "
        "Inspect the repository, run commands as needed, and perform targeted rewrites. "
        "Focus on understanding before editing, explain your intent briefly, then call exactly one tool."
        "If no change is required, say so and stop."
    )

    def run(self, task_name=None, debug=False):
        try:
            return super().run(task_name=task_name, debug=debug)
        except AttributeError as exc:
            error_msg = str(exc)
            sentinel = "'NoneType' object has no attribute 'max_score'"
            if sentinel not in error_msg:
                raise

            root_cause = exc.__context__ or exc.__cause__ or exc
            self.logger.error(
                "FreeAgent failed to reset the environment before receiving initial observations. "
                "Check that the configured container image exists and is accessible."
            )

            raise root_cause
