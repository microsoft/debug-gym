"""Simple agent example for interacting with FreeEnv."""

from debug_gym.agents.base_agent import BaseAgent, register_agent


@register_agent
class FreeAgent(BaseAgent):
    """Minimal reasoning agent tailored for FreeEnv sessions."""

    name = "free_agent"
    system_prompt = (
        "You are assisting in an exploratory debugging session inside an open-ended container.\n"
        "There is no preset task—inspect the workspace, run commands to gather context, and edit only when necessary.\n"
        "Workflow guidance:\n"
        "  1. List the repository tree to understand the project layout.\n"
        "  2. Read relevant files before attempting rewrites.\n"
        "  3. Explain your intent briefly, then call exactly one tool.\n"
        "  4. Prefer targeted rewrites; avoid sweeping edits unless justified.\n"
        "  5. If you determine no action is required, say so and stop."
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
