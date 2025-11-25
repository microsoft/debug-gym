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
