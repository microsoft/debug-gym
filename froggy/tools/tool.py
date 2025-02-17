from abc import ABC, abstractmethod


class EnvironmentTool(ABC):
    name: str = None
    action: str = None
    instructions: str = None

    def __init__(self):
        self.environment = None

    def register(self, environment):
        from froggy.envs.env import Event, RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        self.environment = environment

        # Auto-subscribe to events that have handlers
        for event in Event:
            if hasattr(self, event.handler_name):
                environment.event_hooks.subscribe(event, self)

    def is_triggered(self, action):
        # e.g. ```pdb b src/main.py:42```
        return action.startswith(self.action)

    @abstractmethod
    def use(self, action, environment) -> list[dict]:
        pass

    # TODO: import Event from froggy.envs.env
    def trigger_event(self, event: "Event", **kwargs):
        return self.environment.handle_event(event, source=self, **kwargs)
