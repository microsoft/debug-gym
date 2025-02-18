from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps


@dataclass
class History:
    kwargs: dict
    observation: str


def track_history(func):
    @wraps(func)
    def wrapper(self, **kwargs):
        if not hasattr(self, "_use_history"):
            self._use_history = []

        observation, chain = func(self, **kwargs)

        record = History(kwargs=kwargs, observation=observation)

        self.history.append(record)
        return observation

    return wrapper


class EnvironmentTool(ABC):
    name: str = None
    action: str = None
    instructions: str = None
    history: list[History] = None

    def __init__(self):
        self.environment = None
        self.history = []

    @track_history
    def __call__(self, action):
        return self.use(action)

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
    def use(self, action) -> tuple[str, list[dict]]:
        """This method is invoked directly by `tool()` or by event handlers.
        It returns the primary observation from using the tool along with a list of
        additional observations, which may include chained observations triggered by handlers.
        Note that the list should also include the primary observation.
        """
        pass

    # TODO: import Event from froggy.envs.env
    def trigger_event(self, event: "Event", **kwargs):
        return self.environment.handle_event(event, source=self, **kwargs)

    def on_env_reset(self, **kwargs) -> list[dict]:
        """Reset the tool state on environment reset.
        Please call `super().on_env_reset()` if subclass overrides this method.
        """

        self.history = []
        return []
