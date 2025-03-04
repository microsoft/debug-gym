from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps

from debug_gym.gym.entities import Event, Observation


@dataclass
class Record:
    args: tuple
    kwargs: dict
    observation: Observation


def track_history(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "history"):
            self.history = []
        observation = func(self, *args, **kwargs)
        record = Record(args=args, kwargs=kwargs, observation=observation)
        self.history.append(record)
        return observation

    return wrapper


class EnvironmentTool(ABC):
    name: str = None
    instructions: str = None
    history: list[Record] = None

    def __init__(self):
        self.environment = None
        self.history = []

    @track_history
    def __call__(self, action=None, **kwargs) -> Observation:
        """Forwards `tool()` to the tool.use() method and
        tracks the history of tool usage."""
        return self.use(action, **kwargs)

    def register(self, environment):
        from debug_gym.gym.envs.env import RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        self.environment = environment

        # Auto-subscribe to events that have handlers
        for event in Event:
            if hasattr(self, event.handler_name):
                environment.event_hooks.subscribe(event, self)

    @abstractmethod
    def use(self, action) -> Observation:
        """This method is invoked directly by `tool()` or by event handlers,
        and should be overridden by subclasses. Returns an observation which
        includes the tool's name and the result of the action.
        Don't call this method directly, use `tool()` instead to track history.
        """
        pass

    def queue_event(self, event: Event, **kwargs) -> None:
        self.environment.queue_event(event, source=self, **kwargs)

    def on_env_reset(self, **kwargs) -> Observation:
        """Reset the tool state on environment reset.
        Please call `super().on_env_reset()` if subclass overrides this method.
        """
        self.history = []
        return None
