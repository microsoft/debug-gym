from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict

from debug_gym.gym.entities import Event, Observation
from debug_gym.gym.terminals.terminal import UnrecoverableTerminalError
from debug_gym.gym.workspace import WorkspaceError


@dataclass
class Record:
    args: tuple
    kwargs: dict
    observation: Observation


def track_history(func):
    @wraps(func)
    def wrapper(self, environment, *args, **kwargs):
        """Decorator to track the history of tool usage.
        History does not include the environment instance (first argument).
        """
        if not hasattr(self, "history"):
            self.history = []
        observation = func(self, environment, *args, **kwargs)
        record = Record(args=args, kwargs=kwargs, observation=observation)
        self.history.append(record)
        return observation

    return wrapper


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


class EnvironmentTool(ABC):
    name: str = None
    arguments: Dict[str, Any] = None
    description: str = None
    history: list[Record] = None
    # Shell commands to run during terminal setup when this tool is used.
    # These commands will be executed before the environment is ready.
    # Example: ["apt-get update && apt-get install -y tree"]
    setup_commands: tuple[str, ...] = ()

    def __init__(self):
        self.history = []

    @track_history
    def __call__(self, *args, **kwargs) -> Observation:
        """Forwards `tool()` to the tool.use() method and
        tracks the history of tool usage."""
        try:
            return self.use(*args, **kwargs)
        except WorkspaceError as e:
            return Observation(self.name, str(e))
        except UnrecoverableTerminalError:
            # Ensure fatal terminal failures propagate so the environment can terminate the episode.
            raise
        except Exception as e:
            # Handle exceptions and return an observation
            return Observation(
                self.name, str(e)
            )  # to handle cases where the LLM hallucinates and provide invalid arguments

    def register(self, environment):
        from debug_gym.gym.envs.env import RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        # Auto-subscribe to events that have handlers
        for event in Event:
            if hasattr(self, event.handler_name):
                environment.event_hooks.subscribe(event, self)

        # Run setup commands if this tool has any and the environment is already
        # initialized (tool added after reset). Otherwise, they'll run via
        # on_env_reset when reset() is called (all tools are subscribed to ENV_RESET
        # because EnvironmentTool defines on_env_reset).
        if self.setup_commands:
            if (
                hasattr(environment, "workspace")
                and environment.workspace is not None
                and environment.workspace.working_dir is not None
            ):
                # Environment already reset, run setup commands now
                for cmd in self.setup_commands:
                    environment.terminal.run(cmd, raises=False)

    def unregister(self, environment):
        from debug_gym.gym.envs.env import RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        # Unsubscribe from all events
        for event in Event:
            if hasattr(self, event.handler_name):
                environment.event_hooks.unsubscribe(event, self)

    @abstractmethod
    def use(self, environment, action) -> Observation:
        """This method is invoked directly by `tool()` or by event handlers,
        and should be overridden by subclasses. Returns an observation which
        includes the tool's name and the result of the action.
        Don't call this method directly, use `tool()` instead to track history.
        """
        pass

    def queue_event(self, environment, event: Event, **kwargs) -> None:
        environment.queue_event(event, source=self, **kwargs)

    def on_env_reset(self, environment, **kwargs) -> Observation:
        """Reset the tool state on environment reset.
        Please call `super().on_env_reset()` if subclass overrides this method.
        """
        self.history = []

        # Run setup commands if this tool has any
        for cmd in self.setup_commands:
            environment.terminal.run(cmd, raises=False)

        return None

    def __str__(self):
        args = ", ".join(f"{k}:{v['type'][0]}" for k, v in self.arguments.items())
        return f"{self.name}({args}): {self.description.split('.')[0].strip()}."
