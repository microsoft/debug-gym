from dataclasses import dataclass
from enum import Enum


@dataclass
class EvalOutput:
    success: bool
    output: str


@dataclass
class Observation:
    source: str
    observation: str

    def __str__(self):
        return self.observation


class Event(Enum):
    ENV_START = "env_start"
    ENV_RESET = "env_reset"
    ENV_STEP = "env_step"
    FILE_CHANGE = "file_change"
    REWRITE_SUCCESS = "rewrite_success"
    REWRITE_FAIL = "rewrite_fail"
    SWITCH_CONTEXT = "switch_context"

    @property
    def handler_name(self) -> str:
        """Returns the method name that handles this event, e.g. `on_env_start`"""
        return f"on_{self.value}"

    @classmethod
    def list(cls):
        """Returns list of event names as strings"""
        return [event.value for event in cls]
