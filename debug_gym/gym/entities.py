from dataclasses import dataclass, field
from enum import Enum


@dataclass
class EvalDetails:
    parsed_tests: dict[str, str]
    n_parsed: int
    n_passed: int
    n_failed: int
    reward: float
    n_fail_to_pass: int = 0
    n_pass_to_pass: int = 0
    grading_report: dict = field(default_factory=dict)


@dataclass
class EvalOutput:
    success: bool
    output: str
    details: EvalDetails | None = field(default=None, compare=False)


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
    EDIT_SUCCESS = "edit_success"
    EDIT_FAIL = "edit_fail"
    SWITCH_CONTEXT = "switch_context"

    @property
    def handler_name(self) -> str:
        """Returns the method name that handles this event, e.g. `on_env_start`"""
        return f"on_{self.value}"

    @classmethod
    def list(cls):
        """Returns list of event names as strings"""
        return [event.value for event in cls]
