import pytest

from froggy.pond.entities import Observation
from froggy.pond.envs.env import EnvInfo


@pytest.fixture
def build_env_info():
    def _env_info(
        step_observation="obs",
        all_observations=[],
        eval_observation="eval_observation",
        dir_tree="dir_tree",
        current_code_with_line_number="current_code_with_line_number",
        current_breakpoints="current_breakpoints",
        action="action",
        instructions=None,
        score=5,
        max_score=10,
        done=False,
        rewrite_counter=0,
        tools=None,
    ):
        return EnvInfo(
            step_observation == Observation("tool", step_observation),
            all_observations=all_observations,
            eval_observation=Observation("env", eval_observation),
            dir_tree=dir_tree,
            current_code_with_line_number=current_code_with_line_number,
            current_breakpoints=current_breakpoints,
            action=action,
            instructions=instructions if instructions is not None else {},
            score=score,
            max_score=max_score,
            done=done,
            rewrite_counter=rewrite_counter,
            tools=tools if tools is not None else {},
        )

    return _env_info
