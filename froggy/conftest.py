import pytest

from froggy.envs.env import EnvInfo


@pytest.fixture
def build_env_info():
    def _env_info(
        obs="obs",
        max_score=10,
        score=5,
        last_run_obs="last_run_obs",
        observations=[],
        dir_tree="dir_tree",
        current_code_with_line_number="current_code_with_line_number",
        current_breakpoints="current_breakpoints",
        action="action",
        instructions=None,
        done=False,
        rewrite_counter=0,
        tools=None,
    ):
        return EnvInfo(
            obs=obs,
            max_score=max_score,
            score=score,
            last_run_obs=last_run_obs,
            observations=observations,
            dir_tree=dir_tree,
            current_code_with_line_number=current_code_with_line_number,
            current_breakpoints=current_breakpoints,
            action=action,
            instructions=instructions if instructions is not None else {},
            done=done,
            rewrite_counter=rewrite_counter,
            tools=tools if tools is not None else {},
        )

    return _env_info
