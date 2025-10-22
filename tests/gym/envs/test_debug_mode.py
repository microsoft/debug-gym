import json
import types

import pytest

from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.swe_bench import SWEBenchEnv


class DummyTerminal:
    """Minimal stand-in for a real Terminal to unit test SWEBenchEnv logic.

    Captures commands that would have been sent to a container / shell so we can
    assert on side-effects (e.g., applying the hidden test patch when debug_mode=False).
    """

    def __init__(self):
        self.commands: list[str] = []
        self.env_vars: dict[str, str] = {}
        self.setup_commands: list[str] = []
        self.session_commands: list[str] = []
        self.task_name = ""
        self.base_image = ""

    # Signature compatibility with real Terminal.run
    def run(
        self, command: str, timeout=None, strip_output=True, raises=False
    ):  # noqa: D401
        self.commands.append(command)
        # Simulate running tests: produce output that parser would consider passing.
        if "pytest" in command:
            return True, "ALL TESTS PASSED"
        return True, ""

    def close(self):  # pragma: no cover - nothing to clean
        pass


class DummyDataset:
    """Simple object mimicking the HF dataset row/column access pattern used in SWEBenchEnv."""

    def __init__(self, rows):
        self._rows = rows
        self._columns = {}
        if rows:
            for k in rows[0].keys():
                self._columns[k] = [r[k] for r in rows]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._rows[idx]
        return self._columns[idx]


@pytest.fixture()
def patched_env(monkeypatch):
    """Provide a SWEBenchEnv instance with heavy external dependencies stubbed out.

    We patch load_dataset & setup_terminal so that no network / docker calls are made.
    """

    # Minimal single task identifiers
    instance_id = "dummy__repo-1"
    repo_full = "dummy/dummy"

    row = {
        "instance_id": instance_id,
        "repo": repo_full,
        "version": "v1",
        "patch": "PATCH_CONTENT",  # gold patch (unused by these tests)
        "test_patch": "TEST_PATCH_CONTENT",  # hidden benchmark tests
        "FAIL_TO_PASS": json.dumps(["test_a", "test_b"]),
        "PASS_TO_PASS": json.dumps(["test_c"]),
        "problem_statement": "Fix the bug so tests pass.",
        "base_commit": "abc123",
    }

    def load_dataset_stub(self, problems):  # noqa: D401
        self.ds = DummyDataset([row])
        return {instance_id: 0}

    def setup_terminal_stub(self):  # noqa: D401
        # Skip real container setup; just simulate a git repo being initialized so that
        # later git commands are 'valid'.
        self.terminal.run("git init -b main")
        self.terminal.run("git add . || true")
        self.terminal.run("git commit -am 'Initial' || true")

    # Patch heavy methods on the existing class
    monkeypatch.setattr(SWEBenchEnv, "load_dataset", load_dataset_stub, raising=True)
    monkeypatch.setattr(
        SWEBenchEnv, "setup_terminal", setup_terminal_stub, raising=True
    )

    # Replace the DockerTerminal / KubernetesTerminal symbols with DummyTerminal so the
    # isinstance check inside SWEBenchEnv.__init__ passes without pulling real containers.
    monkeypatch.setattr(
        "debug_gym.gym.envs.swe_bench.DockerTerminal", DummyTerminal, raising=True
    )
    monkeypatch.setattr(
        "debug_gym.gym.envs.swe_bench.KubernetesTerminal", DummyTerminal, raising=True
    )

    # Provide minimal mapping/lookups used in setup_task (patch both the env module and harness constants)
    minimal_specs = {repo_full: {"v1": {"test_cmd": "pytest -q --tb=no"}}}
    monkeypatch.setattr(
        "debug_gym.gym.envs.swe_bench.MAP_REPO_VERSION_TO_SPECS",
        minimal_specs,
        raising=True,
    )
    monkeypatch.setattr(
        "swebench.harness.constants.MAP_REPO_VERSION_TO_SPECS",
        minimal_specs,
        raising=True,
    )

    # Stub make_test_spec to avoid depending on full SWEbench harness internals
    class _MinimalSpec:
        def __init__(self):
            self.instance_image_key = "dummy_image"

    def make_test_spec_stub(_row):  # noqa: D401
        return _MinimalSpec()

    monkeypatch.setattr(
        "debug_gym.gym.envs.swe_bench.make_test_spec", make_test_spec_stub, raising=True
    )
    monkeypatch.setattr(
        "swebench.harness.test_spec.test_spec.make_test_spec",
        make_test_spec_stub,
        raising=True,
    )

    # get_test_directives returns the test files list (these are later checked out during final eval)
    monkeypatch.setattr(
        "debug_gym.gym.envs.swe_bench.get_test_directives",
        lambda _row: ["tests/test_dummy.py"],
        raising=True,
    )

    # Parser map: always mark both fail_to_pass tests as PASSED so score==len(fail_to_pass)
    def parser(_repo_output, _test_spec):  # noqa: D401
        # Return statuses matching TestStatus.PASSED.value
        return {"test_a": "PASSED", "test_b": "PASSED"}

    monkeypatch.setattr(
        "debug_gym.gym.envs.swe_bench.MAP_REPO_TO_PARSER",
        {repo_full: parser},
        raising=True,
    )

    # Instantiate environment with dummy terminal
    env = SWEBenchEnv(split="test", terminal=DummyTerminal())
    env.setup_task(instance_id)
    env.setup_workspace()
    # Provide missing attributes expected later
    env.max_score = len(json.loads(row["FAIL_TO_PASS"]))
    env.done = (
        False  # ensure attribute exists even before first eval when in debug mode
    )
    return env


def test_eval_debug_mode_false_applies_test_patch_and_sets_done(
    patched_env, monkeypatch
):
    env = patched_env
    env.debug_mode = False

    # Sanity: ensure test patch command not yet run
    assert not any("git apply" in c for c in env.terminal.commands)

    result = env.eval()
    assert isinstance(result, EvalOutput)

    # Hidden test patch should have been applied
    apply_cmds = [
        c
        for c in env.terminal.commands
        if "git apply" in c and "TEST_PATCH_CONTENT" in c
    ]
    assert (
        apply_cmds
    ), "Expected the hidden test patch to be applied during final eval when debug_mode=False"

    # Score should equal number of fail_to_pass tests and done should be True
    assert env.score == len(env.fail_to_pass)
    assert env.done is True


def test_eval_debug_mode_true_does_not_apply_test_patch(patched_env):
    env = patched_env
    env.debug_mode = True
    env.eval()
    # Ensure no git apply of test patch occurred
    assert not any(
        "git apply" in c and "TEST_PATCH_CONTENT" in c for c in env.terminal.commands
    )
    # In debug mode we do not auto-complete the task
    assert env.done is False


def test_submit_tool_invokes_eval_and_returns_output(monkeypatch, patched_env):
    from debug_gym.gym.tools.submit import SubmitTool

    env = patched_env
    env.debug_mode = False
    tool = SubmitTool()

    obs = tool.use(env)
    assert "ALL TESTS PASSED" in obs.observation
    # The eval path should have applied the hidden test patch
    assert any(
        "git apply" in c and "TEST_PATCH_CONTENT" in c for c in env.terminal.commands
    )
    assert env.done is True


def test_eval_final_mode_resets_modified_test_directives_before_applying_patch_and_running(
    patched_env,
):
    """Ensure that in non-debug (final) mode the environment issues a git checkout
    of the benchmark test directive paths BEFORE applying the hidden test patch and
    BEFORE executing pytest. This guarantees agent-authored modifications to those
    files aren't influencing the official evaluation.
    """
    env = patched_env
    env.debug_mode = False

    # Clear any setup commands captured during fixture initialization so ordering is easier to assert.
    env.terminal.commands.clear()

    env.eval()

    cmds = env.terminal.commands
    # Collect indices of key operations
    try:
        checkout_idx = next(
            i for i, c in enumerate(cmds) if c.startswith("git checkout -- ")
        )
    except StopIteration:  # pragma: no cover - test will fail below
        checkout_idx = -1
    apply_idx = next(
        i for i, c in enumerate(cmds) if "git apply" in c and "TEST_PATCH_CONTENT" in c
    )
    pytest_idx = next(i for i, c in enumerate(cmds) if "pytest" in c)

    # Assertions: checkout happens first, then patch apply, then pytest run
    assert (
        checkout_idx != -1
    ), "Expected a git checkout command for test directive files"
    assert (
        checkout_idx < apply_idx < pytest_idx
    ), f"Order incorrect:\nCommands: {cmds}\ncheckout_idx={checkout_idx}, apply_idx={apply_idx}, pytest_idx={pytest_idx}"

    # Ensure the exact test directive path is included in the checkout command
    checkout_cmd = cmds[checkout_idx]
    assert (
        "tests/test_dummy.py" in checkout_cmd
    ), "Test directive file not present in git checkout command"
