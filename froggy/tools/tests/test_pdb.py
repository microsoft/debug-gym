from unittest.mock import MagicMock
import pytest

from froggy.envs.env import RepoEnv
from froggy.terminal import DockerTerminal, Terminal
from froggy.tools.pdb import PDBTool


# TODO: move to conftest
import subprocess

if_docker_running = pytest.mark.skipif(
    not subprocess.check_output("docker ps", shell=True),
    reason="Docker not running",
)


@pytest.fixture
def setup_test_repo():
    def _setup_test_repo(base_dir):
        working_dir = base_dir / "tests_pdb"
        working_dir.mkdir()
        with working_dir.joinpath("test_pass.py").open("w") as f:
            f.write("def test_pass():\n    assert True")
        with working_dir.joinpath("test_fail.py").open("w") as f:
            f.write("def test_fail():\n    assert False")
        return working_dir

    return _setup_test_repo


def test_pdb_use(tmp_path, setup_test_repo):
    tests_path = str(setup_test_repo(tmp_path))
    terminal = Terminal()
    environment = RepoEnv(path=tests_path, terminal=terminal)
    pdb = PDBTool()
    pdb.register(environment)
    initial_output = pdb.start_pdb(terminal)
    assert """The pytest entry point.""" in initial_output
    assert "(Pdb)" in initial_output
    output = pdb.use("pdb l")
    assert """The pytest entry point.""" in output
    assert "(Pdb)" in output
    output = pdb.use("pdb c")
    assert "1 failed, 1 passed" in pdb.pdb_obs
    assert "test_fail.py::test_fail FAILED" in pdb.pdb_obs
    assert "test_pass.py::test_pass PASSED" in pdb.pdb_obs
    assert "Reached the end of the file. Restarting the debugging session." in output
    assert "(Pdb)" in output


@if_docker_running
def test_pdb_use_docker_terminal(tmp_path, setup_test_repo):
    """Test PDBTool similar to test_pdb_use but using DockerTerminal"""
    tests_path = str(setup_test_repo(tmp_path))
    terminal = DockerTerminal(
        base_image="python:3.12-slim",
        setup_commands=["pip install pytest"],
    )
    environment = RepoEnv(path=tests_path, terminal=terminal)
    pdb = PDBTool()
    pdb.register(environment)
    pdb_cmd = f"python -m pdb -m pytest -sv ."
    pdb.start_pdb(terminal, pdb_cmd)

    output = pdb.use("pdb l")
    assert """The pytest entry point.""" in output
    assert "(Pdb)" in output
    output = pdb.use("pdb c")
    assert "1 failed, 1 passed" in pdb.pdb_obs
    assert "test_fail.py::test_fail FAILED" in pdb.pdb_obs
    assert "test_pass.py::test_pass PASSED" in pdb.pdb_obs
    assert "Reached the end of the file. Restarting the debugging session." in output
    assert "(Pdb)" in output


@if_docker_running
def test_pdb_use_docker_terminal_no_env(tmp_path, setup_test_repo):
    """Test pdb in docker with no dependencies from RepoEnv"""
    tests_path = str(setup_test_repo(tmp_path))
    terminal = DockerTerminal(
        working_dir=tests_path,
        base_image="python:3.12-slim",
        setup_commands=["pip install pytest"],
        volumes={tests_path: {"bind": tests_path, "mode": "rw"}},
    )
    environment = MagicMock()
    environment.working_dir = tests_path
    pdb = PDBTool()
    pdb.environment = environment
    pdb_cmd = f"python -m pdb -m pytest -sv ."
    pdb.start_pdb(terminal, pdb_cmd)

    output = pdb.use("pdb l")
    assert """The pytest entry point.""" in output
    assert "(Pdb)" in output
    output = pdb.use("pdb c")
    assert "1 failed, 1 passed" in pdb.pdb_obs
    assert "test_fail.py::test_fail FAILED" in pdb.pdb_obs
    assert "test_pass.py::test_pass PASSED" in pdb.pdb_obs
    assert "Reached the end of the file. Restarting the debugging session." in output
    assert "(Pdb)" in output
