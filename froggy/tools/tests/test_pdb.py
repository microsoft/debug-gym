import subprocess
import unittest
from unittest.mock import MagicMock, patch

import pytest

from froggy.envs.env import RepoEnv
from froggy.terminal import DockerTerminal, Terminal
from froggy.tools.pdb import PDBTool
from froggy.utils import TimeoutException

if_docker_running = pytest.mark.skipif(
    not subprocess.check_output(["docker", "ps"]),
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


class TestPDBTool(unittest.TestCase):

    def setUp(self):
        self.env = MagicMock(spec=RepoEnv)
        self.environment = MagicMock()
        self.env.working_dir = "/path/to/repo"
        self.terminal = MagicMock(spec=Terminal)
        self.pdb_tool = PDBTool()
        self.pdb_tool.environment = self.environment
        self.pdb_tool.breakpoints_state = {
            "file1.py|||10": "b file1.py:10",
            "file1.py|||20": "b file1.py:20",
            "file1.py|||30": "b file1.py:30",
            "file2.py|||15": "b file2.py:15",
        }

    def test_initialization(self):
        self.assertIsNone(self.pdb_tool.master)
        self.assertEqual(self.pdb_tool.pdb_obs, "")
        self.assertFalse(self.pdb_tool.persistent_breakpoints)
        self.assertTrue(self.pdb_tool.auto_list)
        self.assertIsNone(self.pdb_tool.current_frame_file)
        self.assertIsNone(self.pdb_tool._terminal)

    def test_register(self):
        self.pdb_tool.register(self.env)
        self.assertEqual(self.pdb_tool.environment, self.env)

    def test_register_invalid_environment(self):
        with self.assertRaises(ValueError):
            self.pdb_tool.register(MagicMock())

    def test_terminal_getter_setter(self):
        with self.assertRaises(ValueError):
            _ = self.pdb_tool.terminal

        self.pdb_tool.terminal = self.terminal
        self.assertEqual(self.pdb_tool.terminal, self.terminal)

    @patch.object(PDBTool, "interact_with_pdb")
    def test_start_pdb(self, mock_interact_with_pdb):
        mock_interact_with_pdb.return_value = "(Pdb)"
        self.pdb_tool.register(self.env)
        self.env.entrypoint = "python script.py"

        output = self.pdb_tool.start_pdb(terminal=self.terminal)
        self.assertEqual(output, "(Pdb)")
        self.assertEqual(self.pdb_tool.pdb_obs, "(Pdb)")

    @patch.object(PDBTool, "interact_with_pdb")
    @patch.object(PDBTool, "close_pdb")
    def test_restart_pdb(self, mock_close_pdb, mock_interact_with_pdb):
        mock_interact_with_pdb.return_value = "(Pdb)"
        self.pdb_tool.register(self.env)
        self.env.entrypoint = "python script.py"

        output = self.pdb_tool.restart_pdb()
        self.assertEqual(output, "(Pdb)")
        self.assertEqual(self.pdb_tool.pdb_obs, "(Pdb)")

    @patch.object(PDBTool, "interact_with_pdb")
    def test_use_command(self, mock_interact_with_pdb):
        mock_interact_with_pdb.return_value = "output"
        self.pdb_tool.register(self.env)
        self.env.current_file = "script.py"
        self.env.all_files = ["script.py"]

        output = self.pdb_tool.use("```pdb p x```")
        self.assertIn("output", output)

    @patch.object(PDBTool, "interact_with_pdb")
    def test_breakpoint_add_clear(self, mock_interact_with_pdb):
        mock_interact_with_pdb.return_value = "output"
        self.pdb_tool.register(self.env)
        self.env.current_file = "script.py"
        self.env.all_files = ["script.py"]

        success, output = self.pdb_tool.breakpoint_add_clear("b 42")
        self.assertTrue(success)
        self.assertIn("output", output)

    @patch.object(PDBTool, "interact_with_pdb", return_value="Breakpoint cleared")
    def test_breakpoint_add_clear_clear_specific(self, mock_interact_with_pdb):
        # Test clearing a specific breakpoint
        # mock_interact_with_pdb.return_value = ""
        # self.pdb_tool.register(self.env)
        self.environment.current_file = "file1.py"
        self.environment.all_files = ["file1.py", "file2.py"]

        success, output = self.pdb_tool.breakpoint_add_clear("cl 20")
        expected_state = {
            "file1.py|||10": "b file1.py:10",
            "file1.py|||30": "b file1.py:30",
            "file2.py|||15": "b file2.py:15",
        }
        self.assertTrue(success)
        self.assertEqual(output, "Breakpoint cleared")
        self.assertEqual(self.pdb_tool.breakpoints_state, expected_state)

    @patch.object(PDBTool, "interact_with_pdb")
    def test_breakpoint_modify_remove(self, mock_interact_with_pdb):
        # Test removing breakpoints within the rewritten code
        self.pdb_tool.register(self.env)
        self.pdb_tool.breakpoint_modify("file1.py", 15, 25, 5)
        expected_state = {
            "file1.py|||10": "b file1.py:10",
            "file1.py|||24": "b file1.py:24",
            "file2.py|||15": "b file2.py:15",
        }
        self.assertEqual(self.pdb_tool.breakpoints_state, expected_state)

    @patch.object(PDBTool, "interact_with_pdb")
    def test_breakpoint_modify_move(self, mock_interact_with_pdb):
        # Test moving breakpoints after the rewritten code
        # self.pdb_tool.register(self.env)
        self.pdb_tool.breakpoint_modify("file1.py", 5, 15, 10)
        expected_state = {
            "file2.py|||15": "b file2.py:15",
            "file1.py|||19": "b file1.py:19",
            "file1.py|||29": "b file1.py:29",
        }
        self.assertEqual(self.pdb_tool.breakpoints_state, expected_state)

    @patch.object(PDBTool, "interact_with_pdb")
    def test_breakpoint_modify_remove_all(self, mock_interact_with_pdb):
        # Test removing all breakpoints in the file
        self.pdb_tool.breakpoint_modify("file1.py", None, None, 0)
        expected_state = {"file2.py|||15": "b file2.py:15"}
        self.assertEqual(self.pdb_tool.breakpoints_state, expected_state)

    @patch.object(PDBTool, "interact_with_pdb")
    def test_breakpoint_modify_no_change(self, mock_interact_with_pdb):
        # Test no change for breakpoints before the rewritten code
        self.pdb_tool.breakpoint_modify("file1.py", 25, 35, 5)
        expected_state = {
            "file1.py|||10": "b file1.py:10",
            "file1.py|||20": "b file1.py:20",
            "file2.py|||15": "b file2.py:15",
        }
        self.assertEqual(self.pdb_tool.breakpoints_state, expected_state)

    @patch.object(PDBTool, "interact_with_pdb")
    def test_current_breakpoints_no_breakpoints(self, mock_interact_with_pdb):
        # Set up the environment with no breakpoints
        self.pdb_tool.breakpoints_state = {}

        # Call the method
        result = self.pdb_tool.current_breakpoints()

        # Assert the result
        assert result == "No breakpoints are set."

    @patch.object(PDBTool, "interact_with_pdb")
    def test_current_breakpoints_with_breakpoints(self, mock_interact_with_pdb):
        # Set up the environment with some breakpoints
        self.pdb_tool.breakpoints_state = {
            "file1.py|||10": "b file1.py:10",
            "file2.py|||20": "b file2.py:20",
            "file1.py|||15": "b file1.py:15",
        }

        # Call the method
        result = self.pdb_tool.current_breakpoints()

        # Assert the result
        expected_result = (
            "line 10 in file1.py\nline 15 in file1.py\nline 20 in file2.py"
        )
        assert result == expected_result

    @patch.object(PDBTool, "interact_with_pdb")
    def test_get_current_frame_file(self, mock_interact_with_pdb):
        mock_interact_with_pdb.return_value = (
            "/home/user/repo/script.py(10)<module>()\n-> line of code"
        )
        self.pdb_tool.register(self.env)
        self.env.working_dir = "/home/user/repo"

        self.pdb_tool.get_current_frame_file()
        # self.assertEqual(self.pdb_tool.current_frame_file, "script.py")


if __name__ == "__main__":
    unittest.main()
