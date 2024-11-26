import os
import numpy as np
import subprocess
import unittest

from os.path import join as pjoin
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path, PosixPath

import pytest
from froggy.envs import RepoEnv, TooledEnv

class TestTooledEnv(unittest.TestCase):
    def setUp(self):
        self.env = TooledEnv()

    def test_seed(self):
        seed_value = 42
        self.env.seed(seed_value)

        # Check if the rng attribute is set to a numpy random state
        self.assertIsInstance(self.env.rng, np.random.RandomState)

        # Check if the random state is initialized with the correct seed
        expected_rng = np.random.RandomState(seed_value)

        state1 = self.env.rng.get_state()
        state2 = expected_rng.get_state()
        self.assertEqual(state1[0], state2[0])  # Check the algorithm
        np.testing.assert_array_equal(state1[1], state2[1])  # Check the state
        self.assertEqual(state1[2:], state2[2:])  # Check the remaining elements

    def test_add_tool(self):
        tool = MagicMock()
        tool.name = "tool1"
        self.env.add_tool(tool)
        self.assertIn("tool1", self.env.tools)
        self.assertEqual(self.env.tools["tool1"], tool)

    def test_add_tool_existing(self):
        tool = MagicMock()
        tool.name = "tool1"
        self.env.add_tool(tool)
        with self.assertRaises(ValueError):
            self.env.add_tool(tool)

    def test_has_tool(self):
        tool = MagicMock()
        tool.name = "tool1"
        self.env.add_tool(tool)
        self.assertTrue(self.env.has_tool("tool1"))
        self.assertFalse(self.env.has_tool("tool2"))

    def test_get_tool(self):
        tool = MagicMock()
        tool.name = "tool1"
        self.env.add_tool(tool)
        self.assertEqual(self.env.get_tool("tool1"), tool)

    def test_get_triggered_tools(self):
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.is_triggered.return_value = True
        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.is_triggered.return_value = False
        self.env.add_tool(tool1)
        self.env.add_tool(tool2)
        triggered_tools = self.env.get_triggered_tools("action")
        self.assertIn(tool1, triggered_tools)
        self.assertNotIn(tool2, triggered_tools)

    def test_actions(self):
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.action = "action1"
        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.action = "action2"
        self.env.add_tool(tool1)
        self.env.add_tool(tool2)
        self.assertEqual(self.env.actions, ["action1", "action2"])

    def test_actions_str(self):
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.action = "action1"
        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.action = "action2"
        self.env.add_tool(tool1)
        self.env.add_tool(tool2)
        self.assertEqual(self.env.actions_str, "action1, action2")

    def test_tool_instructions(self):
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.instructions = "instructions1"
        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.instructions = "instructions2"
        self.env.add_tool(tool1)
        self.env.add_tool(tool2)
        self.assertEqual(self.env.tool_instructions, {"tool1": "instructions1", "tool2": "instructions2"})


@pytest.mark.usefixtures("tmp_path")
class TestRepoEnv(unittest.TestCase):
    @patch('tempfile.TemporaryDirectory')
    @patch('atexit.register')
    @patch('shutil.copytree')
    def test_setup_workspace(self, mock_copytree, mock_atexit_register, mock_tempdir):
        # Mock the temporary directory
        mock_tempdir.return_value.name = '/mock/tempdir'

        # Create an instance of RepoEnv
        repo_env = RepoEnv(run_timeout=10, dir_tree_depth=2, auto_view_change=True)

        # Call setup_workspace
        repo_env.setup_workspace('/mock/path', 'python', ['readonly_pattern'])

        # Assertions
        self.assertEqual(repo_env.path, Path('/mock/path'))
        self.assertEqual(repo_env.working_dir, Path('/mock/tempdir'))

        # Check if the temporary directory was created
        mock_tempdir.assert_called_once_with(prefix='RepoEnv-')

        # Check if atexit.register was called to cleanup the temporary directory
        mock_atexit_register.assert_called_once_with(repo_env.tempdir.cleanup)

        # Check if shutil.copytree was called to copy the directory
        mock_copytree.assert_called_once_with(Path('/mock/path'), Path('/mock/tempdir'), dirs_exist_ok=True)

    @patch('tempfile.TemporaryDirectory')
    @patch('atexit.register')
    @patch('shutil.copytree')
    def test_setup_workspace_with_none_path(self, mock_copytree, mock_atexit_register, mock_tempdir):
        # Create an instance of RepoEnv
        repo_env = RepoEnv(run_timeout=10, dir_tree_depth=2, auto_view_change=True)

        # Call setup_workspace with None path
        repo_env.setup_workspace(None, "/bin/bash")

        # Assertions
        self.assertIsNone(repo_env.path)

        # Check that copytree and tempdir were not called
        mock_tempdir.assert_not_called()
        mock_copytree.assert_not_called()
        mock_atexit_register.assert_not_called()

    @patch('tempfile.TemporaryDirectory')
    def test_cleanup_workspace(self, mock_tempdir):
        mock_tempdir_instance = MagicMock()
        mock_tempdir.return_value = mock_tempdir_instance

        env = RepoEnv()
        env.tempdir = mock_tempdir_instance

        env.cleanup_workspace()

        mock_tempdir_instance.cleanup.assert_called_once()

    def test_instructions(self):
        # Create mock tools
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.instructions = "instructions1"
        tool1.action = "action1"

        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.instructions = "instructions2"
        tool2.action = "action2"

        env = RepoEnv()
        # Add tools to the environment
        env.add_tool(tool1)
        env.add_tool(tool2)

        # Define the expected instructions
        expected_instructions = {
            "Available tools to solve the problem": {
                "tool1": "instructions1",
                "tool2": "instructions2"
            },
            "Available commands": "action1, action2"
        }

        # Get the instructions from the environment
        instructions = env.instructions

        # Assertions
        self.assertEqual(instructions, expected_instructions)

    @patch('shutil.copy2')
    @patch('os.path.isdir', return_value=False)
    @patch('glob.glob', return_value=['/path/to/repo/file1.txt', '/path/to/repo/file2.txt'])
    @patch('os.scandir')
    @patch('os.walk')
    @patch('shutil.copytree')
    def test_restore(self, mock_copytree, mock_os_walk, mock_scandir, mock_glob, mock_isdir, mock_copy2):
        # Mock the return value of os.scandir
        mock_scandir.return_value.__enter__.return_value = [
            MagicMock(is_dir=lambda: False, path='/path/to/repo/file1.txt'),
            MagicMock(is_dir=lambda: False, path='/path/to/repo/file2.txt')
        ]

        # Mock the return value of os.walk
        mock_os_walk.return_value = [
            ('/path/to/repo', ('subdir',), ('file1.txt', 'file2.txt')),
            ('/path/to/repo/subdir', (), ('subfile1.txt',)),
        ]
        # Create an instance of RepoEnv
        env = RepoEnv(path='/path/to/repo')

        # Call the restore method
        env.restore('/path/to/repo/file1.txt', '/path/to/repo/file2.txt')

        # Assertions
        mock_glob.assert_not_called()  # Ensure glob is not called since filepaths are provided
        mock_isdir.assert_any_call(Path('/path/to/repo/file1.txt'))
        mock_isdir.assert_any_call(Path('/path/to/repo/file2.txt'))
        mock_copy2.assert_any_call(Path('/path/to/repo/file1.txt'), Path(env.working_dir) / 'file1.txt')
        mock_copy2.assert_any_call(Path('/path/to/repo/file2.txt'), Path(env.working_dir) / 'file2.txt')

    @patch.object(RepoEnv, 'directory_tree')
    def test_display_files(self, mock_directory_tree):
        # Mock the return value of directory_tree
        mock_directory_tree.return_value = "\n|-- file1.py\n|-- file2.py\n"

        env = RepoEnv()
        # Call the display_files method with editable_only=False
        result = env.display_files(editable_only=False)

        # Define the expected result
        expected_result = "\nAll files:\n|-- file1.py\n|-- file2.py\n"

        # Assertions
        self.assertEqual(result, expected_result)
        mock_directory_tree.assert_called_once_with(editable_only=False)

    @patch('froggy.utils.show_line_number')
    def test_current_code_with_line_number(self, mock_show_line_number):
        # Mock the return value of show_line_number
        mock_show_line_number.return_value = "1    def foo():\n2        return 42"

        # Create an instance of RepoEnv
        env = RepoEnv(path='.')

        # Set the current file and its content
        env.current_file = 'file.py'
        env.current_file_content = 'def foo():\n    return 42'

        # Call the current_code_with_line_number method
        result = env.current_code_with_line_number()

        # Define the expected result
        expected_result = {
            "File name": 'file.py',
            "Content": "\n     1 def foo():\n     2     return 42\n",
        }

        # Assertions
        self.assertEqual(result, expected_result)

    @patch.object(RepoEnv, 'get_triggered_tools')
    @patch.object(RepoEnv, 'get_tool')
    @patch.object(RepoEnv, 'has_tool', return_value=False)
    @patch.object(RepoEnv, 'run')
    @patch.object(RepoEnv, 'display_files')
    @patch.object(RepoEnv, 'current_code_with_line_number')
    def test_step(self, mock_current_code_with_line_number, mock_display_files, mock_run, mock_has_tool, mock_get_tool, mock_get_triggered_tools):
        # Mock the PDBTool
        mock_pdb_tool = MagicMock()
        mock_pdb_tool.use.return_value = "PDB tool used"
        mock_pdb_tool.rewrite_success = True
        mock_pdb_tool.current_frame_file = "file.py"
        mock_pdb_tool.pdb_obs = "PDB started"
        mock_get_tool.return_value = None #mock_pdb_tool

        # Mock the return values of display_files and current_code_with_line_number
        mock_display_files.return_value = "file list"
        mock_current_code_with_line_number.return_value = "code with line numbers"

        # Create an instance of RepoEnv
        env = RepoEnv(path='.')

        # Mock the get_triggered_tools method to return the PDBTool
        mock_get_triggered_tools.return_value = [mock_pdb_tool]

        # Call the step method with an action
        obs, score, done, infos = env.step("some action")

        # Assertions
        mock_get_triggered_tools.assert_called_once_with("some action")
        mock_pdb_tool.use.assert_called_once_with("some action")

        self.assertEqual(obs, "PDB tool used")
        self.assertEqual(score, 0)
        self.assertFalse(done)
        self.assertIn("obs", infos)
        self.assertIn("last_run_obs", infos)
        self.assertIn("dbg_obs", infos)
        self.assertIn("dir_tree", infos)
        self.assertIn("editable_files", infos)
        self.assertIn("current_breakpoints", infos)
        self.assertIn("current_code_with_line_number", infos)
        self.assertIn("action", infos)
        self.assertIn("done", infos)
        self.assertIn("score", infos)
        self.assertIn("max_score", infos)
        self.assertIn("instructions", infos)
        self.assertIn("rewrite_counter", infos)

    @patch('froggy.utils._walk')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_file', return_value=False)
    @patch('os.scandir')
    @patch('os.walk')
    @patch('shutil.copytree')
    @patch('tempfile.TemporaryDirectory')
    def test_directory_tree(self, mock_tempdir, mock_copytree, mock_os_walk, mock_scandir, mock_is_file, mock_exists, mock_walk):
        mock_tempdir.return_value.name = '/mock/tempdir'

        mock_scandir.return_value.__enter__.return_value = [
            MagicMock(is_dir=lambda: False, path='/path/to/repo/file1.txt'),
            MagicMock(is_dir=lambda: False, path='/path/to/repo/file2.txt')
        ]

        # Mock the return value of os.walk
        mock_os_walk.return_value = [
            ('/path/to/repo', ('subdir',), ('file1.py', 'file2.py')),
            ('/path/to/repo/subdir', (), ('subfile1.txt',)),
        ]

        # Create an instance of RepoEnv
        env = RepoEnv(path='/path/to/repo')

        # Call the directory_tree method
        result = env.directory_tree()

        # Define the expected result
        expected_result = (
           "\n\n"
           "/mock/tempdir/\n  "
           "|-- file1.txt\n  "
           "|-- file2.txt\n\n"
        )

        # Assertions
        self.assertEqual(result, expected_result)

    @patch.object(RepoEnv, 'restore')
    @patch.object(RepoEnv, 'run')
    @patch.object(RepoEnv, 'has_tool', return_value=False)
    @patch.object(RepoEnv, 'get_tool')
    @patch('os.scandir')
    @patch('os.walk')
    @patch('shutil.copytree')
    @patch('tempfile.TemporaryDirectory')
    def test_reset(self, mock_tempdir, mock_copytree, mock_os_walk, mock_scandir, mock_get_tool, mock_has_tool, mock_run, mock_restore):
        # Mock the PDBTool
        mock_pdb_tool = MagicMock()
        mock_pdb_tool.start_pseudo_terminal.return_value = None
        mock_pdb_tool.pdb_obs = "PDB started"
        mock_get_tool.return_value = mock_pdb_tool

        mock_tempdir.return_value.name = '/mock/tempdir'

        mock_scandir.return_value.__enter__.return_value = [
            MagicMock(is_dir=lambda: False, path='/path/to/repo/file1.txt'),
            MagicMock(is_dir=lambda: False, path='/path/to/repo/file2.txt')
        ]

        # Mock the return value of os.walk
        mock_os_walk.return_value = [
            ('/path/to/repo', ('subdir',), ('file1.py', 'file2.py')),
            ('/path/to/repo/subdir', (), ('subfile1.txt',)),
        ]

        # Create an instance of RepoEnv
        env = RepoEnv(path='/path/to/repo')

        # Call the reset method
        obs, infos = env.reset(seed=42)

        # Assertions
        mock_restore.assert_called_once()
        mock_run.assert_called_once()
        self.assertEqual(env.current_file, None)
        self.assertEqual(env.current_file_content, None)
        self.assertEqual(env.current_breakpoints_state, {})
        self.assertEqual(env.rewrite_counter, 0)
        self.assertIn("obs", infos)
        self.assertIn("dbg_obs", infos)
        self.assertIn("last_run_obs", infos)
        self.assertIn("dir_tree", infos)
        self.assertIn("editable_files", infos)
        self.assertIn("current_breakpoints", infos)
        self.assertIn("current_code_with_line_number", infos)
        self.assertIn("action", infos)
        self.assertIn("done", infos)
        self.assertIn("score", infos)
        self.assertIn("max_score", infos)
        self.assertIn("instructions", infos)
        self.assertIn("rewrite_counter", infos)

    @patch('os.scandir')
    @patch('os.walk')
    @patch('shutil.copytree')
    @patch('builtins.open', new_callable=mock_open)
    def test_overwrite_file(self, mock_open, mock_copytree, mock_os_walk, mock_scandir):
        mock_scandir.return_value.__enter__.return_value = [
            MagicMock(is_dir=lambda: False, path='/path/to/repo/file1.txt'),
            MagicMock(is_dir=lambda: False, path='/path/to/repo/file2.txt')
        ]

        # Mock the return value of os.walk
        mock_os_walk.return_value = [
            ('/path/to/repo', ('subdir',), ('file1.py', 'file2.py')),
            ('/path/to/repo/subdir', (), ('subfile1.txt',)),
        ]
        # Create an instance of RepoEnv
        env = RepoEnv(path='/path/to/repo')

        # Define the file path and content to be written
        filepath = 'file.py'
        content = 'print("Hello, World!")'

        # Call the overwrite_file method
        env.overwrite_file(filepath, content)

        # Assertions
        mock_open.assert_called_once_with(pjoin(env.working_dir, filepath), 'w')
        mock_open().write.assert_called_once_with(content)


    @patch('os.scandir')
    @patch('os.walk')
    @patch('shutil.copytree')
    @patch('subprocess.run')
    def test_patch(self, mock_subprocess_run, mock_copytree, mock_os_walk, mock_scandir):
        # Mock the return value of subprocess.run
        mock_result = MagicMock()
        mock_result.stdout = "diff --git a/path/to/repo/file1.py b/path/to/repo/file1.py\n"
        mock_subprocess_run.return_value = mock_result

        mock_scandir.return_value.__enter__.return_value = [
            MagicMock(is_dir=lambda: False, path='/path/to/repo/file1.txt'),
            MagicMock(is_dir=lambda: False, path='/path/to/repo/file2.txt')
        ]

        # Mock the return value of os.walk
        mock_os_walk.return_value = [
            ('/path/to/repo', ('subdir',), ('file1.py', 'file2.py')),
            ('/path/to/repo/subdir', (), ('subfile1.txt',)),
        ]
        # Create an instance of RepoEnv
        env = RepoEnv(path='/path/to/repo')

        # Call the patch property
        result = env.patch

        # Define the expected result
        expected_result = "diff --git a/path/to/repo/file1.py b/path/to/repo/file1.py\n"

        # Assertions
        mock_subprocess_run.assert_called_once_with(
            ["git", "diff", "--no-index", env.path, env.working_dir],
            text=True,
            capture_output=True
        )
        self.assertEqual(result, expected_result)


def test_run_success(tmp_path):
    working_dir = str(tmp_path)
    # create a dummy file
    with open(tmp_path / "file.py", "w") as f:
        f.write("print('Hello, World!')")
    env = RepoEnv(path=working_dir, entrypoint="python file.py")
    output, done = env.run()

    assert output == "Hello, World!"
    assert done
    assert env.score == 1


def test_run_timeout(tmp_path):
    working_dir = str(tmp_path)
    # runs for longer than the timeout
    with open(tmp_path / "file.py", "w") as f:
        f.write("import time; time.sleep(5)")
    env = RepoEnv(path=working_dir, entrypoint="python file.py", run_timeout=1)
    output, done = env.run()

    assert output == "Timeout expired."
    assert not done
    assert env.score == 0


if __name__ == '__main__':
    unittest.main()
