import os
import numpy as np
import subprocess
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path, PosixPath
from froggy.envs import RepoEnv, TooledEnv
from froggy.utils import load_config

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

class TestRepoEnv(unittest.TestCase):
    @patch('sys.argv', ['run.py', 'scripts/config.yaml', '--agent', 'cot', '--debug', '-v'])
    def test_workspace(self):
        config, args = load_config()
        config = config[args.agent]

        assert args.config_file == 'scripts/config.yaml'
        assert args.agent == 'cot'
        assert args.debug == True
        assert args.verbose == True
        
        env = RepoEnv(**config["env_kwargs"])

        assert isinstance(env.path, PosixPath)
        assert env.path == PosixPath('data/pytorch')
        
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
        repo_env.setup_workspace(None)
        
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

    @patch('subprocess.Popen')
    def test_run_success(self, mock_popen):
        # Mock the Popen instance and its methods
        mock_process = MagicMock()
        mock_process.communicate.return_value = ("output", "error")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # Create an instance of RepoEnv
        env = RepoEnv(path='.')

        # Call the run method
        output, done = env.run()

        # Assertions
        mock_popen.assert_called_once_with(
            env.entrypoint,
            env=dict(os.environ, NO_COLOR="1"),
            cwd=env.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        mock_process.communicate.assert_called_once_with(timeout=env.run_timeout)
        self.assertEqual(output, "outputerror")
        self.assertTrue(done)
        self.assertEqual(env.score, 1)
  
    @patch('subprocess.Popen')
    def test_run_timeout(self, mock_popen):
        # Mock the Popen instance and its methods
        mock_process = MagicMock()
        mock_process.communicate.side_effect = subprocess.TimeoutExpired(cmd="cmd", timeout=10)
        mock_popen.return_value = mock_process

        # Create an instance of RepoEnv
        env = RepoEnv(path='.')

        # Call the run method
        output, done = env.run()

        # Assertions
        mock_popen.assert_called_once_with(
            env.entrypoint,
            env=dict(os.environ, NO_COLOR="1"),
            cwd=env.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        mock_process.communicate.assert_called_once_with(timeout=env.run_timeout)
        mock_process.kill.assert_called_once()
        self.assertEqual(output, "Timeout expired.")
        self.assertFalse(done)
        self.assertEqual(env.score, 0)

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
            # "Note": "B indicates breakpoint before a certain line of code, this can be changed using pdb commands such as b, cl, etc."
        }

        # Assertions
        self.assertEqual(result, expected_result)
        # mock_show_line_number.assert_called_once_with(
        #     env.current_file_content,
        #     env.current_file,
        #     env.current_breakpoints_state
        # )

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
        # mock_run.assert_called_once()
        #mock_pdb_tool.start_pseudo_terminal.assert_called_once()
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
        self.assertIn("is_rewrite", infos)
        self.assertIn("max_score", infos)
        self.assertIn("instructions", infos)
        self.assertIn("rewrite_counter", infos)

if __name__ == '__main__':
    unittest.main()
    