import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path, PosixPath
from froggy.envs import RepoEnv
from froggy.utils import load_config

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

if __name__ == '__main__':
    unittest.main()
    