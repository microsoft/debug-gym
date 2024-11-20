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
    
    @patch('tempfile.TemporaryDirectory')
    def test_cleanup_workspace(self, mock_tempdir):
        mock_tempdir_instance = MagicMock()
        mock_tempdir.return_value = mock_tempdir_instance

        env = RepoEnv()
        env.tempdir = mock_tempdir_instance

        env.cleanup_workspace()

        mock_tempdir_instance.cleanup.assert_called_once()

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

if __name__ == '__main__':
    unittest.main()
    