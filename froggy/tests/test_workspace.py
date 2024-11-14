from unittest.mock import patch
from froggy.envs import RepoEnv
from froggy.utils import load_config
from pathlib import PosixPath

@patch('sys.argv', ['run.py', 'scripts/config.yaml', '--agent', 'cot', '--debug', '-v'])
def test_workspace():
    config, args = load_config()
    config = config[args.agent]

    assert args.config_file == 'scripts/config.yaml'
    assert args.agent == 'cot'
    assert args.debug == True
    assert args.verbose == True
    
    env = RepoEnv(**config["env_kwargs"])

    assert isinstance(env.path, PosixPath)
    assert env.path == PosixPath('data/pytorch')
    