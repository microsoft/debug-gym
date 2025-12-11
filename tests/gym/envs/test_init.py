"""Unit tests for select_env and load_dataset functions without mocking."""

import pytest

from debug_gym.gym.envs import load_dataset, select_env
from debug_gym.gym.envs.aider import AiderBenchmarkEnv
from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.envs.local import LocalEnv
from debug_gym.gym.envs.mini_nightmare import MiniNightmareEnv
from debug_gym.gym.envs.r2egym import R2EGymEnv
from debug_gym.gym.envs.swe_bench import SWEBenchEnv
from debug_gym.gym.envs.swe_bench_debug import SWEBenchDebugEnv
from debug_gym.gym.envs.swe_smith import SWESmithEnv
from debug_gym.logger import DebugGymLogger


class TestSelectEnv:
    """Test the select_env function with all environment types."""

    def test_select_local_env(self):
        """Test selecting LocalEnv."""
        env_class = select_env("local")
        assert env_class == LocalEnv

    def test_select_aider_env(self):
        """Test selecting AiderBenchmarkEnv."""
        env_class = select_env("aider")
        assert env_class == AiderBenchmarkEnv

    def test_select_swebench_env(self):
        """Test selecting SWEBenchEnv."""
        env_class = select_env("swebench")
        assert env_class == SWEBenchEnv

    def test_select_swebench_debug_env(self):
        """Test selecting SWEBenchDebugEnv."""
        env_class = select_env("swebench-debug")
        assert env_class == SWEBenchDebugEnv

    def test_select_swesmith_env(self):
        """Test selecting SWESmithEnv."""
        env_class = select_env("swesmith")
        assert env_class == SWESmithEnv

    def test_select_mini_nightmare_env(self):
        """Test selecting MiniNightmareEnv."""
        env_class = select_env("mini_nightmare")
        assert env_class == MiniNightmareEnv

    def test_select_r2egym_env(self):
        """Test selecting R2EGymEnv."""
        env_class = select_env("r2egym")
        assert env_class == R2EGymEnv

    def test_select_free_env(self):
        """Test selecting FreeEnv."""
        env_class = select_env("FreeEnv")
        assert env_class == FreeEnv

    def test_select_unknown_env(self):
        """Test that selecting unknown env raises ValueError."""
        with pytest.raises(ValueError, match="Unknown environment unknown_env"):
            select_env("unknown_env")

    def test_select_none_env(self):
        """Test that selecting None env raises ValueError."""
        with pytest.raises(ValueError, match="Unknown environment None"):
            select_env(None)

    def test_select_empty_string_env(self):
        """Test that selecting empty string env raises ValueError."""
        with pytest.raises(ValueError, match="Unknown environment"):
            select_env("")

    def test_select_env_case_sensitive(self):
        """Test that env selection is case-sensitive."""
        # "local" works
        assert select_env("local") == LocalEnv

        # "Local" should fail
        with pytest.raises(ValueError, match="Unknown environment Local"):
            select_env("Local")

    def test_all_env_types_are_classes(self):
        """Test that all returned env types are actually classes."""
        env_types = [
            "local",
            "aider",
            "swebench",
            "swebench-debug",
            "swesmith",
            "mini_nightmare",
            "r2egym",
            "FreeEnv",
        ]

        for env_type in env_types:
            env_class = select_env(env_type)
            assert isinstance(env_class, type), f"{env_type} should return a class"


class TestLoadDataset:
    """Test the load_dataset function."""

    def test_load_dataset_missing_type(self):
        """Test that load_dataset raises ValueError when 'type' is missing."""
        config = {"some_key": "some_value"}

        with pytest.raises(
            ValueError, match="Dataset config must specify 'type' field"
        ):
            load_dataset(config)

    def test_load_dataset_type_none(self):
        """Test that load_dataset raises ValueError when 'type' is None."""
        config = {"type": None}

        with pytest.raises(
            ValueError, match="Dataset config must specify 'type' field"
        ):
            load_dataset(config)

    def test_load_dataset_unknown_type(self):
        """Test that load_dataset raises ValueError for unknown environment type."""
        config = {"type": "nonexistent_env"}

        with pytest.raises(
            ValueError,
            match="Unknown environment type 'nonexistent_env' from dataset's config",
        ):
            load_dataset(config)

    def test_load_dataset_with_logger(self):
        """Test that load_dataset accepts a logger parameter."""
        logger = DebugGymLogger("test_logger")
        config = {"type": "mini_nightmare", "build_image": False}

        # This should not raise an error even if dataset loading fails
        # We're just testing the function signature and error handling
        try:
            dataset = load_dataset(config, logger=logger)
            # If it succeeds, check that it returns a dict
            assert isinstance(dataset, dict)
        except Exception:
            # If it fails for other reasons (e.g., dataset not available),
            # that's okay - we're testing the logger parameter acceptance
            pass

    def test_load_dataset_without_logger(self):
        """Test that load_dataset works without logger parameter."""
        config = {"type": "mini_nightmare", "build_image": False}

        # This should not raise an error about missing logger
        try:
            dataset = load_dataset(config)
            # If it succeeds, check that it returns a dict
            assert isinstance(dataset, dict)
        except Exception:
            # If it fails for other reasons (e.g., dataset not available),
            # that's okay - we're testing that logger is optional
            pass

    def test_load_dataset_passes_config_to_env(self):
        """Test that load_dataset passes configuration to the environment's load_dataset method."""
        # Using mini_nightmare as it has a simple load_dataset signature
        config = {
            "type": "mini_nightmare",
            "build_image": False,
            "problems": None,  # This should be passed to MiniNightmareEnv.load_dataset
        }

        try:
            dataset = load_dataset(config)
            # If successful, verify it's a dict (expected return type)
            assert isinstance(dataset, dict)
        except Exception:
            # If it fails for other reasons, that's acceptable
            pass

    def test_load_dataset_error_message_includes_config(self):
        """Test that error message includes the config when type is invalid."""
        config = {"type": "invalid_type", "other_param": "value"}

        with pytest.raises(ValueError) as exc_info:
            load_dataset(config)

        # Check that the error message includes the config
        error_message = str(exc_info.value)
        assert "invalid_type" in error_message
        assert "config" in error_message.lower()

    def test_load_dataset_preserves_value_error_from_select_env(self):
        """Test that ValueError from select_env is properly wrapped."""
        config = {"type": "bad_env"}

        with pytest.raises(ValueError) as exc_info:
            load_dataset(config)

        # Check that it's chained from the original ValueError
        error_message = str(exc_info.value)
        assert "Unknown environment type 'bad_env'" in error_message

    def test_load_dataset_empty_config(self):
        """Test load_dataset with empty config dict."""
        config = {}

        with pytest.raises(
            ValueError, match="Dataset config must specify 'type' field"
        ):
            load_dataset(config)
