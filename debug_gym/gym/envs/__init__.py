from debug_gym.gym.envs.aider import AiderBenchmarkEnv
from debug_gym.gym.envs.env import RepoEnv, TooledEnv
from debug_gym.gym.envs.free_env import FreeEnv
from debug_gym.gym.envs.local import LocalEnv
from debug_gym.gym.envs.mini_nightmare import MiniNightmareEnv
from debug_gym.gym.envs.r2egym import R2EGymEnv
from debug_gym.gym.envs.swe_bench import SWEBenchEnv
from debug_gym.gym.envs.swe_bench_debug import SWEBenchDebugEnv
from debug_gym.gym.envs.swe_smith import SWESmithEnv
from debug_gym.logger import DebugGymLogger

__all__ = [
    "AiderBenchmarkEnv",
    "RepoEnv",
    "TooledEnv",
    "FreeEnv",
    "LocalEnv",
    "MiniNightmareEnv",
    "R2EGymEnv",
    "SWEBenchEnv",
    "SWEBenchDebugEnv",
    "SWESmithEnv",
    "select_env",
    "load_dataset",
]


def select_env(env_type: str = None) -> type[RepoEnv]:
    match env_type:
        case "local":
            return LocalEnv
        case "aider":
            return AiderBenchmarkEnv
        case "swebench":
            return SWEBenchEnv
        case "swebench-debug":
            return SWEBenchDebugEnv
        case "swesmith":
            return SWESmithEnv
        case "mini_nightmare":
            return MiniNightmareEnv
        case "r2egym":
            return R2EGymEnv
        case "FreeEnv":
            return FreeEnv
        case _:
            raise ValueError(f"Unknown environment {env_type}")


def load_dataset(config: dict, logger: DebugGymLogger | None = None) -> dict:
    """Load dataset based on the given config."""
    if config.get("type") is None:
        raise ValueError("Dataset config must specify 'type' field.")

    try:
        env = select_env(config.get("type"))
    except ValueError as exc:
        raise ValueError(
            f"Unknown environment type '{config.get('type')}' from dataset's config: {config}"
        ) from exc

    dataset = env.load_dataset(logger=logger, **config)
    return dataset
