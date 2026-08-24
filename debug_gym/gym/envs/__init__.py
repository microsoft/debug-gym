from importlib import import_module

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
    "SWEQAEnv",
    "select_env",
    "load_dataset",
]

_ENV_CLASSES = {
    "AiderBenchmarkEnv": "debug_gym.gym.envs.aider",
    "RepoEnv": "debug_gym.gym.envs.env",
    "TooledEnv": "debug_gym.gym.envs.env",
    "FreeEnv": "debug_gym.gym.envs.free_env",
    "LocalEnv": "debug_gym.gym.envs.local",
    "MiniNightmareEnv": "debug_gym.gym.envs.mini_nightmare",
    "R2EGymEnv": "debug_gym.gym.envs.r2egym",
    "SWEBenchEnv": "debug_gym.gym.envs.swe_bench",
    "SWEBenchDebugEnv": "debug_gym.gym.envs.swe_bench_debug",
    "SWESmithEnv": "debug_gym.gym.envs.swe_smith",
    "SWEQAEnv": "debug_gym.gym.envs.swe_qa",
}

_ENV_TYPES = {
    "local": "LocalEnv",
    "aider": "AiderBenchmarkEnv",
    "swebench": "SWEBenchEnv",
    "swebench-debug": "SWEBenchDebugEnv",
    "swesmith": "SWESmithEnv",
    "mini_nightmare": "MiniNightmareEnv",
    "r2egym": "R2EGymEnv",
    "r2e": "R2EGymEnv",
    "FreeEnv": "FreeEnv",
    "sweqa": "SWEQAEnv",
}


def __getattr__(name: str):
    if module_name := _ENV_CLASSES.get(name):
        return getattr(import_module(module_name), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def select_env(env_type: str = None) -> type:
    class_name = _ENV_TYPES.get(env_type)
    if class_name is None:
        raise ValueError(f"Unknown environment {env_type}")
    return __getattr__(class_name)


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
