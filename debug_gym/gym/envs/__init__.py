import logging

from debug_gym.gym.envs.aider import AiderBenchmarkEnv
from debug_gym.gym.envs.env import RepoEnv, TooledEnv
from debug_gym.gym.envs.mini_nightmare import MiniNightmareEnv


def select_env(env_type: str = None) -> type[RepoEnv]:
    match env_type:
        case None:
            return RepoEnv
        case "aider":
            return AiderBenchmarkEnv
        case "swebench":
            from debug_gym.gym.envs.swe_bench import SWEBenchEnv

            logging.getLogger("httpx").setLevel(logging.WARNING)
            return SWEBenchEnv
        case "swesmith":
            from debug_gym.gym.envs.swe_smith import SWESmithEnv

            logging.getLogger("httpx").setLevel(logging.WARNING)
            return SWESmithEnv
        case "mini_nightmare":
            return MiniNightmareEnv
        case _:
            raise ValueError(f"Unknown benchmark {env_type}")
