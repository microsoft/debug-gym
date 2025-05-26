from debug_gym.gym.envs.aider import AiderBenchmarkEnv
from debug_gym.gym.envs.env import RepoEnv, TooledEnv
from debug_gym.gym.envs.mini_nightmare import MiniNightmareEnv
from debug_gym.gym.envs.swe_bench import SWEBenchEnv
from debug_gym.gym.envs.swe_smith import SWESmithEnv


def select_env(env_type: str = None):
    match env_type:
        case None:
            return RepoEnv
        case "aider":
            return AiderBenchmarkEnv
        case "swebench":
            return SWEBenchEnv
        case "swesmith":
            return SWESmithEnv
        case "mini_nightmare":
            return MiniNightmareEnv
        case _:
            raise ValueError(f"Unknown benchmark {env_type}")
