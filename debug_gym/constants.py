from pathlib import Path

DEBUG_GYM_CONFIG_DIR = Path.joinpath(Path.home(), ".config", "debug_gym")
DEBUG_GYM_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_GYM_CACHE_DIR = Path.joinpath(Path.home(), ".cache", "debug_gym")
DEBUG_GYM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
