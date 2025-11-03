import pytest
from filelock import FileLock

from debug_gym.gym.envs import R2EGymEnv, SWEBenchEnv, SWESmithEnv

BUILD_ENV_CONFIGS = {
    "swe_smith": {
        "env_class": SWESmithEnv,
        "problems": ["john-kurkowski__tldextract.3d1bf184.combine_file__1vnuqpt4"],
    },
    "swe_bench": {
        "env_class": SWEBenchEnv,
        "problems": ["astropy__astropy-14096"],
    },
    "r2egym": {
        "env_class": R2EGymEnv,
        "problems": ["aiohttp_final:d7cd0613472fd4d9940e37f1c55921f6a1515324"],
    },
}


def make_env_factory(env_name, worker_id, tmp_path_factory):
    """Build the `env_name`'s docker image only once."""
    # Ref: https://pytest-xdist.readthedocs.io/en/stable/how-to.html#making-session-scoped-fixtures-execute-only-once
    kwargs = dict(BUILD_ENV_CONFIGS[env_name])
    env_class = kwargs.pop("env_class")

    def _make_env():
        return env_class(**kwargs)

    if worker_id == "master":
        # Not running with pytest-xdist or we are in the master process
        _make_env()
    else:
        # When running with pytest-xdist, synchronize between workers using a lock
        root_tmp_dir = tmp_path_factory.getbasetemp().parent
        lock_file = root_tmp_dir / f"{env_class.__name__}_init.lock"
        with FileLock(str(lock_file)):
            # Only the first worker to acquire the lock will initialize the environment
            _make_env()

    return _make_env


@pytest.fixture(scope="session")
def get_swe_smith_env(worker_id, tmp_path_factory):
    return make_env_factory("swe_smith", worker_id, tmp_path_factory)


@pytest.fixture(scope="session")
def get_swe_bench_env(worker_id, tmp_path_factory):
    return make_env_factory("swe_bench", worker_id, tmp_path_factory)


@pytest.fixture(scope="session")
def get_swe_bench_debug_env(worker_id, tmp_path_factory):
    return make_env_factory("swe_bench", worker_id, tmp_path_factory)


@pytest.fixture(scope="session")
def get_r2egym_env(worker_id, tmp_path_factory):
    return make_env_factory("r2egym", worker_id, tmp_path_factory)
