import platform
import subprocess

import pytest


def is_docker_running():
    try:
        subprocess.check_output(["docker", "ps"])
        return True
    except Exception:
        return False


if_docker_running = pytest.mark.skipif(
    not is_docker_running(),
    reason="Docker not running",
)


if_is_linux = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="Interactive ShellSession (pty) requires Linux.",
)


def pytest_configure(config):
    # Make the marker globally available
    pytest.if_docker_running = if_docker_running
    pytest.if_is_linux = if_is_linux
