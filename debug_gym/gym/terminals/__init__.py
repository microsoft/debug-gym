from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.kubernetes import KubernetesTerminal
from debug_gym.gym.terminals.local import LocalTerminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.logger import DebugGymLogger


def select_terminal(
    terminal_config: dict | None = None,
    logger: DebugGymLogger | None = None,
    uuid: str | None = None,
) -> Terminal | None:
    if terminal_config is None:
        return None

    logger = logger or DebugGymLogger("debug-gym")
    terminal_type = terminal_config["type"]
    docker_only = ["base_image", "setup_commands"]
    match terminal_type:
        case "docker":
            terminal_class = DockerTerminal
        case "kubernetes":
            terminal_class = KubernetesTerminal
        case "local":
            terminal_class = LocalTerminal
            if any(cfg in terminal_config for cfg in docker_only):
                logger.warning("Ignoring Docker-only parameters for local terminal.")
        case _:
            raise ValueError(f"Unknown terminal {terminal_type}")

    return terminal_class(
        **terminal_config,
        logger=logger,
        extra_labels={"uuid": uuid},
    )
