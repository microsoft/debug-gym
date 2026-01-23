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

    if isinstance(terminal_config, Terminal):
        return terminal_config

    if not isinstance(terminal_config, dict):
        raise TypeError(
            "terminal configuration must be a dict, Terminal instance, or None",
        )

    config = dict(terminal_config)
    terminal_type = str(config.pop("type", "")).lower()
    if not terminal_type:
        raise ValueError("Terminal configuration must include a 'type' key")

    logger = logger or DebugGymLogger("debug-gym")
    match terminal_type:
        case "docker":
            terminal_class = DockerTerminal
        case "kubernetes":
            terminal_class = KubernetesTerminal
        case "local":
            terminal_class = LocalTerminal
        case _:
            raise ValueError(f"Unknown terminal {terminal_type}")

    extra_labels = config.pop("extra_labels", {}) or {}
    if uuid is not None:
        extra_labels = {**extra_labels, "uuid": uuid}

    # Pass extra_labels to terminals that support it (Kubernetes and Docker)
    if terminal_class in (KubernetesTerminal, DockerTerminal) and extra_labels:
        config["extra_labels"] = extra_labels

    if terminal_class not in (KubernetesTerminal, DockerTerminal):
        config.pop("extra_labels", None)

    return terminal_class(
        logger=logger,
        **config,
    )
