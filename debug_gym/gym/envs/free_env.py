from __future__ import annotations

from pathlib import Path
from typing import Any

from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.kubernetes import KubernetesTerminal
from debug_gym.gym.terminals.local import LocalTerminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.logger import DebugGymLogger


class FreeEnv(RepoEnv):
    """Minimal RepoEnv wrapper that exposes a container image without tasks."""

    DEFAULT_TASK_NAME = "free-session"

    def __init__(
        self,
        image: str,
        *,
        terminal: Terminal | str | None = None,
        mount_path: str | Path | None = None,
        setup_commands: list[str] | None = None,
        instructions: str | None = None,
        init_git: bool = True,
        workspace_dir: str | Path = "/workspace",
        logger: DebugGymLogger | None = None,
        terminal_kwargs: dict[str, Any] | None = None,
        **env_kwargs: Any,
    ):
        self.container_image = image
        self._custom_instructions = instructions or ""
        self.init_git = init_git
        self._setup_commands = list(setup_commands or [])
        self.container_workdir = str(workspace_dir)

        shared_logger = logger or DebugGymLogger("debug-gym")
        terminal_obj = self._ensure_terminal(
            terminal=terminal,
            logger=shared_logger,
            terminal_kwargs=terminal_kwargs or {},
        )

        super().__init__(
            path=str(mount_path) if mount_path is not None else None,
            entrypoint="true",
            debug_entrypoint="true",
            max_score=0,
            terminal=terminal_obj,
            logger=shared_logger,
            **env_kwargs,
        )

    def _ensure_terminal(
        self,
        *,
        terminal: Terminal | str | None,
        logger: DebugGymLogger,
        terminal_kwargs: dict[str, Any],
    ) -> Terminal:
        terminal_kwargs = dict(terminal_kwargs)
        terminal_kwargs.setdefault("working_dir", self.container_workdir)

        if terminal is None or isinstance(terminal, str):
            kind = (terminal or "docker").lower()
            if kind in ("kubernetes", "k8s"):
                k8s_terminal = KubernetesTerminal(
                    base_image=self.container_image,
                    setup_commands=list(self._setup_commands),
                    logger=logger,
                    **terminal_kwargs,
                )
                k8s_terminal.task_name = self.DEFAULT_TASK_NAME
                return k8s_terminal
            if kind != "docker":
                raise ValueError(
                    "FreeEnv terminal must be 'docker', 'kubernetes', or a Terminal instance."
                )
            return DockerTerminal(
                base_image=self.container_image,
                setup_commands=list(self._setup_commands),
                logger=logger,
                **terminal_kwargs,
            )

        if not isinstance(terminal, Terminal):
            raise TypeError("terminal must be a Terminal instance, a string, or None")

        self._configure_terminal(terminal=terminal, logger=logger)
        return terminal

    def _configure_terminal(
        self, *, terminal: Terminal, logger: DebugGymLogger
    ) -> None:
        if hasattr(terminal, "base_image"):
            setattr(terminal, "base_image", self.container_image)

        if hasattr(terminal, "working_dir") and not isinstance(terminal, LocalTerminal):
            terminal.working_dir = self.container_workdir

        if self._setup_commands and hasattr(terminal, "setup_commands"):
            existing = list(getattr(terminal, "setup_commands", []))
            for cmd in self._setup_commands:
                if cmd not in existing:
                    existing.append(cmd)
            setattr(terminal, "setup_commands", existing)

        if isinstance(terminal, KubernetesTerminal):
            try:
                terminal.task_name = self.DEFAULT_TASK_NAME
            except ValueError:
                logger.debug(
                    "Kubernetes pod already created; unable to update task name."
                )

        terminal.logger = logger

    def set_entrypoints(self, entrypoint: str, debug_entrypoint: str | None = None):
        self._entrypoint = entrypoint
        self._debug_entrypoint = debug_entrypoint or entrypoint
        self.entrypoint = entrypoint
        self.debug_entrypoint = debug_entrypoint or entrypoint

    def load_dataset(self, problems: str | list[str] | None = None):
        return {self.DEFAULT_TASK_NAME: {"image": self.container_image}}

    def setup_task(self, task_name: str | None, options: dict | None = None) -> None:
        self.task_name = task_name or self.DEFAULT_TASK_NAME
        if isinstance(self.terminal, KubernetesTerminal):
            try:
                self.terminal.task_name = self.task_name
            except ValueError:
                self.logger.debug(
                    "Kubernetes pod already created; keeping existing task name."
                )
        self.base_image = self.container_image

    def setup_workspace(self) -> None:
        self.workspace.reset()
        target_dir = self.workspace.working_dir
        if not isinstance(self.terminal, LocalTerminal):
            self.workspace.working_dir = Path(self.container_workdir)
            self.terminal.working_dir = self.container_workdir
            target_dir = self.workspace.working_dir
            # Ensure the working directory exists inside the container/pod.
            self.terminal.run(f"mkdir -p {target_dir}", raises=True)
        if self.path:
            self.workspace.copy_content(self.path)
        self.workspace.setup_file_filters()

    def setup_terminal(self) -> None:
        if not self.init_git:
            return
        if not self._git_available():
            self.logger.debug(
                "Git is not available in the container; skipping repository setup."
            )
            return
        self.terminal.run("touch .debugignore .debugreadonly")
        super().setup_terminal()

    def _git_available(self) -> bool:
        success, _ = self.terminal.run("command -v git")
        return success

    @property
    def instructions(self) -> str:
        if self._custom_instructions:
            return self._custom_instructions
        return (
            "Interact freely with the environment.\n"
            "There is no predefined task or evaluation.\n"
            f"Container image: {self.container_image}"
        )

    def calculate_max_score(self, eval_output):
        return 0

    def calculate_score(self, eval_output):
        return 0

    def calculate_resolved(self, eval_output) -> bool:
        return False

    def calculate_terminated(self, eval_output) -> bool:
        return False

    def eval(self, **kwargs):
        raise NotImplementedError("FreeEnv does not support evaluation.")
