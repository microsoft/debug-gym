from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.local import LocalTerminal
from debug_gym.gym.terminals.terminal import Terminal
from debug_gym.logger import DebugGymLogger


class FreeEnv(RepoEnv):
    """Lightweight RepoEnv wrapper for running arbitrary container images."""

    DEFAULT_TASK_NAME = "free-session"

    def __init__(
        self,
        image: str,
        *,
        terminal: Terminal | None = None,
        mount_path: str | Path | None = None,
        setup_commands: list[str] | None = None,
        instructions: str | None = None,
        init_git: bool = True,
        workspace_dir: str | Path = "/testbed",
        logger: DebugGymLogger | None = None,
        **env_kwargs: Any,
    ) -> None:
        """Create a free-form environment backed by an existing repository terminal."""
        self.container_image = image
        self._custom_instructions = (instructions or "").strip()
        self.init_git = init_git
        self._setup_commands = list(setup_commands or [])
        self._workspace_dir = str(workspace_dir)

        shared_logger = logger or DebugGymLogger("debug-gym")

        super().__init__(
            path=str(mount_path) if mount_path is not None else None,
            entrypoint="true",
            debug_entrypoint="true",
            max_score=0,
            terminal=terminal,
            logger=shared_logger,
            **env_kwargs,
        )

        if self.terminal is not None:
            self._apply_terminal_settings()

    def _apply_terminal_settings(self) -> None:
        """Keep terminal metadata (image/setup commands) in sync with env state."""
        terminal = self.terminal
        if terminal is None:
            return
        if hasattr(terminal, "base_image"):
            setattr(terminal, "base_image", self.container_image)

        if hasattr(terminal, "setup_commands"):
            terminal.setup_commands = list(self._setup_commands)

        if hasattr(terminal, "working_dir") and not isinstance(terminal, LocalTerminal):
            try:
                terminal.working_dir = self._workspace_dir
            except ValueError:
                self.logger.debug(
                    "Terminal already active; keeping working_dir=%s",
                    getattr(terminal, "working_dir", self._workspace_dir),
                )

        if hasattr(terminal, "task_name"):
            try:
                terminal.task_name = self.DEFAULT_TASK_NAME
            except ValueError:
                self.logger.debug(
                    "Terminal already active; keeping existing task name."
                )

        terminal.logger = self.logger

    def load_dataset(self, problems: str | list[str] | None = None):
        """Expose a single synthetic task keyed by DEFAULT_TASK_NAME."""
        return {self.DEFAULT_TASK_NAME: {"image": self.container_image}}

    def setup_task(self, task_name: str | None, options: dict | None = None) -> None:
        """Record base image metadata for consistency with RepoEnv expectations."""
        self.task_name = task_name or self.DEFAULT_TASK_NAME
        self.base_image = self.container_image
        if hasattr(self.terminal, "base_image"):
            setattr(self.terminal, "base_image", self.base_image)

    def setup_workspace(self) -> None:
        """Ensure the remote workspace matches the configured working directory."""
        if isinstance(self.terminal, LocalTerminal):
            super().setup_workspace()
            return

        self.workspace.reset()
        self.workspace.working_dir = Path(self._workspace_dir)
        if self.terminal is not None:
            current_dir = getattr(self.terminal, "working_dir", None)
            if current_dir != self._workspace_dir:
                try:
                    self.terminal.working_dir = self._workspace_dir
                except ValueError:
                    self.logger.debug(
                        "Terminal already active; keeping working_dir=%s", current_dir
                    )
            # Ensure core utilities exist before RepoEnv renders directory listings.
            self.terminal.run(
                "apt-get update -y && apt-get install -y tree", raises=True
            )
            self.terminal.run(
                f"mkdir -p {shlex.quote(self._workspace_dir)}",
                raises=True,
            )

        if self.path:
            self.workspace.copy_content(self.path)

        self.workspace.setup_file_filters()

    def setup_terminal(self) -> None:
        """Apply FreeEnv tweaks and reuse RepoEnv git bootstrapping when enabled."""
        self._apply_terminal_settings()

        if self.terminal is not None:
            self.terminal.run("touch .debugignore .debugreadonly")

        if not self.init_git:
            return
        if not self._git_available():
            self.logger.debug(
                "Git is not available in the container; skipping repository setup.",
            )
            return
        super().setup_terminal()

    def _git_available(self) -> bool:
        """Check for git presence before attempting repository initialization."""
        if self.terminal is None:
            return False
        success, _ = self.terminal.run("command -v git")
        return success

    @property
    def instructions(self) -> str:
        """Provide user-facing guidance, falling back to a generic sandbox blurb."""
        return (
            self._custom_instructions
            or "You are placed in an isolated Linux environment, use the available tools to interact with the environment effectively."
        )

    def reset(self, *, options: dict | None = None):
        """Allow callers to mutate container settings before delegating to RepoEnv."""
        options = options or {}

        image = options.get("image")
        workspace_dir = options.get("workspace_dir")
        setup_commands = options.get("setup_commands")
        instructions = options.get("instructions")
        init_git = options.get("init_git")

        restart_terminal = False

        if image and image != self.container_image:
            self.container_image = image
            restart_terminal = True

        if workspace_dir and str(workspace_dir) != self._workspace_dir:
            self._workspace_dir = str(workspace_dir)
            restart_terminal = True

        if setup_commands is not None:
            new_commands = list(setup_commands)
            if new_commands != self._setup_commands:
                self._setup_commands = new_commands
                restart_terminal = True

        if instructions is not None:
            self._custom_instructions = instructions

        if init_git is not None:
            self.init_git = bool(init_git)

        if restart_terminal and self.terminal is not None:
            try:
                self.terminal.close()
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                self.logger.debug("Failed to close terminal cleanly: %s", exc)

        self._apply_terminal_settings()

        return super().reset(options=options)
