from __future__ import annotations

from pathlib import Path
from typing import Any

from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.kubernetes import KubernetesTerminal
from debug_gym.gym.terminals.terminal import Terminal


class FreeEnv(RepoEnv):
    """Repo environment that allows an agent to freely explore a codebase."""

    def __init__(
        self,
        task_data: dict | None = None,
        *,
        image: str | None = None,
        terminal: Terminal | None = None,
        local_path: str | Path | None = None,
        workspace_dir: str | Path = "/testbed",
        setup_commands: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        terminal = terminal or DockerTerminal(logger=kwargs.get("logger"))
        if not isinstance(terminal, (DockerTerminal, KubernetesTerminal)):
            raise ValueError(
                f"{self.__class__.__name__} only supports DockerTerminal and KubernetesTerminal."
            )

        task_data = task_data or {
            "image": image,
            "local_path": local_path,
            "workspace_dir": workspace_dir,
            "setup_commands": setup_commands
            or ["apt-get update -y && apt-get install -y git"],
        }
        super().__init__(
            task_data=task_data,
            terminal=terminal,
            **kwargs,
        )

    @property
    def task_name(self):
        return f"FreeEnv({self.task_data['image']})"

    def setup_task(self) -> None:
        self.terminal.task_name = self.task_name
        self.terminal.base_image = self.task_data["image"]

    def setup_workspace(self):
        self.workspace.working_dir = self.task_data.get("workspace_dir", "/testbed")
        self.workspace.reset()

        if self.task_data.get("local_path"):
            self.logger.info(
                f"Copying content from {self.task_data['local_path']} to {self.workspace.working_dir}."
            )
            self.workspace.copy_content(
                src=self.task_data["local_path"], target=self.workspace.working_dir
            )

        self.workspace.setup_file_filters()  # Use codebase's .debugignore and .debugreadonly.

    def setup_terminal(self) -> None:
        """Apply FreeEnv tweaks and reuse RepoEnv git boo? but the agent cantstrapping when enabled."""

        self.logger.info(f"Configuring {self.terminal}...")

        # Ensure core utilities exist before RepoEnv renders directory listings.
        for cmd in self.task_data.get("setup_commands", []):
            self.terminal.run(cmd, raises=True)

        # self.terminal.run(
        #     f"mkdir -p {shlex.quote(self._workspace_dir)}",
        #     raises=True,
        # )

        if self._git_available():
            self.terminal.run("git init")
            self.terminal.run("git config user.name 'debug-gym'")
            self.terminal.run("git config user.email '<>'")

            # self.terminal.run(
            #     "git add *.py *.txt"
            # )  # Aider tasks only have Python and text files.
            # self.terminal.run("git commit -am 'Init'")

            # self.terminal.run(
            #     "git add .debugignore .debugreadonly"
            # )  # Aider tasks come with those.
            # self.terminal.run("git commit -am 'Add debug-gym ignore and read-only files'")

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
            "You are placed in an isolated Linux environment,"
            "use the available tools to interact with the environment effectively."
        )
