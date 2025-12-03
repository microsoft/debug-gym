from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.local import LocalTerminal
from debug_gym.gym.terminals.terminal import Terminal


class LocalEnv(RepoEnv):

    def __init__(
        self,
        path: str,
        terminal: Terminal | None = None,
        entrypoint: str = "python -m pytest -sq .",
        debug_entrypoint: str | None = None,
        **kwargs,
    ):
        task_data = {"path": path}
        terminal = terminal or LocalTerminal()
        super().__init__(
            task_data=task_data,
            terminal=terminal,
            entrypoint=entrypoint,
            debug_entrypoint=debug_entrypoint,
            **kwargs,
        )

    @property
    def instructions(self) -> str:
        return f"Investigate the current repository, run the tests to figure out any issues, then rewrite the code to fix them."

    @property
    def task(self) -> str:
        return self.task_data["path"].split("/")[-1]

    def setup_task(self) -> None:
        """Setup the task information. Called once at reset."""
        self.path = self.task_data["path"]

    def setup_workspace(self) -> None:
        """Setup the workspace. Called once at reset."""
        self.workspace.reset()
        self.workspace.copy_content(self.path)
        self.workspace.setup_file_filters()

    def setup_terminal(self) -> None:
        """Setup the terminal. Called once at reset."""

        self.logger.debug(f"Configuring {self.terminal}...")

        self.terminal.run("git init -b main")
        self.terminal.run("git config user.name 'debug-gym'")
        self.terminal.run("git config user.email '<>'")

        self.terminal.run("git add *")
        self.terminal.run("git commit -am 'Init'")

        self.terminal.run("git add .debugignore .debugreadonly")
        self.terminal.run("git commit -am 'Add debug-gym ignore and read-only files'")
