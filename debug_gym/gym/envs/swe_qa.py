import subprocess

import datasets

from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.kubernetes import KubernetesTerminal
from debug_gym.gym.terminals.terminal import DebugGymLogger, Terminal
from debug_gym.gym.utils import filter_problems

# From https://github.com/peng-weihan/SWE-QA-Bench/blob/e90e5d32c9a7318cafa50c284c43da01768752fe/repos.txt
SWEQA_REPOS = [
    {"url": "https://github.com/astropy/astropy", "commit": "0a041d3"},
    {"url": "https://github.com/django/django", "commit": "14fc2e9"},
    {"url": "https://github.com/pallets/flask", "commit": "85c5d93"},
    {"url": "https://github.com/matplotlib/matplotlib", "commit": "a5e1f60"},
    {"url": "https://github.com/pylint-dev/pylint", "commit": "44740e5"},
    {"url": "https://github.com/pytest-dev/pytest", "commit": "5989efe"},
    {"url": "https://github.com/psf/requests", "commit": "46e939b"},
    {"url": "https://github.com/scikit-learn/scikit-learn", "commit": "adb1ae7"},
    {"url": "https://github.com/sphinx-doc/sphinx", "commit": "6c9e320"},
    {"url": "https://github.com/sqlfluff/sqlfluff", "commit": "db9801b"},
    {"url": "https://github.com/sympy/sympy", "commit": "3c817ed"},
    {"url": "https://github.com/pydata/xarray", "commit": "40119bf"},
    {"url": "https://github.com/conan-io/conan", "commit": "52f43d9"},
    {"url": "https://github.com/reflex-dev/reflex", "commit": "fe0f946"},
    {"url": "https://github.com/streamlink/streamlink", "commit": "ab1f365"},
]


class SWEQAEnv(RepoEnv):
    """Software Engineering Question Answering Environment

    Reference:
    https://github.com/peng-weihan/SWE-QA-Bench
    """

    CACHE = DEBUG_GYM_CACHE_DIR / "sweqa"

    def __init__(
        self,
        task_data: dict,
        terminal: Terminal | None = None,
        **kwargs,
    ):
        terminal = terminal or DockerTerminal(logger=kwargs.get("logger"))
        if not isinstance(terminal, (DockerTerminal, KubernetesTerminal)):
            raise ValueError(
                f"{self.__class__.__name__} only supports DockerTerminal and KubernetesTerminal."
            )

        super().__init__(task_data=task_data, terminal=terminal, **kwargs)

    @property
    def instructions(self) -> str:
        return self.task_data["question"]

    @property
    def task_name(self) -> str:
        return self.task_data["instance_id"]

    def setup_task(self):
        self.repo_name = self.task_data["instance_id"].split("-")[0]
        self.problem_idx = int(self.task_data["instance_id"].split("-")[1])
        self.base_image = "python:3.12"
        self.answer = self.task_data["answer"]

    def setup_workspace(self):
        self.terminal.task_name = self.task_name
        self.terminal.base_image = self.base_image
        self.workspace.reset()

        self.logger.debug("Copying files..")
        self.workspace.copy_content(
            src=self.CACHE / self.repo_name, target=self.workspace.working_dir
        )

    def setup_terminal(self):
        self.logger.debug(f"Configuring {self.terminal}...")

        self.terminal.env_vars["PATH"] = "/root/.local/bin:/bin"

        # Change ownership to root user.
        self.terminal.run(f"chown -R root:root {self.workspace.working_dir}")

        # Install git, uv.
        # self.terminal.run("apt update && apt install -y git curl")
        self.terminal.run("curl -LsSf https://astral.sh/uv/install.sh | sh")

        # Create venv
        self.terminal.run("uv venv && source .venv/bin/activate")
        self.terminal.run("uv pip install pip")

        # Setup session commands.
        self.terminal.session_commands.append(
            f"source {self.workspace.working_dir}/.venv/bin/activate"
        )

    @classmethod
    def load_dataset(
        cls,
        dataset_id: str = "Raymone023/SWE-QA-Benchmark",
        dataset_revision: str = "7f8d77650ec5939a31b15b3256152f0275e7d6e3",
        problems: list | None = None,
        logger: DebugGymLogger | None = None,
        **kwargs,
    ) -> dict:
        ds = datasets.load_dataset(dataset_id, revision=dataset_revision)
        splits = list(ds.keys())  # Keep ordering.

        # Join all splits into one and use split name + idx as instance_id.
        all_data = []
        for split in splits:
            for i, item in enumerate(ds[split]):
                item["instance_id"] = f"{split}-{i}"
                all_data.append(item)
        ds = datasets.Dataset.from_list(all_data)

        # Memory efficient filtering of problems.
        id2idx = {id: i for i, id in enumerate(ds["instance_id"])}
        problems = filter_problems(id2idx, problems)
        dataset = {problem: ds[id2idx[problem]] for problem in problems}

        # Add env_type to each task_data.
        for task_data in dataset.values():
            task_data["env_type"] = "sweqa"

        cls.CACHE.mkdir(parents=True, exist_ok=True)
        for repo in SWEQA_REPOS:
            repo_name = repo["url"].split("/")[-1]
            repo_path = cls.CACHE / repo_name
            if not repo_path.exists():
                logger.debug(f"Cloning {repo['url']} at {repo['commit']}...")
                subprocess.run(
                    ["git", "clone", repo["url"], str(repo_path)], check=True
                )
                subprocess.run(
                    ["git", "-C", str(repo_path), "checkout", repo["commit"]],
                    check=True,
                )

        return dataset
