import json

import datasets
import docker
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS, TestStatus
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.test_spec.test_spec import make_test_spec

from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.kubernetes import KubernetesTerminal
from debug_gym.gym.terminals.terminal import DebugGymLogger, Terminal
from debug_gym.gym.utils import filter_problems


class SWEBenchEnv(RepoEnv):
    CACHE = DEBUG_GYM_CACHE_DIR / "swe-bench"

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

        self.test_directives = []
        super().__init__(task_data=task_data, terminal=terminal, **kwargs)

    @property
    def instructions(self) -> str:
        return self.task_data["problem_statement"]

    @property
    def task_name(self) -> str:
        return self.task_data["instance_id"]

    def setup_task(self):
        self.repo = self.task_data["repo"]
        self.package_name = self.repo.split("/")[1]
        self.version = self.task_data["version"]
        self.install_configs = MAP_REPO_VERSION_TO_SPECS[self.repo][self.version]
        self.gold_patch = self.task_data["patch"]
        self.test_spec = make_test_spec(self.task_data)
        self.base_image = f"swebench/{self.test_spec.instance_image_key}".replace(
            "__", "_1776_"
        )
        self.base_commit = self.task_data["base_commit"]
        self.test_patch = self.task_data["test_patch"]
        self.fail_to_pass = json.loads(self.task_data["FAIL_TO_PASS"])
        self.pass_to_pass = json.loads(self.task_data["PASS_TO_PASS"])
        self.test_cmd = self.install_configs["test_cmd"]
        self.test_directives = get_test_directives(self.task_data)

        self.entrypoint = " ".join([self.test_cmd, *self.test_directives])

        if self.package_name == "sphinx" or self.package_name == "sympy":
            if self.entrypoint.startswith("PYTHONWARNINGS"):
                # Move PYTHONWARNINGS from the entrypoint to the session commands
                export, remaining = self.entrypoint.split(" ", 1)
                self.terminal.session_commands.append(f"export {export}")
                self.entrypoint = remaining

        if self.package_name == "django":
            self.terminal.env_vars["LANG"] = "en_US.UTF-8"
            self.terminal.env_vars["LANGUAGE"] = "en_US:en"
            self.terminal.env_vars["LC_ALL"] = "en_US.UTF-8"
            self.terminal.setup_commands += self.install_configs.get(
                "eval_commands", []
            )
        elif self.package_name == "requests":
            self.terminal.env_vars["CURL_CA_BUNDLE"] = ""

        # -s (capture=no) with pytest allows for debugging with pdb
        # -q (quiet) with pytest avoids long pytest output
        self.debug_entrypoint = self.entrypoint.replace("pytest", "pytest -sq")

        if self.package_name == "sphinx" or self.package_name == "sympy":
            # use pytest instead of `sympy bin/test` and `sphinx tox` so pdb breakpoints work
            expression = " ".join(self.test_directives)
            self.debug_entrypoint = f"python -m pytest {expression}"

        # --tb=short with pytest keeps the output concise
        self.entrypoint = self.entrypoint.replace("--tb=no", "--tb=short")

        self.git_apply_cmd = f"git apply -"

    def setup_workspace(self):
        self.terminal.task_name = self.task_name
        self.terminal.base_image = self.base_image
        # Ignore hidden files (dotfiles) and any contents under hidden directories
        self.workspace.reset(
            ignore_patterns=["**/.*"], readonly_patterns=self.test_directives
        )
        self.set_entrypoints(self.entrypoint, self.debug_entrypoint)

    def setup_terminal(self):
        self.logger.debug(f"Configuring {self.terminal}...")

        # Install tree for listdir.
        self.terminal.run("apt update && apt install -y tree")

        self.terminal.session_commands.append("source /opt/miniconda3/bin/activate")
        self.terminal.session_commands.append("conda activate testbed")

        if self.package_name == "astropy":
            self.terminal.run("sed -i '/^addopts = -p no:warnings/s/^/# /' setup.cfg")
        elif self.package_name == "requests":
            # To avoid using httpbin.org which is unresponsive at time.
            self.terminal.run(
                "pip install httpbin[mainapp]==0.10.2 pytest-httpbin==2.1.0"
            )
            self.terminal.run("nohup gunicorn -b 127.0.0.1:80 -k gevent httpbin:app &")
            self.terminal.run(
                "nohup gunicorn -b 127.0.0.1:443 --certfile=/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/pytest_httpbin/certs/server.pem --keyfile=/opt/miniconda3/envs/testbed/lib/python3.9/site-packages/pytest_httpbin/certs/server.key -k gevent httpbin:app &"
            )
            self.terminal.run('echo "127.0.0.1    httpbin.org" >> /etc/hosts')
        elif self.task_name == "pylint-dev__pylint-4661":
            self.terminal.run("pip install appdirs==1.4.4")
        elif self.package_name == "sphinx" or self.package_name == "sympy":
            self.terminal.run("pip install pytest")

        # Apply any changes needed to the install commands.
        self.terminal.run("git config user.name 'debug-gym'")
        self.terminal.run("git config user.email '<>'")
        self.terminal.run(f"git commit -am 'Setting up {self.task_name}'")

        # Remove the remote so the agent won't see newer commits.
        self.terminal.run("git remote remove origin")

    def apply_gold_patch(self):
        self.logger.debug(f"Applying gold patch to {self.working_dir}.")
        command = self.git_apply_cmd + f" <<'EOF'\n{self.gold_patch}\nEOF"
        self.terminal.run(command, raises=True)
        self.logger.debug("Patch applied successfully.")

    def eval(self, **kwargs) -> EvalOutput:
        # We need to apply the test patch before running any evaluation.
        # Reset any changes made to test_directives files.
        self.terminal.run(f"git checkout -- {' '.join(self.test_directives)}")

        # Apply official test patch (hidden until now)
        self.terminal.run(f"git apply - <<'EOF'\n{self.test_patch}\nEOF")

        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)
        self.last_eval = EvalOutput(success, output)

        # Reset any changes made to test_directives files.
        self.terminal.run(f"git checkout -- {' '.join(self.test_directives)}")

        return self.last_eval

    def calculate_max_score(self, eval_output: EvalOutput) -> int:
        return len(self.fail_to_pass)

    def calculate_score(self, eval_output: EvalOutput) -> int:
        test_status_map = MAP_REPO_TO_PARSER[self.repo](
            eval_output.output, self.test_spec
        )
        self.logger.debug(f"fail_to_pass: {self.fail_to_pass}")
        self.logger.debug(f"Test status map: {test_status_map}")
        f2p_score = sum(
            1
            for test in self.fail_to_pass
            # *Do not* assume silent success for now as done in SWE-Bench grading.py
            if test_status_map.get(test, TestStatus.ERROR.value)
            in (TestStatus.PASSED.value, TestStatus.XFAIL.value)
        )
        p2p_score = sum(
            1
            for test in self.pass_to_pass
            if test_status_map.get(test, TestStatus.PASSED.value)
            in (TestStatus.PASSED.value, TestStatus.XFAIL.value)
        )

        # The final score is f2p_score only if all pass_to_pass tests passed.
        score = f2p_score if p2p_score == len(self.pass_to_pass) else 0

        assert score <= self.max_score
        return score

    @classmethod
    def load_dataset(
        cls,
        dataset_id: str = "SWE-bench/SWE-bench_Verified",
        dataset_revision: str = "99450355ca8c611021187a57ffac304b66666738",
        split: str = "test",
        problems: list | None = None,
        prepull_images: bool = False,
        logger: DebugGymLogger | None = None,
        **kwargs,
    ) -> dict:
        ds = datasets.load_dataset(dataset_id, revision=dataset_revision)[split]

        # Memory efficient filtering of problems.
        id2idx = {id: i for i, id in enumerate(ds["instance_id"])}
        problems = filter_problems(id2idx, problems)
        dataset = {problem: ds[id2idx[problem]] for problem in problems}

        # Add env_type to each task_data.
        for task_data in dataset.values():
            task_data["env_type"] = "swebench"

        image_names = set(
            f"sweb.eval.x86_64.{id.replace('__', '_1776_')}" for id in dataset
        )

        if prepull_images:
            # Download all images needed for SWE-Bench.
            client = docker.from_env()
            tagged_image_names = set(f"swebench/{name}:latest" for name in image_names)

            existing_images = set(
                tag for image in client.images.list() for tag in image.tags
            )
            missing_images = tagged_image_names - existing_images
            if missing_images:
                if logger:
                    logger.info(f"Found {len(missing_images)} missing Docker images.")
                for i, image_name in enumerate(missing_images):
                    if logger:
                        logger.info(
                            f"Pulling Docker images {i + 1}/{len(missing_images)}: `{image_name}`."
                        )
                    client.images.pull(image_name)
        return dataset
