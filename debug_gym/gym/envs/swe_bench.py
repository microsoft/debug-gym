import base64
import json
import shlex

import datasets
import docker
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_ONLY_REPOS,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.python import get_test_directives
from swebench.harness.test_spec.test_spec import make_test_spec

from debug_gym.constants import DEBUG_GYM_CACHE_DIR
from debug_gym.gym.entities import EvalDetails, EvalOutput
from debug_gym.gym.envs.env import RepoEnv
from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.terminals.kubernetes import KubernetesTerminal
from debug_gym.gym.terminals.terminal import DebugGymLogger, Terminal
from debug_gym.gym.utils import filter_problems

PASSING_STATUSES = (TestStatus.PASSED.value, TestStatus.XFAIL.value)


def _parse_test_list(value: str | list[str]) -> list[str]:
    return json.loads(value) if isinstance(value, str) else list(value)


class SWEBenchEnv(RepoEnv):
    CACHE = DEBUG_GYM_CACHE_DIR / "swe-bench"
    EVAL_SCRIPT_PATH = "/eval.sh"

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
        self.base_image = self.task_data.get("docker_image") or (
            f"swebench/{self.test_spec.instance_image_key}".replace("__", "_1776_")
        )
        self.base_commit = self.task_data["base_commit"]
        self.test_patch = self.task_data["test_patch"]
        self.fail_to_pass = _parse_test_list(self.task_data["FAIL_TO_PASS"])
        self.pass_to_pass = _parse_test_list(self.task_data["PASS_TO_PASS"])
        self.test_cmd = self.install_configs["test_cmd"]
        self.test_directives = get_test_directives(self.task_data)
        self.eval_script = self._make_eval_script()

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
        self.workspace.reset()
        self.set_entrypoints(self.entrypoint, self.debug_entrypoint)

    def _strip_future_git_history(self) -> None:
        target_commit = shlex.quote(self.base_commit or "HEAD")
        commands = [
            "git remote remove origin 2>/dev/null || true",
            f"TARGET_COMMIT={target_commit}",
            'TARGET_TIMESTAMP=$(git show -s --format=%ct "$TARGET_COMMIT")',
            (
                'git checkout --detach "$TARGET_COMMIT" 2>/dev/null '
                "|| git checkout --detach 2>/dev/null || true"
            ),
            (
                "git for-each-ref --format='delete %(refname)' "
                "refs/heads refs/remotes "
                "| git update-ref --stdin 2>/dev/null || true"
            ),
            (
                "git tag -l | while read -r tag; do "
                'TAG_COMMIT=$(git rev-list -n 1 "$tag") || continue; '
                'TAG_TIME=$(git show -s --format=%ct "$TAG_COMMIT") || continue; '
                'if [ "$TAG_TIME" -gt "$TARGET_TIMESTAMP" ]; then '
                'git tag -d "$tag"; fi; done'
            ),
            "rm -f .git/FETCH_HEAD .git/ORIG_HEAD",
            "git reflog expire --expire=now --all 2>/dev/null || true",
            "git gc --prune=now 2>/dev/null || true",
            "AFTER_TIMESTAMP=$((TARGET_TIMESTAMP + 1))",
            (
                "COMMIT_COUNT=$(git log --oneline --all "
                '--after="@$AFTER_TIMESTAMP" | wc -l)'
            ),
            '[ "$COMMIT_COUNT" -eq 0 ] || exit 1',
        ]
        self.terminal.run(commands, timeout=300, raises=True)

    def _setup_local_httpbin(self) -> None:
        python = "/opt/miniconda3/envs/testbed/bin/python"
        commands = [
            (
                f"{python} -m pip install "
                "'httpbin[mainapp]==0.10.2' 'pytest-httpbin==2.1.0'"
            ),
            (
                "CERT_DIR=$("
                f'{python} -c "import pytest_httpbin.certs, os; '
                'print(os.path.dirname(pytest_httpbin.certs.__file__))"'
                ")"
            ),
            (
                "(nohup gunicorn -b 127.0.0.1:80 -k gevent "
                "httpbin:app > /dev/null 2>&1 &)"
            ),
            (
                "(nohup gunicorn -b 127.0.0.1:443 "
                "--certfile=$CERT_DIR/server.pem "
                "--keyfile=$CERT_DIR/server.key "
                "-k gevent httpbin:app > /dev/null 2>&1 &)"
            ),
            'echo "127.0.0.1    httpbin.org" >> /etc/hosts',
            'echo "export CURL_CA_BUNDLE=" >> ~/.bashrc',
        ]
        self.terminal.run(commands, timeout=300, raises=True)

    def setup_terminal(self):
        self.logger.debug(f"Configuring {self.terminal}...")

        self.terminal.session_commands.append("source /opt/miniconda3/bin/activate")
        self.terminal.session_commands.append("conda activate testbed")

        setup_commands = [
            "ln -s /opt/miniconda3/envs/testbed /root/.venv",
            "python -m pip install chardet",
        ]
        self.terminal.run(setup_commands, timeout=300, raises=True)
        self._strip_future_git_history()

        fixup_commands = []
        if self.package_name == "astropy":
            fixup_commands.append(
                "sed -i '/^addopts = -p no:warnings/s/^/# /' setup.cfg"
            )
        elif self.package_name == "requests":
            self._setup_local_httpbin()
        elif self.task_name == "pylint-dev__pylint-4661":
            fixup_commands.append("pip install appdirs==1.4.4")
        elif self.package_name == "sphinx" or self.package_name == "sympy":
            fixup_commands.append("pip install pytest")

        if fixup_commands:
            self.terminal.run(fixup_commands, timeout=300, raises=True)

        setup_commit = shlex.quote(f"Setting up {self.task_name}")
        self.terminal.run(
            [
                "git config user.name 'debug-gym'",
                "git config user.email '<>'",
                f"git diff --quiet || git commit -am {setup_commit}",
            ],
            timeout=300,
            raises=True,
        )

    def apply_gold_patch(self):
        self.logger.debug(f"Applying gold patch to {self.working_dir}.")
        command = self.git_apply_cmd + f" <<'EOF'\n{self.gold_patch}\nEOF"
        self.terminal.run(command, raises=True)
        self.logger.debug("Patch applied successfully.")

    @staticmethod
    def _get_test_files_from_patch(patch: str) -> tuple[list[str], list[str]]:
        """Separate modified test files from files added by the test patch."""
        modified_files = []
        new_files = []
        source_file = None
        for line in patch.splitlines():
            if line.startswith("--- "):
                source_file = line[4:].split("\t", 1)[0]
            elif line.startswith("+++ ") and source_file is not None:
                target_file = line[4:].split("\t", 1)[0]
                if source_file == "/dev/null" and target_file.startswith("b/"):
                    new_files.append(target_file[2:])
                elif source_file.startswith("a/"):
                    modified_files.append(source_file[2:])
                source_file = None
        return modified_files, new_files

    def _test_reset_commands(self) -> list[str]:
        modified_files, new_files = self._get_test_files_from_patch(self.test_patch)
        commands = []
        if modified_files:
            commands.append(
                f"git checkout {self.base_commit} {shlex.join(modified_files)}"
            )
        if new_files:
            commands.append(f"rm -f {shlex.join(new_files)}")
        return commands

    def _make_eval_script(self) -> str:
        """Backport the upstream new-test-file reset fix missing from 4.1.0."""
        commands = list(self.test_spec.eval_script_list)
        reset_commands = self._test_reset_commands()
        _, new_files = self._get_test_files_from_patch(self.test_patch)

        def is_test_reset(command: str) -> bool:
            if command.startswith(f"git checkout {self.base_commit}"):
                return True
            if not command.startswith("rm -f "):
                return False
            try:
                return set(shlex.split(command)[2:]) == set(new_files)
            except ValueError:
                return False

        apply_index = next(
            index
            for index, command in enumerate(commands)
            if command.startswith("git apply -v - <<")
        )
        reset_start = apply_index
        while reset_start > 0 and is_test_reset(commands[reset_start - 1]):
            reset_start -= 1
        commands[reset_start:apply_index] = reset_commands

        end_index = commands.index(f": '{END_TEST_OUTPUT}'")
        reset_end = end_index + 1
        while reset_end < len(commands) and is_test_reset(commands[reset_end]):
            del commands[reset_end]
        commands[reset_end:reset_end] = reset_commands

        return "\n".join(["#!/bin/bash", "set -uxo pipefail", *commands]) + "\n"

    def _restore_test_files(self) -> None:
        reset_commands = self._test_reset_commands()
        if reset_commands:
            self.terminal.run(reset_commands)

    def eval(self, **kwargs) -> EvalOutput:
        self._restore_test_files()
        encoded_script = base64.b64encode(self.eval_script.encode()).decode()
        command = (
            f"printf %s {shlex.quote(encoded_script)} | base64 -d "
            f"> {self.EVAL_SCRIPT_PATH} && "
            f"/bin/bash {self.EVAL_SCRIPT_PATH}; "
            f"status=$?; rm -f {self.EVAL_SCRIPT_PATH}; exit $status"
        )
        success, output = self.terminal.run(command, timeout=self.run_timeout)
        details = self._calculate_eval_details(output)
        self.last_eval = EvalOutput(success, output, details=details)
        self._restore_test_files()

        return self.last_eval

    def _get_logs_eval(
        self,
        content: str,
        *,
        require_markers: bool = True,
    ) -> tuple[dict[str, str], bool]:
        log_parser = MAP_REPO_TO_PARSER[self.repo]

        bad_codes = [
            code
            for code in (
                APPLY_PATCH_FAIL,
                RESET_FAILED,
                TESTS_ERROR,
                TESTS_TIMEOUT,
            )
            if code in content
        ]
        if bad_codes:
            self.logger.error(f"Bad code found in log: {bad_codes}")
            return {}, False

        has_markers = START_TEST_OUTPUT in content and END_TEST_OUTPUT in content
        if require_markers and not has_markers:
            self.logger.error("Official SWE-bench test output markers not found.")
            return {}, False

        test_content = content
        if has_markers:
            test_content = content.split(START_TEST_OUTPUT, 1)[1].split(
                END_TEST_OUTPUT, 1
            )[0]
        self.logger.info(f"using swebench log_parser for repo: {self.repo}")
        status_map = log_parser(test_content, self.test_spec)
        if has_markers and not status_map:
            status_map = log_parser(content, self.test_spec)
        return status_map, True

    def _calculate_eval_details(
        self,
        output: str,
        *,
        assume_missing_p2p_passed: bool = False,
        require_markers: bool = True,
    ) -> EvalDetails:
        test_status_map, found = self._get_logs_eval(
            output,
            require_markers=require_markers,
        )
        grading_status_map = dict(test_status_map)
        if assume_missing_p2p_passed:
            for test_name in self.pass_to_pass:
                grading_status_map.setdefault(test_name, TestStatus.PASSED.value)
        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.fail_to_pass,
            PASS_TO_PASS: self.pass_to_pass,
        }
        eval_type = (
            EvalType.FAIL_ONLY
            if self.test_spec.repo in FAIL_ONLY_REPOS
            else EvalType.PASS_AND_FAIL
        )
        report = get_eval_tests_report(
            grading_status_map,
            eval_ref,
            eval_type=eval_type,
        )
        reward = int(
            found and get_resolution_status(report) == ResolvedStatus.FULL.value
        )
        n_passed = sum(
            status in PASSING_STATUSES for status in test_status_map.values()
        )
        return EvalDetails(
            parsed_tests=test_status_map,
            n_parsed=len(test_status_map),
            n_passed=n_passed,
            n_failed=len(test_status_map) - n_passed,
            reward=reward,
            n_fail_to_pass=len(self.fail_to_pass),
            n_pass_to_pass=len(self.pass_to_pass),
            grading_report=report,
        )

    def calculate_max_score(self, eval_output: EvalOutput) -> int:
        return 1

    def calculate_score(self, eval_output: EvalOutput) -> int:
        if eval_output.details is not None:
            return int(eval_output.details.reward)
        return int(self._calculate_eval_details(eval_output.output).reward)

    @classmethod
    def load_dataset(
        cls,
        dataset_id: str = "SWE-bench/SWE-bench_Verified",
        dataset_revision: str = "91aa3ed51b709be6457e12d00300a6a596d4c6a3",
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

        image_names = {
            task_data.get("docker_image")
            or (
                f"swebench/sweb.eval.x86_64."
                f"{instance_id.replace('__', '_1776_')}:latest"
            )
            for instance_id, task_data in dataset.items()
        }

        if prepull_images:
            # Download all images needed for SWE-Bench.
            client = docker.from_env()

            existing_images = set(
                tag for image in client.images.list() for tag in image.tags
            )
            missing_images = image_names - existing_images
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
