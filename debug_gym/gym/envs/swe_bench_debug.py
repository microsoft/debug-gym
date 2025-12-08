from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.swe_bench import SWEBenchEnv


class SWEBenchDebugEnv(SWEBenchEnv):

    def setup_terminal(self):
        super().setup_terminal()

        # Apply official test patch since this is a debugging task.
        self.terminal.run(f"git apply - <<'EOF'\n{self.test_patch}\nEOF")
        self.terminal.run(f"git commit -am 'Applying test patch for {self.task_name}'")

    def eval(self, **kwargs) -> EvalOutput:
        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)
        self.last_eval = EvalOutput(success, output)
        return self.last_eval

    @classmethod
    def load_dataset(cls, *args, **kwargs) -> dict:
        dataset = SWEBenchEnv.load_dataset(*args, **kwargs)

        # Add env_type to each task_data.
        for task_data in dataset.values():
            task_data["env_type"] = "swebench-debug"

        return dataset
