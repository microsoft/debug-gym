from debug_gym.gym.entities import EvalOutput
from debug_gym.gym.envs.swe_bench import SWEBenchEnv


class SWEBenchDebugEnv(SWEBenchEnv):

    def setup_terminal(self):
        super().setup_terminal()

        # Apply official test patch since this is a debugging task.
        self.terminal.run(f"git apply - <<'EOF'\n{self.test_patch}\nEOF")

    def eval(self, **kwargs) -> EvalOutput:
        success, output = self.terminal.run(self.entrypoint, timeout=self.run_timeout)
        self.last_eval = EvalOutput(success, output)
        return self.last_eval
