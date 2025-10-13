from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class SubmitTool(EnvironmentTool):
    name = "submit"
    description = (
        "Finalize the task: applies the hidden official benchmark test patch, switches to the "
        "official test entrypoint, runs the benchmark evaluation, and returns the final results. "
        "After this call the environment is marked as submitted and further development should stop."
    )
    arguments = {}

    def use(self, environment, **kwargs) -> Observation:
        if not hasattr(environment, "final_submit"):
            return Observation(
                self.name, "Environment does not support final submission."
            )
        eval_output = environment.final_submit()
        summary = [
            "=== FINAL SUBMISSION RESULTS ===",
            f"Success: {eval_output.success}",
            "--- Raw Output (truncated to 2000 chars) ---",
            eval_output.output,
        ]
        return Observation(self.name, "\n".join(summary))
