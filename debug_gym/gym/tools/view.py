import os
from os.path import join as pjoin

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.gym.utils import is_subdirectory


@Toolbox.register()
class ViewTool(EnvironmentTool):
    name: str = "view"
    examples = [
        """view(path="main.py") to navigate to a file called 'main.py' in the root.""",
        """view(path="src/util.py") to navigate to a file called 'util.py' in a subdirectory called 'src'.""",
    ]
    description = (
        "Specify a file path to set as current working file. The file path should be relative to the root directory of the repository."
        + "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):\n"
        + "\n".join(examples)
    )
    arguments = {
        "path": {
            "type": ["string"],
            "description": "The path to the file to be viewed. The path should be relative to the root directory of the repository.",
        },
    }

    def use(self, environment, path: str) -> Observation:
        new_file = path.strip()
        if new_file == "":
            obs = "Invalid file path. Please specify a valid file path."
            return Observation(self.name, obs)

        if new_file.startswith(str(environment.working_dir)):
            new_file = new_file[len(str(environment.working_dir)) + 1 :]

        if not is_subdirectory(new_file, environment.working_dir):
            obs = (
                "Invalid file path. The file path must be inside "
                f"the root directory: `{environment.working_dir}`."
            )
        elif os.path.isfile(pjoin(environment.working_dir, new_file)):
            read_only = " (read-only)" if not environment.is_editable(new_file) else ""
            obs = (
                f"Viewing `{new_file}`{read_only}:"
                f"\n\n```\n{environment.read_file(new_file)}\n```\n\n"
            )
        else:
            obs = (
                f"File not found. Could not navigate to `{new_file}`. "
                f"Make sure that the file path is given relative to the root: `{environment.working_dir}`."
            )

        return Observation(self.name, obs)
