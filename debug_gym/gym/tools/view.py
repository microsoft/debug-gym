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

    def is_editable(self, environment, filepath):
        return filepath in environment.editable_files

    def use(self, environment, path: str) -> Observation:
        new_file = path.strip()
        if new_file == "":
            obs = [
                "Invalid file path. Please specify a file path.",
                f"Current file: `{environment.current_file}`.",
            ]
            # if current file is None, then no need to check if it is editable
            if environment.current_file is not None:
                obs.append(
                    "The file is editable."
                    if self.is_editable(environment, environment.current_file)
                    else "The file is read-only, it is not editable."
                )

            return Observation(self.name, " ".join(obs))

        if new_file.startswith(str(environment.working_dir)):
            new_file = new_file[len(str(environment.working_dir)) + 1 :]

        if not is_subdirectory(new_file, environment.working_dir):
            obs = [
                f"Invalid file path. The file path must be inside the root directory: `{environment.working_dir}`.",
                f"Current file: `{environment.current_file}`.",
            ]
            # if current file is None, then no need to check if it is editable
            if environment.current_file is not None:
                obs.append(
                    "The file is editable."
                    if self.is_editable(environment, environment.current_file)
                    else "The file is read-only, it is not editable."
                )

        elif new_file == environment.current_file:
            obs = [
                f"Already viewing `{new_file}`.",
                (
                    "The file is editable."
                    if self.is_editable(environment, new_file)
                    else "The file is read-only, it is not editable."
                ),
            ]

        elif os.path.isfile(pjoin(environment.working_dir, new_file)):
            environment.load_current_file(filepath=new_file)
            environment.current_file = new_file
            obs = [
                f"Viewing `{new_file}`.",
                (
                    "The file is editable."
                    if self.is_editable(environment, new_file)
                    else "The file is read-only, it is not editable."
                ),
            ]

        else:
            obs = [
                f"File not found. Could not navigate to `{new_file}`.",
                f"Make sure that the file path is given relative to the root: `{environment.working_dir}`.",
                f"Current file: `{environment.current_file}`.",
            ]
            # if current file is None, then no need to check if it is editable
            if environment.current_file is not None:
                obs.append(
                    "The file is editable."
                    if self.is_editable(environment, environment.current_file)
                    else "The file is read-only, it is not editable."
                )

        return Observation(self.name, " ".join(obs))
