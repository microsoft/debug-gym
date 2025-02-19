import os
from os.path import join as pjoin

from froggy.entities import Observation
from froggy.tools.tool import EnvironmentTool
from froggy.tools.toolbox import Toolbox
from froggy.utils import is_subdirectory


@Toolbox.register()
class ViewTool(EnvironmentTool):
    name: str = "view"
    action: str = "```view"
    instructions = {
        "template": "```view <path/to/file.py>```",
        "description": "Specify a file path to set as current working file. The file path should be relative to the root directory of the repository.",
        "examples": [
            "```view main.py``` to navigate to a file called 'main.py' in the root",
            "```view src/util.py``` to navigate to a file called 'util.py' in a subdirectory called 'src'",
        ],
    }

    def is_editable(self, filepath):
        return filepath in self.environment.editable_files

    def use(self, action) -> Observation:
        new_file = action.strip("`").split(" ", 1)[1].strip()
        if new_file.startswith(str(self.environment.working_dir)):
            new_file = new_file[len(str(self.environment.working_dir)) + 1 :]

        if not is_subdirectory(new_file, self.environment.working_dir):
            obs = [
                f"Invalid file path. The file path must be inside the root directory: `{self.environment.working_dir}`.",
                f"Current file: `{self.environment.current_file}`.",
                (
                    "The file is editable."
                    if self.is_editable(self.environment.current_file)
                    else "The file is read-only, it is not editable."
                ),
            ]

        elif new_file == self.environment.current_file:
            obs = [
                f"Already viewing `{new_file}`.",
                (
                    "The file is editable."
                    if self.is_editable(new_file)
                    else "The file is read-only, it is not editable."
                ),
            ]

        elif os.path.isfile(pjoin(self.environment.working_dir, new_file)):
            self.environment.load_current_file(filepath=new_file)
            self.environment.current_file = new_file
            obs = [
                f"Viewing `{new_file}`.",
                (
                    "The file is editable."
                    if self.is_editable(new_file)
                    else "The file is read-only, it is not editable."
                ),
            ]

        else:
            obs = [
                f"File not found. Could not navigate to `{new_file}`.",
                f"Make sure that the file path is given relative to the root: `{self.environment.working_dir}`.",
                f"Current file: `{self.environment.current_file}`.",
                (
                    "The file is editable."
                    if self.is_editable(self.environment.current_file)
                    else "The file is read-only, it is not editable."
                ),
            ]

        obs = " ".join(obs)
        return Observation(self.name, obs)
