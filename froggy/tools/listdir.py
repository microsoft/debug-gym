from os.path import join as pjoin
from pathlib import Path

from froggy.entities import Observation
from froggy.tools.tool import EnvironmentTool
from froggy.tools.toolbox import Toolbox


@Toolbox.register()
class ListdirTool(EnvironmentTool):
    name: str = "listdir"
    action: str = "```listdir"

    @property
    def instructions(self):
        assert hasattr(self, "environment")
        instruction = {
            "template": "```listdir <path/to/subdirectory> <depth>```",
            "description": f"List the file and folder contents of a subdirectory within the working directory, up to a specified 'depth' (default depth is {self.environment.dir_tree_depth}).",
            "examples": [
                f"```listdir``` to list the contents of the working directory.",
                f"```listdir src/util``` to list the contents of the 'util' subdirectory within the 'src' subdirectory.",
                f"```listdir src 2``` to list the contents of the 'src' subdirectory up to a depth of 2.",
            ],
        }
        return instruction

    def use(self, action) -> Observation:
        try:
            listdir_path, depth = self.clean_action(action)
            startpath = pjoin(self.environment.working_dir, listdir_path)
            obs = self.environment.directory_tree(root=startpath, max_depth=depth)
        except ValueError as e:
            # Raise instead of returning it as an observation?
            obs = str(e)
        return Observation(self.name, obs)

    def clean_action(self, action):
        listdir_path = action.strip("`").strip()
        tmp = listdir_path.split(" ")
        depth = None
        if len(tmp) == 1:
            # e.g., ```listdir```
            listdir_path = "."
        elif len(tmp) == 2:
            # e.g., ```listdir src```
            listdir_path = tmp[1].strip()
        elif len(tmp) == 3:
            # e.g., ```listdir src depth```
            listdir_path = tmp[1].strip()
            depth = int(tmp[2].strip())
            if depth <= 0:
                raise ValueError(f"Depth must be 1 or greater: {depth}")
        else:
            raise ValueError(f"Invalid action (too many arguments): {action}")

        return listdir_path, depth
