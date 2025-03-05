from os.path import join as pjoin
from pathlib import Path

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class ListdirTool(EnvironmentTool):
    name: str = "listdir"

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

    def use(self, tool_args) -> Observation:
        try:
            listdir_path, depth = self.parse_args(tool_args)
            startpath = pjoin(self.environment.working_dir, listdir_path)
            obs = self.environment.directory_tree(root=startpath, max_depth=depth)
        except ValueError as e:
            obs = str(e)
        return Observation(self.name, obs)

    def parse_args(self, tool_args):
        depth = None
        if tool_args == "":
            # e.g., ```listdir```
            listdir_path = "."
        else:
            arg_list = tool_args.split(" ")
            if len(arg_list) == 1:
                # e.g., ```listdir src```
                listdir_path = arg_list[0].strip()
            elif len(arg_list) == 2:
                # e.g., ```listdir src depth```
                listdir_path = arg_list[0].strip()
                depth = int(arg_list[1].strip())
                if depth <= 0:
                    raise ValueError(f"Depth must be 1 or greater: {depth}")
            else:
                raise ValueError(f"Invalid action (too many arguments): {tool_args}")

        return listdir_path, depth
