from os.path import join as pjoin

from froggy.tools import EnvironmentTool
from .toolbox import Toolbox

@Toolbox.register()
class ListdirTool(EnvironmentTool):
    name: str = "listdir"
    action: str = "```listdir"
    description: str = "List the directory tree at a given subdirectory."

    @property
    def instructions(self):
        assert hasattr(self, "environment")
        instruction = {
            "template": "```listdir <path/to/subdirectory>```",
            "description": f"List the file and folder contents of a subdirectory within the working directory, up to a maximum depth {self.environment.dir_tree_depth}.",
            "examples": [
                f"```listdir``` to list the contents of the working directory.",
                f"```listdir src``` to list the contents of the 'src' subdirectory.",
                f"```listdir src/util``` to list the contents of the 'util' subdirectory within the 'src' subdirectory.",
            ],
        }
        return instruction

    def register(self, environment):
        from froggy.envs import RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        self.environment = environment

    def is_triggered(self, action):
        return action.startswith(self.action)

    def use(self, action):
        clean_action = self.clean_action(action)
        startpath = pjoin(self.environment.working_dir, clean_action)
        result = self.environment.directory_tree(root=startpath)
        return f"Listdir at {clean_action}:\n{result}"

    def clean_action(self, action):
        listdir_path = action.strip("`").strip()
        tmp = listdir_path.split(" ", 1)
        if len(tmp) == 1:
            # e.g., ```listdir```
            listdir_path = "."
        else:
            # e.g., ```listdir src```
            listdir_path = tmp[1].strip()
        return listdir_path
