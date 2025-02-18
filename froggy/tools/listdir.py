from os.path import join as pjoin
from pathlib import Path

from froggy.tools.tool import EnvironmentTool
from froggy.tools.toolbox import Toolbox


@Toolbox.register()
class ListdirTool(EnvironmentTool):
    name: str = "listdir"

    @property
    def instructions(self):
        assert hasattr(self, "environment")
        instruction = {
            "template": """listdir(path: str, depth: int)""",
            "description": f"List the file and folder contents of a subdirectiory specified by 'path', up to a specified 'depth'. The default 'path' is the working directory, and the default 'depth' is {self.environment.dir_tree_depth}. The path must be relative to the working directory, and the depth must be 1 or greater.",
            "examples": [
                """listdir() to list the contents of the working directory.""",
                """listdir("src/util") to list the contents of the 'util' subdirectory within the 'src' subdirectory.""",
                """listdir(path="src", depth=2) to list the contents of the 'src' subdirectory up to a depth of 2.""",
                """listdir("src/test", 1) to list the contents of the 'src/test' subdirectory up to a depth of 1.""",
            ],
        }
        return instruction

    def use(self, path: str = ".", depth: int = None):
        if depth is None:
            depth = self.environment.dir_tree_depth
        if depth <= 0:
            raise ValueError(f"Depth must be 1 or greater: {depth}")
        try:
            startpath = pjoin(self.environment.working_dir, path)
            result = self.environment.directory_tree(root=startpath, max_depth=depth)
        except ValueError as e:
            return str(e)

        return result
