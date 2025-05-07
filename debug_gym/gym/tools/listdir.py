from os.path import join as pjoin
from pathlib import Path

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class ListdirTool(EnvironmentTool):
    name: str = "listdir"
    examples = [
        """listdir(path=None, depth=None) to list the contents of the working directory.""",
        """listdir(path="src/util", depth=None) to list the contents of the 'util' subdirectory within the 'src' subdirectory.""",
        """listdir(path="src", depth=2) to list the contents of the 'src' subdirectory up to a depth of 2.""",
    ]
    arguments = {
        "path": {
            "type": ["string", "null"],
            "description": "The path to the subdirectory within the working directory. If None, the current working directory will be used.",
        },
        "depth": {
            "type": ["number", "null"],
            "description": "The maximum depth to which the directory tree should be explored. If None, the default depth will be used.",
        },
    }

    @property
    def description(self):
        assert hasattr(self, "environment")
        description = (
            f"List the file and folder contents of a subdirectory within the working directory, up to a specified 'depth' (default depth is {self.environment.dir_tree_depth}). The path should be relative to the working directory. If no path is provided, the current working directory will be used. If no depth is provided, the default depth will be used."
            + "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):\n"
            + "\n".join(self.examples)
        )
        return description

    def use(self, path: str = ".", depth: int = None) -> Observation:
        if depth is None:
            depth = self.environment.dir_tree_depth
        if depth <= 0:
            return Observation(self.name, f"Depth must be 1 or greater: {depth}")
        try:
            startpath = pjoin(self.environment.working_dir, path)
            result = self.environment.directory_tree(root=startpath, max_depth=depth)
        except ValueError as e:
            Observation(self.name, f"Depth must be 1 or greater: {str(e)}")
        return Observation(self.name, result)
