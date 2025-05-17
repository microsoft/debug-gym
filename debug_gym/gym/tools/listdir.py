from os.path import join as pjoin

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
    description = (
        "List the file and folder contents of a subdirectory within the working directory, up to a specified 'depth' (default depth is 1). "
        "The path should be relative to the working directory. If no path is provided, the current working directory will be used. "
        "If no depth is provided, the default depth will be used."
        "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):\n"
        f"{'\n'.join(examples)}"
    )
    arguments = {
        "path": {
            "type": ["string", "null"],
            "description": "The path to the subdirectory within the working directory. If None, the current working directory will be used.",
        },
        "depth": {
            "type": ["number", "null"],
            "description": "The maximum depth to which the directory tree should be explored. Default depth is 1.",
        },
    }

    def use(self, environment, path: str = ".", depth: int = 1) -> Observation:
        # TODO: reimplement dir_tree_depth via ListdirTool __init__
        # if depth is None:
        #     depth = environment.dir_tree_depth
        if depth <= 0:
            return Observation(self.name, f"Depth must be 1 or greater: {depth}")
        try:
            startpath = pjoin(environment.working_dir, path)
            result = environment.directory_tree(root=startpath, max_depth=depth)
        except ValueError as e:
            return Observation(self.name, f"Depth must be 1 or greater: {str(e)}")
        return Observation(self.name, result)
