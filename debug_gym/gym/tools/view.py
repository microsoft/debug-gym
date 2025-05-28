import os
from os.path import join as pjoin

from debug_gym.gym.entities import Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.gym.utils import is_subdirectory, show_line_number


@Toolbox.register()
class ViewTool(EnvironmentTool):
    name: str = "view"
    examples = [
        """view(path="main.py") to show the content of a file called 'main.py' in the root. The content will be annotated with line numbers and current breakpoints because include_line_numbers_and_breakpoints is True by default.""",
        """view(path="utils/vector.py", include_line_numbers_and_breakpoints=True) to show the content of a file called 'vector.py' in a subdirectory called 'utils'. The content will be annotated with line numbers and current breakpoints.""",
        """view(path="src/util.py", include_line_numbers_and_breakpoints=False) to show the content of a file called 'util.py' in a subdirectory called 'src'. The line numbers and breakpoints will not be included in the output.""",
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
        "include_line_numbers_and_breakpoints": {
            "type": ["boolean", "null"],
            "description": "Whether to annotate the file content with line numbers and current breakpoints before each line of code. For example, a line can be shown as 'B  426         self.assertEqual(CustomUser._default_manager.count(), 0)', where 'B' indicates a breakpoint before this line of code. '426' is the line number. This argument is optional and defaults to True. If set to False, the file content will be shown without line numbers and breakpoints.",
        },
    }

    def use(
        self,
        environment,
        path: str,
        include_line_numbers_and_breakpoints: bool = True,
    ) -> Observation:
        # TODO: Decide whether to use natural language or json like format for the tool call observation.
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
            file_content = environment.read_file(new_file)
            breakpoints_message = ""
            if include_line_numbers_and_breakpoints:
                file_content = show_line_number(
                    file_content,
                    code_path=new_file,
                    breakpoints_state=environment.current_breakpoints_state,
                )
                if environment.current_breakpoints_state:
                    breakpoints_message = (
                        ". B indicates breakpoint before a certain line of code"
                    )
            read_only = " (read-only)" if not environment.is_editable(new_file) else ""
            obs = (
                f"Viewing `{new_file}`{read_only}{breakpoints_message}:"
                f"\n\n```\n{file_content}\n```\n\n"
            )
        else:
            obs = (
                f"File not found. Could not navigate to `{new_file}`. "
                f"Make sure that the file path is given relative to the root: `{environment.working_dir}`."
            )

        return Observation(self.name, obs)
