import difflib

from debug_gym.gym.entities import Event, Observation
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class CreateTool(EnvironmentTool):
    name = "create"
    examples = [
        """create(path="code/newfile.py", content="print('hola')") will rewrite the specified file 'code/newfile.py' (the entire code) to be print('hola'), because no line number is provided.""",
        """create(path="code/file.py", content="    print('hello')\\n    print('hi again')", overwrite=True) will create a new file at the path 'code/existingfile.py' containing the two lines provided, both with indents ahead (in this case, 4 spaces). Additionally, with overwrite=True, if the file already exists, it will be overwritten with the new content.""",
    ]
    description = (
        "Create a new file with the specified file path, with the content. Optionally overwrite existing files with new content. The new code should be valid python code include proper indentation (can be determined from context)."
        + "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):"
        + "\n".join(examples)
    )
    arguments = {
        "path": {
            "type": ["string"],
            "description": "A file path to be rewritten.",
        },
        "content": {
            "type": ["string"],
            "description": "The content of the new file. The code should be valid python code include proper indentation (can be determined from context).",
        },
        "overwrite": {
            "type": ["boolean", "null"],
            "description": "If True, overwrite the existing file. If False, create a new file.",
        },
    }

    def _create_file(self, environment, file_path: str, content: str):
        original_content = ""
        if environment.workspace.has_file(file_path):
            original_content = environment.workspace.read_file(file_path)

        environment.workspace.write_file(file_path, content)

        # Calculate diff between original and new content
        new_content = environment.workspace.read_file(file_path)
        diff = "".join(
            difflib.unified_diff(
                original_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile="original",
                tofile="current",
            )
        )

        return diff

    def fail(self, environment, message: str) -> Observation:
        self.create_success = False
        message = f"Create failed. Error message:\n{message}\n"
        self.queue_event(
            environment=environment,
            event=Event.CREATE_FAIL,
            message=message,
        )
        return Observation(self.name, message)

    def use(
        self,
        environment,
        path: str = None,
        content: str = "",
        overwrite: bool = False,
    ) -> Observation:
        self.create_success = False
        if path is None:
            return self.fail(environment, "File path is None.")
        if environment.workspace.has_file(path):
            if not overwrite:
                return self.fail(
                    environment,
                    "File already exists. To overwrite, Please specify overwrite=True.",
                )
            if not environment.workspace.is_editable(path):
                return self.fail(environment, f"`{path}` is not editable.")

        abs_filepath = environment.workspace.resolve_path(path)
        if environment.workspace._is_ignored_func(abs_filepath):
            return self.fail(
                environment,
                f"`{path}` is ignored by the ignore patterns and cannot be created.",
            )

        try:
            diff = self._create_file(environment, path, content)
        except Exception as e:
            return self.fail(environment, str(e))

        self.create_success = True
        message = f"The file `{path}` has been created successfully.\n\nDiff:\n\n{diff}"

        self.queue_event(
            environment=environment,
            event=Event.CREATE_SUCCESS,
            message=message,
            file=path,
        )
        return Observation(self.name, message)
