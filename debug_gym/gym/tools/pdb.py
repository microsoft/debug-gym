import copy
import re

from debug_gym.gym.entities import Observation
from debug_gym.gym.terminal import ShellSession
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox


@Toolbox.register()
class PDBTool(EnvironmentTool):
    name: str = "pdb"
    examples = [
        """pdb(command="b mttl/models/modifiers/mlp.py:42") to set a breakpoint at line 42 in the file with the path 'mttl/models/modifiers/mlp.py'.""",
        """pdb(command="c") to continue the execution until the next breakpoint.""",
        """pdb(command="p x") to print the value of the variable x in the current context.""",
        """pdb(command="cl src/code.py:26") to clear the breakpoint at line 26 in the file 'src/code.py'.""",
    ]
    description = (
        "An interface to the Python debugger PDB. Send a command to the PDB terminal. The command should be a valid PDB command."
        + "\nWhen using the breakpoint command (e.g., 'b', 'break', 'cl', 'clear'), make sure you specify the file path and line number in the format `file_path:line_number`."
        + "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):"
        + "\n".join(examples)
    )
    arguments = {
        "command": {
            "type": ["string"],
            "description": "The command to be sent to the PDB terminal. The command should be a valid PDB command. See https://docs.python.org/3/library/pdb.html for more information.",
        },
    }

    def __init__(self):
        super().__init__()
        self.current_frame_file = None
        self._session: ShellSession = None

    def __getstate__(self):
        """Handles serialisation of the PDBTool instance (for pickle) without un-picklable attributes"""
        state = self.__dict__.copy()
        for k in ["_session", "current_frame_file"]:
            del state[k]
        return state

    def __setstate__(self, state):
        """Handles de-serialisation of the PDBTool instance (for pickle) without un-picklable attributes"""
        self.__dict__.update(state)
        self.current_frame_file = None
        self._session = None

    def __deepcopy__(self, memo):
        """Create a deep copy of the PDBTool instance with _session set to None."""
        result = type(self).__new__(self.__class__)
        memo[id(self)] = result
        # Copy all attributes except _session
        for k, v in self.__dict__.items():
            # drop the session which is not serializable
            if k == "_session":
                setattr(result, k, None)
            # drop the current_frame_file which is None at the beginning
            # and will be set when the PDB session starts
            elif k == "current_frame_file":
                setattr(result, k, None)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    @property
    def pdb_is_running(self):
        return self._session is not None and self._session.is_running

    def interact_with_pdb(self, command: str, timeout: int):
        try:
            output = self._session.run(command, read_until="(Pdb)", timeout=timeout)
        except TimeoutError as e:
            output = f"The command `{command}` has timed out. {e!r}"

        return output.replace("(Pdb)", "").strip()  # remove the prompt

    def close_pdb(self):
        self._session.close()
        self.current_frame_file = None

    def start_pdb(self, environment) -> str:
        self._session = environment.terminal.new_shell_session()
        # init pdb and wait for the prompt
        initial_output = self._session.start(
            environment.debug_entrypoint, read_until="(Pdb)"
        )

        if "The program finished and will be restarted" in initial_output:
            self.close_pdb()
        else:
            if environment.persistent_breakpoints:
                # restore persistent breakpoints
                for _, _command in environment.current_breakpoints_state.items():
                    self.interact_with_pdb(_command, environment.run_timeout)
                if len(environment.current_breakpoints_state) > 0:
                    initial_output = "\n".join(
                        [initial_output, "Breakpoints have been restored."]
                    )
            self.set_current_frame_file(environment)
        return initial_output

    def on_env_reset(self, environment, **kwargs) -> Observation:
        super().on_env_reset(environment, **kwargs)
        obs = self.start_pdb(environment)
        return Observation(self.name, obs)

    def on_rewrite_success(
        self, environment, file, head, tail, length, **kwargs
    ) -> Observation:
        self.breakpoint_modify(environment, file, head, tail, length)
        obs = self.restart_pdb(environment)
        obs = "\nDebugging terminal started:\n" f"{obs}\n"
        return Observation(self.name, obs)

    def restart_pdb(self, environment) -> str:
        """Restart the pdb session and restore the breakpoints."""
        self.close_pdb()
        return self.start_pdb(environment)

    def use(self, environment, command: str) -> Observation:
        if command == "":
            return Observation(
                self.name, "Failure calling pdb:\nEmpty commands are not allowed."
            )

        _warning = ""
        # if print, it's OK to have ";" or "\n" in the command
        # otherwise, only the first command will be executed
        if not (command.split()[0] in ["p", "pp"] or command.startswith("print(")):
            splits = re.split("\n|;", command)
            if len(splits) > 1:
                command = splits[0].strip()
                _warning += "Multiple commands are not supported. Only the first command will be executed."

        success, output = True, ""
        if not self.pdb_is_running:
            output += self.start_pdb(environment)

        if not self.pdb_is_running:
            # pdb failed to start
            return Observation(self.name, f"Failure calling pdb:\n{output}")

        if command in ["b", "break"]:
            # list all breakpoints
            success, output = (
                True,
                f"Breakpoints:\n{environment.current_breakpoints()}\n",
            )
        elif command in ["cl", "clear"]:
            # clear all breakpoints
            environment.current_breakpoints_state = {}
            self.restart_pdb(environment)
            success, output = True, "All breakpoints have been cleared."
        else:  # other pdb commands, send directly
            try:
                pdb_out = self.interact_with_pdb(command, environment.run_timeout)
                # remove the working dir from the output
                pdb_out = pdb_out.replace(f"{environment.working_dir}/", "")
                if pdb_out in (
                    "End of file",
                    "Blank or comment",
                    "*** Blank or comment",
                ):
                    # if out of bounds, pdb will return "End of file"
                    # https://github.com/python/cpython/blob/main/Lib/pdb.py#L1464-L1485
                    success = False
                    output = f"Invalid line number: {pdb_out}."
                else:
                    output += f"Pdb command output:\n{pdb_out}"
                self.update_breakpoints(environment)
            except Exception:
                success = False

        if not success:
            if _warning:  # prevend additional \n
                obs = f"Invalid pdb command: {command}\n{_warning}\n{output.strip()}"
            else:
                obs = f"Invalid pdb command: {command}\n{output.strip()}"
            return Observation(self.name, obs)

        # sometimes it will run into the end of the program
        # we need to put the stdout before:
        # The program exited via sys.exit().
        # into self.last_eval_output, and remove them from the output
        if "The program exited via sys.exit()." in output:
            # end index is the last occurrence of the program exited (from the \n after)
            start_index = output.rfind("The program exited via sys.exit().")
            end_index = output.find("\n", start_index) + 1
            output = (
                output[:start_index]
                + "\nReached the end of the program. Restarting the debugging session.\n"
                + output[end_index:]
            )
        if _warning:
            obs = f"{_warning}\n{output.strip()}\n"
        else:
            obs = f"{output.strip()}\n"

        # Add the current frame information to the observation.
        if self.pdb_is_running:
            # read the current frame info to determine the current file
            current_frame = self.set_current_frame_file(environment)

            # free 'list' to provide context around the current frame
            list_output = ""
            if environment.auto_list and command.split()[0] not in ["l", "list"]:
                list_output = self.interact_with_pdb("l .", environment.run_timeout)

            if current_frame:
                obs += f"\nCurrent frame:\n{current_frame}\n"
            if list_output:
                indented_output = self._indent_first_line(list_output)
                obs += f"\nContext around the current frame:\n{indented_output}\n"

        return Observation(self.name, obs)

    def _indent_first_line(self, list_output: str) -> str:
        """Add indentation to the first line of the list output to match the
        indentation of the other lines, based on the second line's indentation."""

        lines = list_output.splitlines()
        # Check if we have enough lines to process
        if len(lines) <= 1:
            return list_output

        # Get the first two lines for comparison
        first_line = lines[0]
        second_line = lines[1]

        # Find the spaces at the beginning of both lines
        first_line_match = re.match(r"^(\s*)(\d+)", first_line)
        second_line_match = re.match(r"^(\s*)(\d+)", second_line)

        if first_line_match and second_line_match:
            first_spaces = first_line_match.group(1)
            second_spaces = second_line_match.group(1)

            # If first line has fewer spaces, add the difference
            if len(first_spaces) < len(second_spaces):
                spaces_to_add = second_spaces[len(first_spaces) :]
                return spaces_to_add + list_output

        # If no adjustment needed, return original
        return list_output

    def breakpoint_modify(
        self, environment, rewrite_file, rewrite_head, rewrite_tail, new_code_length
    ):
        # handle breakpoints line number changes caused by rewriting
        # this is a wrapper that manages the self.breakpoints_state, which does not reset at each pseudo terminal start
        # self.breakpoints_state is a dict, the keys are "|||".join([file_path, str(line_number)]) and values are breakpoint_command
        if len(environment.current_breakpoints_state) == 0:
            return
        current_breakpoints_state_copy = copy.deepcopy(
            environment.current_breakpoints_state
        )
        rewrite_file = environment.resolve_path(rewrite_file)
        for _key in environment.current_breakpoints_state.keys():
            _file_path, _line_number = _key.split("|||")
            _file_path = environment.resolve_path(_file_path)
            if _file_path != rewrite_file:
                # the breakpoints are not in the current file, no need to modify
                continue
            _line_number = int(_line_number)
            if rewrite_head is None:
                # no line number is provided, rewrite the whole code
                # we remove all breakpoints in the current file
                del current_breakpoints_state_copy[_key]
            else:
                # if a breakpoint was set in between the rewritten code, we need to remove it
                if rewrite_head <= _line_number <= rewrite_tail:
                    del current_breakpoints_state_copy[_key]
                # if a breakpoint was set after the rewritten code, we need to move it
                elif _line_number > rewrite_tail:
                    new_line_number = (
                        _line_number
                        + new_code_length
                        - (rewrite_tail - rewrite_head + 1)
                    )
                    new_key = "|||".join([str(_file_path), str(new_line_number)])
                    _new_value = environment.current_breakpoints_state[_key].split(":")
                    _new_value[1] = " ".join(
                        [str(new_line_number), " ".join(_new_value[1].split()[1:])]
                    )
                    current_breakpoints_state_copy[new_key] = ":".join(
                        _new_value
                    ).strip()
                    del current_breakpoints_state_copy[_key]
                # if a breakpoint was set before the rewritten code, we don't need to do anything
                else:
                    pass
        environment.current_breakpoints_state = current_breakpoints_state_copy

    def update_breakpoints(self, environment):
        """Updates the environment's current_breakpoints_state by
        parsing the output of the PDB 'b' (breakpoints) command.

        The new_breakpoints dictionary keys are in the format "file_path|||line_number",
        and the values are the corresponding PDB breakpoint commands.

        environment.current_breakpoints_state = {
            "path/to/file.py|||line_number": "b path/to/file.py:line_number",
            ...
        }"""

        command = "b"  # list all breakpoints
        output = self.interact_with_pdb(command, environment.run_timeout)
        # parse the output to update the current_breakpoints_state
        # example output:
        # Num Type         Disp Enb   Where
        # 1   breakpoint   keep yes   at /tmp/RepoEnv-_ha8r7_2/constants.py:6
        # 2   breakpoint   keep yes   at /tmp/RepoEnv-_ha8r7_2/constants.py:10
        # 3   breakpoint   keep yes   at /tmp/RepoEnv-_ha8r7_2/constants.py:14
        # -> ACTION_TO_INDEX = {
        new_breakpoints = {}
        breakpoint_pattern = re.compile(
            r"^\s*\d+\s+breakpoint\s+keep\s+yes\s+at\s+(.+):(\d+)$"
        )
        for line in output.splitlines():
            match = breakpoint_pattern.match(line)
            if match:
                # extract the file path and line number from the regex match
                file_path, line_number = match.groups()
                key = "|||".join([file_path, line_number])
                new_breakpoints[key] = f"b {file_path}:{line_number}"
        environment.current_breakpoints_state = new_breakpoints

    def set_current_frame_file(self, environment) -> str | None:
        """A free 'where' to obtain the current frame (line number), hidden from the agent."""
        command = "where"
        output = self.interact_with_pdb(command, environment.run_timeout)
        # parse the output to get the current frame
        # example output:
        #    /home/eryua/venvs/pdb/lib/python3.12/bdb.py(606)run()
        # -> exec(cmd, globals, locals)
        #    <string>(1)<module>()
        # > /tmp/RepoEnv-_ha8r7_2/constants.py(6)<module>()
        # -> ACTION_TO_INDEX = {
        sep = "> "
        file_path = None
        for line in output.splitlines():
            # find the line that starts with "> "
            if line.startswith(sep):
                # extract the file path from the line,
                # remove the leading "> ", the trailing "(line_number)<module>()", and working_dir
                # constants.py(6)<module>()
                # -> ACTION_TO_INDEX = {
                file_path = line[len(sep) :].split("(")[0]
                break
        if self.current_frame_file != file_path:
            self.current_frame_file = file_path
        return file_path
