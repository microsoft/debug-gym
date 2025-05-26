import copy
import re

from debug_gym.gym.entities import Observation
from debug_gym.gym.terminal import ShellSession
from debug_gym.gym.tools.tool import EnvironmentTool
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.gym.utils import get_code_length


@Toolbox.register()
class PDBTool(EnvironmentTool):
    name: str = "pdb"
    examples = [
        """pdb(command="p x") to print the value of the variable x in the current context.""",
        """pdb(command="b 42") to set a breakpoint at line 42 in the current file.""",
        """pdb(command="cl src/code.py:26") to clear the breakpoint at line 26 in the file src/code.py.""",
        """pdb(command="c") to continue the execution until the next breakpoint.""",
    ]
    description = (
        "An interface to the Python debugger PDB. Send a command to the PDB terminal. The command should be a valid PDB command."
        + "\nExamples (for demonstration purposes only, you need to adjust the tool calling format according to your specific syntax):"
        + "\n".join(examples)
    )
    arguments = {
        "command": {
            "type": ["string"],
            "description": "The command to be sent to the PDB terminal. The command should be a valid PDB command. See https://docs.python.org/3/library/pdb.html for more information.",
        },
    }

    def __init__(
        self,
        persistent_breakpoints: bool = False,
        auto_list: bool = True,
    ):
        super().__init__()
        self.pdb_obs = ""
        self.persistent_breakpoints = persistent_breakpoints
        self.auto_list = auto_list
        self.current_frame_file = None
        self._session: ShellSession = None

    def __deepcopy__(self, memo):
        """Create a deep copy of the PDBTool instance with _session set to None."""
        result = type(self).__new__(self.__class__)
        memo[id(self)] = result
        # Copy all attributes except _session
        for k, v in self.__dict__.items():
            if k == "_session":
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

    def start_pdb(self, environment) -> str:
        self._session = environment.terminal.new_shell_session()
        # init pdb and wait for the prompt
        initial_output = self._session.start(
            environment.debug_entrypoint, read_until="(Pdb)"
        )

        if "The program finished and will be restarted" in initial_output:
            self.close_pdb()
        else:
            if self.persistent_breakpoints:
                # restore persistent breakpoints
                for _, _command in environment.current_breakpoints_state.items():
                    self.interact_with_pdb(_command, environment.run_timeout)
                if len(environment.current_breakpoints_state) > 0:
                    initial_output = "\n".join(
                        [initial_output, "Breakpoints have been restored."]
                    )
        self.pdb_obs = initial_output
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
        _warning = ""
        if (
            command == ""
            or command.split()[0] in ["p", "pp"]
            or command.startswith("print(")
        ):
            # OK to have ";" or "\n" in the command
            pass
        else:
            splits = re.split("\n|;", command)
            if len(splits) > 1:
                command = splits[0].strip()
                _warning += "Multiple commands are not supported. Only the first command will be executed."

        success, output = True, ""
        if not self.pdb_is_running:
            output += self.start_pdb(environment)

        if not self.pdb_is_running:
            return Observation(self.name, f"Failure calling pdb:\n{output}")
        if command == "":  # empty command
            return Observation(
                self.name, "Failure calling pdb:\nEmpty commands are not allowed."
            )

        if command in ["b", "break"]:
            # list all breakpoints
            success, output = True, environment.current_breakpoints()
        elif command in ["cl", "clear"]:
            # clear all breakpoints
            environment.current_breakpoints_state = {}
            self.restart_pdb(environment)
            success, output = True, "All breakpoints have been cleared."
        elif (
            command.split()[0] in ["b", "break", "cl", "clear"]
            and command.split()[1].isnumeric()
        ):
            # wrapper handle adding/removing breakpoints
            # TODO: Not sure we can or should use self.current_frame_file here
            success, output = self.breakpoint_add_clear(
                environment, command, self.current_frame_file
            )
        elif (
            command.split()[0] in ["b", "break", "cl", "clear"]
            and ":" in command.split()[1]
            and command.split()[1].split(":")[1][0].isnumeric()
        ):
            # e.g., b src/main.py:42 some_other_args
            which_file, _bp_args = command.split(maxsplit=1)[1].split(":")
            _command_without_file = f"{command.split()[0]} {_bp_args}"
            success, output = self.breakpoint_add_clear(
                environment, _command_without_file, which_file
            )
        else:
            # other pdb commands, send directly
            try:
                output += self.interact_with_pdb(command, environment.run_timeout)
                self.pdb_obs = output
            except Exception:  # TODO: catch specific exceptions
                success = False

        if success:
            # sometimes it will run into the end of the program
            # we need to put the stdout before:
            # The program exited via sys.exit().
            # into self.last_eval_output, and remove them from the output
            if "The program exited via sys.exit()." in output:
                # end index is the last occurrence of the program exited (from the \n after)
                end_index = (
                    output.find(
                        "\n", output.rfind("The program exited via sys.exit().")
                    )
                    + 1
                )
                output = (
                    "Reached the end of the file. Restarting the debugging session.\n"
                    + output[end_index:]
                )
            obs = "\n".join([_warning, output]).strip()

            # Add the current frame information to the observation.
            if (
                self.pdb_is_running
                and self.auto_list
                and command.split()[0] not in ["l", "list"]
            ):
                if '"""The pytest entry point."""' not in obs:
                    obs += "\nlist .\n" + self.interact_with_pdb(
                        "l .", environment.run_timeout
                    )
        else:
            obs = "\n".join([f"Invalid pdb command: {command}", _warning, output])

        if self.pdb_is_running:
            # read the current frame info, find the current file, so we can change view to that file.
            self.get_current_frame_file(environment)

        return Observation(self.name, obs)

    def breakpoint_add_clear(self, environment, action: str, which_file):
        # handle adding/removing breakpoints
        # this is a wrapper that manages the self.breakpoints_state, which does not reset at each pseudo terminal start
        # self.breakpoints_state is a dict, the keys are "|||".join([file_path, str(line_number)]) and values are breakpoint_command
        # TODO: we don't support tbreak
        manipulation = "set" if action.startswith("b") else "clear"
        if which_file.startswith(str(environment.working_dir)):
            which_file = which_file[len(str(environment.working_dir)) + 1 :]
        if which_file not in environment.all_files:
            return (
                False,
                f"Failed to {manipulation} breakpoint. `{which_file}` is not found in the repository.",
            )
        # IMPORTANT: insert the viewing file into breakpoint command
        # for example, "b 42" -> "b src/main.py:42" if the current file is "src/main.py"
        # for example, "cl 42" -> "cl src/main.py:42" if the current file is "src/main.py"
        action_split = action.split(maxsplit=2)
        _action_type, _line_number, _bp_args = (
            action_split[0],
            action_split[1],
            action_split[2] if len(action_split) > 2 else "",
        )
        _key = "|||".join([which_file, _line_number])
        assert _line_number.isnumeric()
        success, output = True, ""
        joined_args = " ".join([_line_number, _bp_args])
        command = f"{which_file}:{joined_args}".strip()
        if _action_type in ["b", "break"]:
            command = "b " + command
            if _key in environment.current_breakpoints_state.keys():
                # breakpoint already exists
                return (
                    True,
                    f"Breakpoint already exists at line {_line_number} in `{which_file}`.",
                )
            else:
                # check if line number is valid
                code_string = environment.read_file(which_file)
                code_length = get_code_length(code_string)
                if int(_line_number) > code_length or int(_line_number) < 1:
                    return (
                        False,
                        f"Invalid line number: {_line_number}, expected between 1 and {code_length}.",
                    )
                try:
                    output = self.interact_with_pdb(command, environment.run_timeout)
                    self.pdb_obs = output
                    # when success, the output always repeats the command, we can remove it
                    output = output.strip()
                    if output.startswith(command):
                        output = output[len(command) :].strip()
                    environment.current_breakpoints_state[_key] = command
                except BaseException:
                    success = False
        elif _action_type in ["cl", "clear"]:
            command = "cl " + command
            if _key not in environment.current_breakpoints_state.keys():
                # breakpoint does not exist
                return (
                    True,
                    f"No breakpoint exists at line {_line_number} in `{which_file}`.",
                )
            else:
                try:
                    output = self.interact_with_pdb(command, environment.run_timeout)
                    self.pdb_obs = output
                    # when success, the output always repeats the command, we can remove it
                    output = output.strip()
                    if output.startswith(command):
                        output = output[len(command) :].strip()
                    del environment.current_breakpoints_state[_key]
                except BaseException:
                    success = False
        else:
            return False, output

        return success, output

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
        if rewrite_file.startswith(str(environment.working_dir)):
            rewrite_file = rewrite_file[len(str(environment.working_dir)) + 1 :]
        for _key in environment.current_breakpoints_state.keys():
            _file_path, _line_number = _key.split("|||")
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
                    new_key = "|||".join([_file_path, str(new_line_number)])
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

    def get_current_frame_file(self, environment):
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
        sep = "> " + str(environment.working_dir) + "/"
        if sep not in output:
            return
        output = output.rsplit(sep, 1)[1]
        # constants.py(6)<module>()
        # -> ACTION_TO_INDEX = {
        try:
            file_path = output.split("(")[0]
            if file_path != self.current_frame_file:
                self.current_frame_file = file_path
        except BaseException:
            pass
