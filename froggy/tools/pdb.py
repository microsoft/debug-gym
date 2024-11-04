import copy
import errno
import fcntl
import os
import pty
import re
import subprocess
import time
from typing import Optional

from froggy.tools import EnvironmentTool
from froggy.utils import TimeoutException, time_limit


class PDBTool(EnvironmentTool):
    name: str = "pdb"
    action: str = "```pdb"
    description: str = "A suite of PDB commands available to the user."
    instructions = {
        "template": "```pdb <command>```",
        "description": "Send a command to the PDB terminal. The command should be a valid PDB command.",
        "examples": [
            "```pdb p x``` to print the value of the variable x in the current context.",
            "```pdb b 42``` to set a breakpoint at line 42 in the current file.",
            "```pdb cl src/code.py:26``` to clear the breakpoint at line 26 in the file src/code.py.",
            "```pdb c``` to continue the execution until the next breakpoint.",
        ],
        "commonly used pdb commands": {
            "note": "The commands recognized by the debugger are listed below. Most commands can be abbreviated to one or two letters as indicated; e.g. h(elp) means that either h or help can be used to enter the help command (but not he or hel, nor H or Help or HELP). Arguments to commands must be separated by whitespace (spaces or tabs). Optional arguments are enclosed in square brackets ([]) in the command syntax; the square brackets must not be typed. Alternatives in the command syntax are separated by a vertical bar (|).",
            "commands": {
                "h(elp) [command]": "Without argument, print the list of available commands. With a command as argument, print help about that command.",
                "w(here)": "Print a stack trace, with the most recent frame at the bottom. An arrow (>) indicates the current frame, which determines the context of most commands.",
                "d(own) [count]": "Move the current frame count (default one) levels down in the stack trace (to a newer frame).",
                "u(p) [count]": "Move the current frame count (default one) levels up in the stack trace (to an older frame).",
                "b(reak) [([filename:]lineno | function) [, condition]]": "With a lineno argument, set a break at line lineno in the current file. The line number may be prefixed with a filename and a colon, to specify a breakpoint in another file. Accepatable forms of filename are /abspath/to/file.py, relpath/file.py, module and package.module. With a function argument, set a break at the first executable statement within that function. function can be any expression that evaluates to a function in the current namespace. If a second argument is present, it is an expression which must evaluate to true before the breakpoint is honored. Without argument, list all breaks, including for each breakpoint, the number of times that breakpoint has been hit, the current ignore count, and the associated condition if any. Each breakpoint is assigned a number to which all the other breakpoint commands refer.",
                "cl(ear) [filename:lineno | bpnumber ...]": "With a filename:lineno argument, clear all the breakpoints at this line. With a space separated list of breakpoint numbers, clear those breakpoints. Without argument, clear all breaks.",
                "s(tep)": "Execute the current line, stop at the first possible occasion (either in a function that is called or on the next line in the current function).",
                "n(ext)": "Continue execution until the next line in the current function is reached or it returns. (The difference between next and step is that step stops inside a called function, while next executes called functions at (nearly) full speed, only stopping at the next line in the current function.)",
                "unt(il) [lineno]": "Without argument, continue execution until the line with a number greater than the current one is reached. With lineno, continue execution until a line with a number greater or equal to lineno is reached. In both cases, also stop when the current frame returns.",
                "c(ont(inue))": "Continue execution, only stop when a breakpoint is encountered.",
                "l(ist) [first[, last]]": "List source code for the current file. Without arguments, list 11 lines around the current line or continue the previous listing. With . as argument, list 11 lines around the current line. With one argument, list 11 lines around at that line. With two arguments, list the given range; if the second argument is less than the first, it is interpreted as a count.",
                "ll | longlist": "List all source code for the current function or frame. Interesting lines are marked as for list.",
                "a(rgs)": "Print the arguments of the current function and their current values.",
                "p expression": "Evaluate expression in the current context and print its value.",
                "pp expression": "Like the p command, except the value of expression is pretty-printed using the pprint module.",
                "whatis expression": "Print the type of expression.",
            },
        },
    }

    def __init__(
        self,
        persistent_breakpoints: bool = False,
        auto_list: bool = True,
    ):
        super().__init__()
        self.master = None
        self.pdb_obs = ""
        self.persistent_breakpoints = persistent_breakpoints
        self.auto_list = auto_list
        self.current_frame_file = None

    def register(self, environment):
        from froggy.envs import RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        self.environment = environment

    def is_triggered(self, action):
        # e.g. ```pdb b src/main.py:42```
        return action.startswith(self.action)

    def use(self, action):
        command = action.strip("`").split(" ", 1)[1].strip()
        if not self.has_pseudo_terminal():
            self.start_pseudo_terminal()

        _warning = ""
        splits = re.split("\n|;", command)
        if len(splits) > 1:
            command = splits.strip()
            _warning += f"Multiple commands are not supported. Only the first command will be executed.\n"

        success, output = True, ""
        if command in ["b", "break"]:
            # list all breakpoints
            success, output = True, self.current_breakpoints()
        elif command in ["cl", "clear"]:
            # clear all breakpoints
            self.environment.current_breakpoints_state = {}
            self.start_pseudo_terminal()
            success, output = True, "All breakpoints have been cleared."
        elif (
            command.split()[0] in ["b", "break", "cl", "clear"]
            and command.split()[1].isnumeric()
        ):
            # wrapper handle adding/removing breakpoints
            success, output = self.breakpoint_add_clear(command)
        elif (
            command.split()[0] in ["b", "break", "cl", "clear"]
            and ":" in command.split()[1]
            and command.split()[1].split(":")[1][0].isnumeric()
        ):
            # e.g., b src/main.py:42 some_other_args
            which_file, _bp_args = command.split(maxsplit=1)[1].split(":")
            _command_without_file = f"{command.split()[0]} {_bp_args}"
            success, output = self.breakpoint_add_clear(
                _command_without_file, which_file
            )
        else:
            # other pdb commands, send directly
            try:
                output = self.interact_with_pseudo_terminal(command)
            except:  # TODO: catch specific exceptions
                success = False

        if success:
            # sometimes it will run into the end of the program
            # we need to put the stdout before:
            # The program exited via sys.exit().
            # into self.last_eval_obs, and remove them from the output
            if "The program exited via sys.exit()." in output:
                # end index is the last occurrence of the program exited (from the \n after)
                end_index = (
                    output.find(
                        "\n", output.rfind("The program exited via sys.exit().")
                    )
                    + 1
                )
                self.environment.last_run_obs = output[:end_index]
                output = (
                    "Reached the end of the file. Restarting the debugging session.\n"
                    + output[end_index:]
                )
            obs = "\n".join([_warning, output]).strip()

            # Add the current frame information to the observation.
            if self.auto_list and command.split()[0] not in ["l", "list"]:
                if '"""The pytest entry point."""' not in obs:
                    obs += f" list .\n" + self.interact_with_pseudo_terminal("l .")
        else:
            obs = "\n".join([f"Invalid action: {action}", _warning, output])

        # read the current frame info, find the current file, so we can change view to that file.
        self.get_current_frame_file()
        return obs

    def current_breakpoints(self):
        if len(self.environment.current_breakpoints_state) == 0:
            return "No breakpoints are set."
        else:
            # print the breakpoints sorted by file names and line number
            breakpoints = []
            for _key in self.environment.current_breakpoints_state.keys():
                _file_path, _line_number = _key.split("|||")
                _line_number = int(_line_number)
                breakpoints.append([_file_path, _line_number])
            # sort by file name, if file names are same, sort by line number
            breakpoints = sorted(breakpoints, key=lambda x: (x[0], x[1]))
            breakpoints = [
                f"line {_line_number} in {_file_path}"
                for _file_path, _line_number in breakpoints
            ]
            return "\n".join(breakpoints)

    def breakpoint_add_clear(self, action: str, which_file=None):
        # handle adding/removing breakpoints
        # this is a wrapper that manages the self.breakpoints_state, which does not reset at each pseudo terminal start
        # self.breakpoints_state is a dict, the keys are "|||".join([file_path, str(line_number)]) and values are breakpoint_command
        # TODO: we don't support tbreak
        if which_file is None:
            which_file = self.environment.current_file
        manipulation = "set" if action.startswith("b") else "clear"
        if which_file is None:
            return (
                False,
                f"Failed to {manipulation} breakpoint. No file is currently open.",
            )
        if which_file.startswith(str(self.environment.working_dir)):
            which_file = which_file[len(str(self.environment.working_dir)) + 1 :]
        if which_file not in self.environment.all_files:
            return (
                False,
                f"Failed to {manipulation} breakpoint. {which_file} is not in the current repository.",
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
            if _key in self.environment.current_breakpoints_state.keys():
                # breakpoint already exists
                return (
                    True,
                    f"Breakpoint already exists at line {_line_number} in {which_file}.",
                )
            else:
                try:
                    output = self.interact_with_pseudo_terminal(command)
                    # when success, the output always repeats the command, we can remove it
                    output = output.strip()
                    if output.startswith(command):
                        output = output[len(command) :].strip()
                    self.environment.current_breakpoints_state[_key] = command
                except:
                    success = False
        elif _action_type in ["cl", "clear"]:
            command = "cl " + command
            if _key not in self.environment.current_breakpoints_state.keys():
                # breakpoint does not exist
                return (
                    True,
                    f"No breakpoint exists at line {_line_number} in {which_file}.",
                )
            else:
                try:
                    output = self.interact_with_pseudo_terminal(command)
                    # when success, the output always repeats the command, we can remove it
                    output = output.strip()
                    if output.startswith(command):
                        output = output[len(command) :].strip()
                    del self.environment.current_breakpoints_state[_key]
                except:
                    success = False
        else:
            return False, output

        return success, output

    def breakpoint_modify(self, rewrite_head, rewrite_tail, new_code_length):
        # handle breakpoints line number changes caused by rewriting
        # this is a wrapper that manages the self.breakpoints_state, which does not reset at each pseudo terminal start
        # self.breakpoints_state is a dict, the keys are "|||".join([file_path, str(line_number)]) and values are breakpoint_command
        if len(self.environment.current_breakpoints_state) == 0:
            return
        current_breakpoints_state_copy = copy.copy(
            self.environment.current_breakpoints_state
        )
        for _key in self.environment.current_breakpoints_state.keys():
            _file_path, _line_number = _key.split("|||")
            if _file_path != self.environment.current_file:
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
                    _new_value = self.environment.current_breakpoints_state[_key].split(
                        ":"
                    )
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
        self.environment.current_breakpoints_state = current_breakpoints_state_copy

    def set_fd_nonblocking(self):
        flags = fcntl.fcntl(self.master, fcntl.F_GETFL)
        fcntl.fcntl(self.master, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def has_pseudo_terminal(self):
        return self.master is not None

    def start_pseudo_terminal(self):
        if self.has_pseudo_terminal():
            self.close_pseudo_terminal()

        self.master, slave = pty.openpty()
        self.set_fd_nonblocking()
        debug_entrypoint = "python -m pdb".split() + self.environment.entrypoint[1:]
        _env = os.environ.copy()
        _env["NO_COLOR"] = "1"
        process = subprocess.Popen(
            debug_entrypoint,
            env=_env,
            cwd=self.environment.working_dir,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            text=True,
            close_fds=True,
        )

        # close slave, end in the parent process
        os.close(slave)

        # initial output from pdb
        initial_output = self.read_pseudo_terminal_output()
        if "The program finished and will be restarted" in initial_output:
            self.close_pseudo_terminal()
        else:
            if self.persistent_breakpoints:
                # restore consistent breakpoints
                for _, _command in self.environment.current_breakpoints_state.items():
                    self.interact_with_pseudo_terminal(_command)
                if len(self.environment.current_breakpoints_state) > 0:
                    initial_output = "\n".join(
                        ["Breakpoints have been restored.", initial_output]
                    )

        if self.auto_list:
            if '"""The pytest entry point."""' not in initial_output:
                initial_output += "list .\n" + self.interact_with_pseudo_terminal("l .")

        self.pdb_obs = initial_output
        return initial_output

    def close_pseudo_terminal(self):
        if self.master is not None:
            os.close(self.master)
            self.master = None

    def _read_pseudo_terminal_output(self):
        """Read from the PTY master until "(Pdb)" is found."""
        output = ""
        while True:
            try:
                data = os.read(self.master, 1024).decode(
                    "utf-8"
                )  # [TODO] 1024? We might need more than that.
                if data:
                    output += data
                    if "(Pdb)" in data:
                        break

            except BlockingIOError:
                time.sleep(0.01)
                continue

            except OSError as e:
                if e.errno == errno.EIO:
                    # end of file
                    break

                if e.errno != errno.EAGAIN:
                    raise

        return output

    def read_pseudo_terminal_output(self):
        """wrapper of _read_pseudo_terminal_output that catches timeout."""
        try:
            with time_limit(self.environment.run_timeout):
                return self._read_pseudo_terminal_output()
        except TimeoutException as e:
            return "Terminal output reading timeout."

    def interact_with_pseudo_terminal(self, command):
        # send command to pdb:
        os.write(self.master, command.encode("utf-8") + b"\n")
        # get output back:
        output = self.read_pseudo_terminal_output()

        # when success, the output always repeats the command, we can remove it
        output = output.strip()
        if output.startswith(command):
            output = output[len(command) :].strip("\r\n")

        if "The program finished and will be restarted" in output or command in [
            "exit",
            "quit",
            "q",
        ]:
            self.close_pseudo_terminal()
            restart_pseudo_terminal_output = self.start_pseudo_terminal()
            output = "\n".join([output, restart_pseudo_terminal_output])

        self.pdb_obs = output
        return output

    def get_current_frame_file(self):
        """A free 'where' to obtain the current frame (line number), hidden from the agent."""
        if not self.has_pseudo_terminal():
            self.start_pseudo_terminal()
        command = "where"
        os.write(self.master, command.encode("utf-8") + b"\n")
        # get output back:
        output = self.read_pseudo_terminal_output()

        # parse the output to get the current frame
        # example output:
        #    /home/eryua/venvs/pdb/lib/python3.12/bdb.py(606)run()
        # -> exec(cmd, globals, locals)
        #    <string>(1)<module>()
        # > /tmp/RepoEnv-_ha8r7_2/constants.py(6)<module>()
        # -> ACTION_TO_INDEX = {
        sep = "> " + str(self.environment.working_dir) + "/"
        if sep not in output:
            return
        output = output.rsplit(sep, 1)[1]
        # constants.py(6)<module>()
        # -> ACTION_TO_INDEX = {
        try:
            file_path = output.split("(")[0]
            self.current_frame_file = file_path
        except:
            pass
