class BaseSimulator:
    def __init__(self):
        self.file_system = {"/": {}}  # Root directory
        self.current_dir = "/"

    def run(self):
        while True:
            command = input(f"{self.current_dir}$ ").strip().split()
            if not command:
                continue
            cmd = command[0]
            args = command[1:]
            if cmd == "mkdir":
                self.mkdir(args)
            elif cmd == "rmdir":
                self.rmdir(args)
            elif cmd == "cd":
                self.cd(args)
            elif cmd == "list" or cmd == "ls":
                self.list_dir()
            elif cmd == "create_file":
                self.create_file(args)
            elif cmd == "pwd":
                self.pwd()
            elif cmd == "help":
                self.help()
            elif cmd == "exit":
                break
            else:
                print(f"Unknown command: {cmd}")

    def mkdir(self, args):
        if len(args) != 1:
            print("Usage: mkdir <dirname>")
            return
        path = args[0].strip("/")
        dirs = path.split("/")
        current_fs = self._resolve_path(self.current_dir)

        for i, dirname in enumerate(dirs):
            if dirname in current_fs:
                if isinstance(current_fs[dirname], dict):
                    current_fs = current_fs[dirname]
                else:
                    raise KeyError(f"{dirname} exists and is a file.")
            else:
                if i == len(dirs) - 1:  # Last directory to create
                    current_fs[dirname] = {}
                else:
                    raise KeyError(
                        f"Parent directory does not exist: {'/'.join(dirs[:i+1])}"
                    )
                current_fs = current_fs[dirname]

    def rmdir(self, args):
        if len(args) != 1:
            print("Usage: rmdir <dirname>")
            return
        dirname = args[0].strip("/")
        current_fs = self._resolve_path(self.current_dir)

        if dirname in current_fs:
            if isinstance(current_fs[dirname], dict):
                if current_fs[dirname]:  # Check if directory is non-empty
                    raise KeyError(f"Directory {dirname} is not empty.")
                else:  # Empty directory
                    del current_fs[dirname]
            else:
                raise KeyError(f"{dirname} is not a directory.")
        else:
            raise KeyError(f"No such directory: {dirname}")

    def create_file(self, args):
        raise NotImplementedError("create_file method is not implemented.")

    def _resolve_path(self, path):
        raise NotImplementedError("_resolve_path method is not implemented.")

    def cd(self, args):
        raise NotImplementedError("cd method is not implemented.")

    def list_dir(self):
        raise NotImplementedError("list_dir method is not implemented.")

    def pwd(self):
        """Print the current working directory."""
        print(self.current_dir)

    def help(self):
        """Display help information for all commands."""
        help_text = """
Available commands:
mkdir <dirname>   : Create a directory with the name <dirname>.
rmdir <dirname>   : Remove an empty directory with the name <dirname>.
cd <dirname>      : Change to directory <dirname>. Use '..' to go up one level and '/' to go to the root.
list              : List the contents of the current directory.
ls                : Same as 'list', lists contents of the current directory.
create_file <filename>  : Create an empty file with the name <filename>.
pwd               : Print the current working directory.
help              : Show this help message.
exit              : Exit the terminal.
        """
        print(help_text)
