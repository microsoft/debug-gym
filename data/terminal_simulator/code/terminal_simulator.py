from code import base_simulator


class TerminalSimulator(base_simulator.BaseSimulator):
    def __init__(self):
        super().__init__()

    def create_file(self, args):
        if len(args) != 1:
            print("Usage: create_file <filename>")
            return
        filename = args[0]
        current_fs = self._resolve_path(self.current_dir)

        parts = filename.split("/")
        subdirs, filename = parts[:-1], parts[-1]
        for subdir in subdirs:
            if subdir not in current_fs:
                raise KeyError(f"intermediate directory {subdir} does not exist")
            current_fs = current_fs[subdir]

        if filename in current_fs:
            raise KeyError(f"File {filename} already exists.")
        else:
            current_fs[filename] = None  # None represents a file

    def _resolve_path(self, path):
        """Helper function to resolve a path in the file system."""
        dirs = path.strip("/").split("/")
        current_fs = self.file_system["/"]
        for d in dirs:
            if d:
                if d in current_fs and isinstance(current_fs[d], dict):
                    current_fs = current_fs[d]
                else:
                    raise KeyError(f"Directory {d} does not exist.")
        return current_fs

    def cd(self, args):
        if len(args) != 1:
            print("Usage: cd <dirname>")
            return
        path = args[0].strip("/")

        if path == "":
            self.current_dir = "/"
            return

        dirs = path.split("/")
        current_fs = self._resolve_path(self.current_dir)
        for dirname in dirs:
            if dirname == "..":
                self.current_dir = (
                    "/".join(self.current_dir.rstrip("/").split("/")[:-1]) or "/"
                )
                current_fs = self._resolve_path(self.current_dir)
            elif dirname == "/":
                self.current_dir = "/"
                current_fs = self.file_system["/"]
            elif dirname in current_fs:
                if isinstance(current_fs[dirname], dict):
                    if self.current_dir == "/":
                        self.current_dir = f"/{dirname}"
                    else:
                        self.current_dir = f"{self.current_dir}/{dirname}"
                    current_fs = current_fs[dirname]
                else:
                    raise KeyError(f"{dirname} is not a directory.")
            else:
                raise KeyError(f"No such directory: {dirname}")

    def list_dir(self):
        try:
            current_fs = self._resolve_path(self.current_dir)
        except KeyError as e:
            raise KeyError(f"No such directory: {self.current_dir}")
        items = current_fs.keys()
        if items:
            for item in items:
                if isinstance(current_fs[item], dict):
                    print(f"[DIR] {item}")
                else:
                    print(f"[FILE] {item}")
        else:
            print("Directory is empty.")
