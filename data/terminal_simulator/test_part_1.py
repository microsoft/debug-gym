import io
import sys

import pytest


def _capture_output(func, *args):
    """Utility function to capture printed output."""
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    try:
        func(*args)
    finally:
        sys.stdout = old_stdout
    return new_stdout.getvalue()


def test_mkdir(terminal):
    """Test creating a directory."""
    terminal.mkdir(["dir1"])
    assert "dir1" in terminal.file_system["/"]
    assert isinstance(terminal.file_system["/"]["dir1"], dict)


def test_rmdir(terminal):
    """Test removing an empty directory."""
    terminal.mkdir(["dir1"])
    terminal.rmdir(["dir1"])
    assert "dir1" not in terminal.file_system["/"]


def test_rmdir_non_empty(terminal):
    """Test removing a non-empty directory."""
    terminal.mkdir(["dir1"])
    terminal.cd(["dir1"])
    terminal.create_file(["file1"])
    terminal.cd([".."])
    with pytest.raises(KeyError):
        terminal.rmdir(["dir1"])


def test_cd(terminal):
    """Test changing directory."""
    terminal.mkdir(["dir1"])
    terminal.cd(["dir1"])
    assert terminal.current_dir == "/dir1"


def test_cd_root(terminal):
    """Test changing to the root directory."""
    terminal.mkdir(["dir1"])
    terminal.cd(["dir1"])
    terminal.cd(["/"])
    assert terminal.current_dir == "/"


def test_cd_parent(terminal):
    """Test navigating to the parent directory."""
    terminal.mkdir(["dir1"])
    terminal.cd(["dir1"])
    terminal.cd([".."])
    assert terminal.current_dir == "/"


def test_list_empty(terminal):
    """Test listing an empty directory."""
    output = _capture_output(terminal.list_dir)
    assert output.strip() == "Directory is empty."


def test_list_non_empty(terminal):
    """Test listing a directory with files and directories."""
    terminal.mkdir(["dir1"])
    terminal.create_file(["file1"])
    output = _capture_output(terminal.list_dir)
    assert "[DIR] dir1" in output
    assert "[FILE] file1" in output


def test_create_file(terminal):
    """Test creating a file."""
    terminal.create_file(["file1"])
    assert "file1" in terminal.file_system["/"]
    assert terminal.file_system["/"]["file1"] is None


def test_mkdir_rmdir(terminal):
    """Test creating and removing a directory."""
    terminal.mkdir(["dir1"])
    assert "dir1" in terminal.file_system["/"]
    terminal.rmdir(["dir1"])
    assert "dir1" not in terminal.file_system["/"]
