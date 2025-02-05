import io
import sys

import pytest
from test_part_1 import _capture_output


def test_create_file_cd_list(terminal):
    """Test creating files, changing directories, and listing contents."""
    terminal.mkdir(["dir1"])
    terminal.create_file(["file1"])
    terminal.cd(["dir1"])
    terminal.create_file(["file2"])
    terminal.cd([".."])

    output = _capture_output(terminal.list_dir)
    assert "[DIR] dir1" in output
    assert "[FILE] file1" in output

    terminal.cd(["dir1"])
    output = _capture_output(terminal.list_dir)
    assert "[FILE] file2" in output


def test_pwd(terminal):
    """Test the pwd command."""
    output = _capture_output(terminal.pwd)
    assert output.strip() == "/"

    terminal.mkdir(["dir1"])
    terminal.cd(["dir1"])
    output = _capture_output(terminal.pwd)
    assert output.strip() == "/dir1"


def test_ls(terminal):
    """Test that ls works the same as list."""
    terminal.create_file(["file1"])

    output_list = _capture_output(terminal.list_dir)
    output_ls = _capture_output(
        terminal.list_dir
    )  # ls and list should give the same result
    assert output_list == output_ls


def test_nested_directory_creation(terminal):
    """Test creating nested directories by manually creating the parent."""
    # Create first-level directory
    terminal.mkdir(["dir1"])
    terminal.mkdir(["dir1/dir2"])  # Now that 'dir1' exists, 'dir2' can be created
    assert "dir1" in terminal.file_system["/"]
    assert "dir2" in terminal.file_system["/"]["dir1"]

    # Navigate into the first-level directory
    terminal.cd(["dir1"])
    # Navigate into the second-level directory
    terminal.cd(["dir2"])

    # Check if current directory is updated correctly
    assert terminal.current_dir == "/dir1/dir2"


def test_mkdir_existing_file(terminal):
    """Test trying to create a directory where a file with the same name exists."""
    terminal.create_file(["file1"])
    with pytest.raises(KeyError):
        terminal.mkdir(
            ["file1"]
        )  # This should raise an error because a file with that name exists


def test_rmdir_nonexistent(terminal):
    """Test removing a directory that does not exist."""
    with pytest.raises(KeyError):
        terminal.rmdir(
            ["nonexistent_dir"]
        )  # This should raise an error because the directory does not exist


def test_mkdir_nested_nonexistent(terminal):
    """Test creating nested directories when the parent directory does not exist."""
    with pytest.raises(KeyError):
        terminal.mkdir(
            ["nonexistent_parent/dir1"]
        )  # This should raise an error because 'nonexistent_parent' does not exist


def test_deeply_nested_directory_creation(terminal):
    """Test creating deeply nested directories in one command."""
    with pytest.raises(KeyError):
        terminal.mkdir(
            ["dir1/dir2/dir3"]
        )  # This should raise a KeyError because 'dir1' and 'dir2' do not exist


def test_create_directory_inside_file(terminal):
    """Test attempting to create a directory inside a file."""
    terminal.create_file(["file1"])  # Create a file
    with pytest.raises(KeyError):
        terminal.mkdir(
            ["file1/dir1"]
        )  # This should raise a KeyError because 'file1' is a file, not a directory


def test_overwrite_directory_with_file(terminal):
    """Test attempting to create a file with the same name as an existing directory."""
    terminal.mkdir(["dir1"])  # Create a directory
    with pytest.raises(KeyError):
        terminal.create_file(
            ["dir1"]
        )  # This should raise a KeyError because 'dir1' is a directory


def test_create_directory_with_invalid_path(terminal):
    """Test attempting to create a directory with an invalid path."""
    terminal.create_file(["file1"])  # Create a file
    with pytest.raises(KeyError):
        terminal.mkdir(
            ["file1/dir1/dir2"]
        )  # This should raise a KeyError because 'file1' is a file


def test_cd_into_file(terminal):
    """Test attempting to change directory into a file."""
    terminal.create_file(["file1"])  # Create a file
    with pytest.raises(KeyError):
        terminal.cd(
            ["file1"]
        )  # This should raise a KeyError because 'file1' is a file, not a directory


def test_remove_non_empty_directory(terminal):
    """Test attempting to remove a non-empty directory."""
    terminal.mkdir(["dir1"])  # Create a directory
    terminal.create_file(["dir1/file1"])  # Create a file inside the directory
    with pytest.raises(KeyError):
        terminal.rmdir(
            ["dir1"]
        )  # This should raise a KeyError because 'dir1' is not empty


def test_cd_non_existent_parent(terminal):
    """Test navigating to a non-existent parent directory."""
    with pytest.raises(KeyError):
        terminal.cd(
            ["dir1/dir2"]
        )  # This should raise a KeyError because 'dir1' and 'dir2' do not exist


def test_list_non_existent_directory(terminal):
    """Test listing the contents of a non-existent directory."""
    with pytest.raises(KeyError):
        terminal.cd(["dir1"])  # Try to navigate to a non-existent directory
        terminal.list_dir()  # This should raise a KeyError


def test_remove_directory_with_subdirectories(terminal):
    """Test attempting to remove a directory that contains subdirectories."""
    terminal.mkdir(["dir1"])  # Create a directory
    terminal.mkdir(["dir1/dir2"])  # Create a subdirectory
    with pytest.raises(KeyError):
        terminal.rmdir(
            ["dir1"]
        )  # This should raise a KeyError because 'dir1' is not empty


def test_cd_parent_from_root(terminal):
    """Test navigating to the parent directory from the root."""
    terminal.cd([".."])  # Navigate up from root
    assert terminal.current_dir == "/"  # Current directory should still be root
