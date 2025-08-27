import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from debug_gym.gym.entities import EvalOutput, Event, Observation
from debug_gym.gym.envs.env import EnvInfo, EventHooks, RepoEnv, TooledEnv
from debug_gym.gym.terminal import DockerTerminal, Terminal
from debug_gym.gym.tools.tool import ToolCall
from debug_gym.gym.tools.toolbox import Toolbox
from debug_gym.gym.workspace import RemoteWorkspace


@pytest.fixture
def workspace():
    terminal = Terminal()
    workspace = RemoteWorkspace(terminal)
    workspace.reset()

    repo_path = workspace.working_dir
    subdir_path = repo_path / "subdir"
    subdir_path.mkdir()
    (repo_path / "file1.txt").touch()
    (repo_path / "file2.txt").touch()
    (subdir_path / "subfile1.txt").touch()

    return workspace


def test_reset_and_cleanup_workspace():
    # Setup workspace with a native terminal.
    terminal = Terminal()
    workspace = RemoteWorkspace(terminal)

    assert workspace._tempdir is None
    assert workspace.working_dir is None

    workspace.reset()
    assert workspace._tempdir is not None
    assert isinstance(workspace.working_dir, Path)
    assert str(workspace.working_dir) == workspace._tempdir.name
    assert str(os.path.basename(workspace.working_dir)).startswith("DebugGym-")
    assert os.path.isdir(workspace.working_dir)

    working_dir = str(workspace.working_dir)
    workspace.cleanup()
    assert workspace._tempdir is None
    assert workspace.working_dir is None
    assert not os.path.isdir(working_dir)

    # Setup workspace with a remote terminal.
    terminal = DockerTerminal()
    workspace = RemoteWorkspace(terminal)

    assert workspace._tempdir is None
    assert workspace.working_dir is None

    workspace.reset()
    assert workspace._tempdir is None
    assert isinstance(workspace.working_dir, Path)
    assert str(workspace.working_dir) == "/testbed"
    # Nothing should be created on the host.
    assert not os.path.isdir(workspace.working_dir)

    workspace.cleanup()
    assert workspace.working_dir is None


def test_display_files(workspace):
    result = workspace.display_files(dir_tree_depth=2)
    assert result == (
        "Listing files in the current working directory. (read-only) indicates read-only files. Max depth: 2.\n"
        f"{workspace.working_dir}/\n"
        "|-- file1.txt\n"
        "|-- file2.txt\n"
        "|-- subdir/\n"
        "  |-- subfile1.txt"
    )


def test_display_files_read_only(workspace):
    read_only_path = workspace.working_dir / "read-only-file.txt"
    read_only_path.touch()

    debugreadonly = workspace.working_dir / ".debugreadonly"
    debugreadonly.write_text("read-only-file.txt")

    # Reset filters to take into account the .debugreadonly file.
    workspace.setup_file_filters()

    result = workspace.display_files(dir_tree_depth=2)
    assert result == (
        "Listing files in the current working directory. (read-only) indicates read-only files. Max depth: 2.\n"
        f"{workspace.working_dir}/\n"
        "|-- file1.txt\n"
        "|-- file2.txt\n"
        "|-- read-only-file.txt (read-only)\n"
        "|-- subdir/\n"
        "  |-- subfile1.txt"
    )


def test_display_files_ignore(workspace):
    debugignore = workspace.working_dir / ".debugignore"
    debugignore.write_text("file1.*\nsubdir/\n")

    # Reset filters to take into account the .debugignore file.
    workspace.setup_file_filters()

    result = workspace.display_files(dir_tree_depth=2)
    assert result == (
        "Listing files in the current working directory. (read-only) indicates read-only files. Max depth: 2.\n"
        f"{workspace.working_dir}/\n"
        "|-- file2.txt"
    )


@pytest.mark.parametrize("debugignore", ["", ".?*"])
def test_resolve_path(workspace, debugignore):

    # terminal = Terminal()
    # workspace = RemoteWorkspace(terminal)
    # workspace.reset()

    (workspace.working_dir / ".debugignore").write_text(debugignore)
    workspace.setup_file_filters()  # Reset filters.

    abs_path = (workspace.working_dir / "file.txt").resolve()
    (abs_path).touch()

    # env.working_dir itself
    path_from_env = workspace.resolve_path(str(workspace.working_dir), raises=True)
    assert path_from_env == workspace.working_dir.resolve()
    # relative path
    path_from_env = workspace.resolve_path("file.txt")
    assert path_from_env == abs_path
    # relative path with ./
    path_from_env = workspace.resolve_path("./file.txt")
    assert path_from_env == abs_path
    # absolute path
    path_from_env = workspace.resolve_path(str(abs_path))
    assert path_from_env == abs_path
    # relative path with Path object
    path_from_env = workspace.resolve_path(Path("file.txt"))
    assert path_from_env == abs_path
    # absolute path with Path object
    path_from_env = workspace.resolve_path(abs_path)
    assert path_from_env == abs_path
    # return an absolute path regardless of existence
    non_existent_path = workspace.resolve_path("non_existent_file.txt")
    assert (
        non_existent_path == (workspace.working_dir / "non_existent_file.txt").resolve()
    )
    # non-existent absolute path
    non_existent_path = workspace.resolve_path("/tmp/non_existent_file.txt").resolve()
    assert non_existent_path == Path("/tmp/non_existent_file.txt").resolve()


def test_resolve_path_raises(workspace):
    # Non-existent file with raises=True
    with pytest.raises(FileNotFoundError):
        workspace.resolve_path("non_existent_file.txt", raises=True)
    # Non-existent absolute path with raises=True
    with pytest.raises(FileNotFoundError):
        workspace.resolve_path("/tmp/non_existent_file.txt", raises=True)
    with pytest.raises(FileNotFoundError):
        workspace.resolve_path("..", raises=True)
    # Invalid path type
    with pytest.raises(TypeError):
        workspace.resolve_path(123, raises=True)
    with pytest.raises(TypeError):
        workspace.resolve_path(None, raises=True)


def test_resolve_path_do_not_raise_working_dir(workspace):
    # Do not raise for working directory even if the ignore patterns match
    (workspace.working_dir / ".debugignore").write_text(".*")
    workspace.setup_file_filters()  # Reset filters.
    assert (
        workspace.resolve_path(workspace.working_dir, raises=True)
        == workspace.working_dir
    )


def test_setup_file_filters_basic(workspace):
    # Setup a fake repo structure
    subdir = workspace.working_dir / "subdir"
    files = [
        workspace.working_dir / "file1.txt",
        workspace.working_dir / "file2.txt",
        workspace.working_dir / "ignored.txt",
        workspace.working_dir / "readonly.txt",
        workspace.working_dir / "subdir/subfile1.txt",
    ]
    [f.touch() for f in files]
    files.append(subdir)
    workspace.setup_file_filters()
    # All files should be indexed
    assert all(workspace.has_file(f) for f in files)
    # All files should be editable if no readonly patterns
    assert all(workspace.is_editable(f) for f in files)


def test_setup_file_filters_with_ignore_patterns(workspace):
    (workspace.working_dir / "file1.txt").touch()
    (workspace.working_dir / "file2.txt").touch()
    (workspace.working_dir / "ignoreme.txt").touch()
    (workspace.working_dir / "subdir" / "file3.txt").touch()

    # Ignore files matching "ignoreme.txt"
    workspace.setup_file_filters(ignore_patterns=["ignoreme.txt"])
    assert not workspace.has_file("ignoreme.txt")
    assert workspace.has_file("file1.txt")
    assert workspace.has_file("file2.txt")
    assert workspace.has_file("subdir/file3.txt")


def test_setup_file_filters_with_readonly_patterns(workspace):
    (workspace.working_dir / "file1.txt").touch()
    (workspace.working_dir / "readonly.txt").touch()

    # Mark "readonly.txt" as read-only
    workspace.setup_file_filters(readonly_patterns=["readonly.txt"])
    assert workspace.is_editable("file1.txt")
    assert not workspace.is_editable("readonly.txt")


def test_setup_file_filters_with_debugignore_and_debugreadonly(workspace):
    (workspace.working_dir / "file1.txt").touch()
    (workspace.working_dir / "file2.txt").touch()
    (workspace.working_dir / "ignoreme.txt").touch()
    (workspace.working_dir / "readonly.txt").touch()
    # Write .debugignore and .debugreadonly
    (workspace.working_dir / ".debugignore").write_text("ignoreme.txt\n")
    (workspace.working_dir / ".debugreadonly").write_text("readonly.txt\n")

    workspace.setup_file_filters()  # Reset filters.
    assert not workspace.has_file("ignoreme.txt")
    assert workspace.has_file("file1.txt")
    assert workspace.has_file("file2.txt")
    assert workspace.has_file("readonly.txt")
    # Check that readonly.txt is not editable
    assert not workspace.is_editable("readonly.txt")
    # Check that file1.txt and file2.txt are editable
    assert workspace.is_editable("file1.txt")
    assert workspace.is_editable("file2.txt")
    with pytest.raises(FileNotFoundError):
        workspace.is_editable("ignoreme.txt")


def test_setup_file_filters_combined_patterns(workspace):
    (workspace.working_dir / "file1.txt").touch()
    (workspace.working_dir / "file2.txt").touch()
    (workspace.working_dir / "ignoreme.txt").touch()
    (workspace.working_dir / "readonly.txt").touch()
    (workspace.working_dir / ".debugignore").write_text("ignoreme.txt\n")
    (workspace.working_dir / ".debugreadonly").write_text("readonly.txt\n")

    # Also ignore file2.txt and mark file1.txt as readonly via patterns
    workspace.setup_file_filters(
        ignore_patterns=["file2.txt"],
        readonly_patterns=["file1.txt"],
    )
    assert not workspace.has_file("ignoreme.txt")
    assert not workspace.has_file("file2.txt")
    assert workspace.has_file("file1.txt")
    assert workspace.has_file("readonly.txt")
    # Both file1.txt and readonly.txt should be readonly
    assert not workspace.is_editable("file1.txt")
    assert not workspace.is_editable("readonly.txt")
    with pytest.raises(FileNotFoundError):
        assert not workspace.is_editable("ignoreme.txt")
    with pytest.raises(FileNotFoundError):
        assert not workspace.is_editable("file2.txt")


def test_read_file_reads_existing_file(workspace):
    file_path = workspace.working_dir / "test.txt"
    file_content = "Hello, DebugGym!\n"
    file_path.write_text(file_content)
    # Read file using relative path
    result = workspace.read_file(str(workspace.working_dir / "test.txt"))
    assert result == file_content
    # Read file using just the filename (should also work)
    result = workspace.read_file("test.txt")
    assert result == file_content


def test_read_file_raises_for_nonexistent_file(workspace):
    (workspace.working_dir / "test.txt").touch()
    # relative path that does not exist
    with pytest.raises(FileNotFoundError):
        workspace.read_file("does_not_exist.txt")
    # absolute path matching a file in the working_dir
    with pytest.raises(FileNotFoundError):
        workspace.read_file("/test.txt")


def test_write_file(workspace):
    file_path = workspace.working_dir / "test.txt"
    file_content = "Hello, DebugGym!\n"
    workspace.write_file("test.txt", file_content)
    assert file_path.read_text() == file_content

    # Test with single line, no newline at the end.
    file_content_single_line = "Hello, DebugGym!"
    workspace.write_file("test.txt", file_content_single_line)
    assert file_path.read_text() == file_content_single_line

    # Test with newlines
    file_content_with_newlines = "Hello, DebugGym!\nThis is a test.\n"
    workspace.write_file("test.txt", file_content_with_newlines)
    assert file_path.read_text() == file_content_with_newlines
