from pathlib import Path

import pytest

from debug_gym.gym.terminals.docker import DockerTerminal
from debug_gym.gym.workspace import Workspace, WorkspaceReadError, WorkspaceWriteError
from tests.helpers import docker_test_terminal


@pytest.fixture
def workspace():
    terminal = docker_test_terminal()
    workspace = Workspace(terminal)
    workspace.reset()
    terminal.run(
        "mkdir -p subdir && touch .hidden file1.txt file2.txt subdir/subfile1.txt"
    )
    yield workspace
    terminal.close()


def test_directory_tree(workspace):
    result = workspace.directory_tree(max_depth=2)
    assert result == (
        f"{workspace.working_dir}/\n"
        "|-- .hidden\n"
        "|-- file1.txt\n"
        "|-- file2.txt\n"
        "|-- subdir/\n"
        "  |-- subfile1.txt"
    )


@pytest.if_docker_running
def test_reset_and_cleanup_workspace():
    terminal = DockerTerminal(base_image="ubuntu:latest")
    workspace = Workspace(terminal)

    assert workspace.working_dir is None

    workspace.reset()
    assert isinstance(workspace.working_dir, Path)
    assert str(workspace.working_dir) == "/testbed"

    workspace.cleanup()
    assert workspace.working_dir is None
    terminal.close()


def test_resolve_path(workspace):
    abs_path = (workspace.working_dir / "file.txt").resolve()
    workspace.write_file("file.txt", "")

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
    # Do not raise for working directory
    assert (
        workspace.resolve_path(workspace.working_dir, raises=True)
        == workspace.working_dir
    )


def test_read_file_reads_existing_file(workspace):
    file_content = "Hello, DebugGym!\n"
    workspace.write_file("test.txt", file_content)
    # Read file using relative path
    result = workspace.read_file(str(workspace.working_dir / "test.txt"))
    assert result == file_content
    # Read file using just the filename (should also work)
    result = workspace.read_file("test.txt")
    assert result == file_content


def test_read_file_raises_for_nonexistent_file(workspace):
    workspace.write_file("test.txt", "")
    # relative path that does not exist
    with pytest.raises(WorkspaceReadError):
        workspace.read_file("does_not_exist.txt")
    # absolute path matching a file in the working_dir
    with pytest.raises(WorkspaceReadError):
        workspace.read_file("/test.txt")


def test_write_file_basic(workspace):
    file_content = "Hello, DebugGym!\n\n\n"
    workspace.write_file("test.txt", file_content)
    assert workspace.read_file("test.txt") == file_content


def test_write_file_single_line_no_newline(workspace):
    file_content_single_line = "Hello, DebugGym!"
    workspace.write_file("test.txt", file_content_single_line)
    assert workspace.read_file("test.txt") == file_content_single_line


def test_write_file_with_delimiter(workspace):
    file_content_single_line = "Hello, DebugGym!nDEBUGGYM_DEL"
    workspace.write_file("test.txt", file_content_single_line)
    assert workspace.read_file("test.txt") == file_content_single_line


def test_write_file_with_newlines(workspace):
    file_content_with_newlines = "Hello, DebugGym!\nThis is a test.\n"
    workspace.write_file("test.txt", file_content_with_newlines)
    assert workspace.read_file("test.txt") == file_content_with_newlines


def test_write_file_empty_content(workspace):
    file_content_empty = ""
    workspace.write_file("test.txt", file_content_empty)
    assert workspace.read_file("test.txt") == file_content_empty


def test_write_file_exceeding_max_command_length(workspace):
    file_content_exceeding_max_command_length = "A" * (2 * 1024**2)  # 2MB of 'A's
    workspace.write_file("test.txt", file_content_exceeding_max_command_length)
    assert workspace.read_file("test.txt") == file_content_exceeding_max_command_length


def test_write_file_does_not_execute_content(workspace):
    side_effect = workspace.working_dir / "side-effect"
    content = f"payload\nDEBUGGYM_EOF\n); touch {side_effect}\n#"

    workspace.write_file("payload.txt", content)

    assert workspace.read_file("payload.txt") == content
    success, _ = workspace.terminal.run(f"test ! -e {side_effect}")
    assert success


def test_write_file_path_outside_workspace_relative(workspace):
    """Ensure path traversal attacks are blocked."""
    with pytest.raises(WorkspaceWriteError) as exc_info:
        workspace.write_file("../outside.txt", "should not be created")
    assert "outside the workspace" in str(exc_info.value)


def test_write_file_path_outside_workspace_absolute(workspace):
    """Ensure absolute paths outside workspace are blocked."""
    with pytest.raises(WorkspaceWriteError) as exc_info:
        workspace.write_file("/tmp/outside.txt", "should not be created")
    assert "outside the workspace" in str(exc_info.value)
