import json
import logging
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.markup import escape

from debug_gym.logger import (
    DebugGymLogger,
    StatusColumn,
    StripAnsiFormatter,
    TaskProgress,
    TaskProgressManager,
    log_file_path,
    log_with_color,
)


@pytest.fixture
def DebugGymLoggerTest():
    """Create a new DebugGymLogger class for each test to avoid
    interference between tests when setting as worker."""

    class TestDebugGymLogger(DebugGymLogger):
        LOG_QUEUE = mp.Queue(maxsize=10000)
        PROGRESS_QUEUE = mp.Queue(maxsize=10000)
        _is_worker = False

    yield TestDebugGymLogger


def test_task_progress_pending_status():
    # Test a task in 'pending' status
    task_pending = TaskProgress(
        problem_id="test_pending",
        step=0,
        total_steps=10,
        score=0,
        max_score=100,
        status="pending",
    )
    assert task_pending.completed is False


@pytest.mark.parametrize(
    "status,expected_marker",
    [
        ("resolved", "✓"),
        ("unresolved", "✗"),
        ("skip-resolved", "✓"),
        ("skip-unresolved", "✗"),
        ("error", "!"),
        ("running", "⠋"),
        ("pending", "⠋"),
    ],
)
def test_taskprogress_marker_valid(status, expected_marker):
    assert TaskProgress.marker(status) == expected_marker


def test_taskprogress_marker_invalid():
    with pytest.raises(ValueError):
        TaskProgress.marker("not-a-status")


@pytest.mark.parametrize(
    "status,expected_color",
    [
        ("resolved", "green"),
        ("unresolved", "red"),
        ("skip-resolved", "yellow"),
        ("skip-unresolved", "yellow"),
        ("error", "red"),
        ("running", "blue"),
        ("pending", "yellow"),
    ],
)
def test_taskprogress_color_valid(status, expected_color):
    assert TaskProgress.color(status) == expected_color


def test_taskprogress_color_invalid():
    with pytest.raises(ValueError):
        TaskProgress.color("not-a-status")


def test_status_column_render():
    # Test the StatusColumn renders correctly for different task states
    column = StatusColumn()

    # Mock tasks in different states
    running_task = MagicMock()
    running_task.finished = False

    completed_task = MagicMock()
    completed_task.finished = True
    completed_task.fields = {"status": "resolved"}
    completed_result = column.render(completed_task)
    assert completed_result.plain == "✓"
    assert "green" in completed_result.style

    failed_task = MagicMock()
    failed_task.finished = True
    failed_task.fields = {"status": "unresolved"}
    failed_result = column.render(failed_task)
    assert failed_result.plain == "✗"
    assert "red" in failed_result.style

    skip_resolved_task = MagicMock()
    skip_resolved_task.finished = True
    skip_resolved_task.fields = {"status": "skip-resolved"}
    skip_resolved_result = column.render(skip_resolved_task)
    assert skip_resolved_result.plain == "✓"
    assert "yellow" in skip_resolved_result.style

    skip_unresolved_task = MagicMock()
    skip_unresolved_task.finished = True
    skip_unresolved_task.fields = {"status": "skip-unresolved"}
    skip_unresolved_result = column.render(skip_unresolved_task)
    assert skip_unresolved_result.plain == "✗"
    assert "yellow" in skip_unresolved_result.style

    error_task = MagicMock()
    error_task.finished = True
    error_task.fields = {"status": "error"}
    error_result = column.render(error_task)
    assert error_result.plain == "!"
    assert "red" in error_result.style

    unknown_task = MagicMock()
    unknown_task.finished = True
    unknown_task.fields = {"status": "unknown-status"}
    with pytest.raises(ValueError):
        column.render(unknown_task)


def test_strip_ansi_formatter():
    # Test that the StripAnsiFormatter removes ANSI color codes
    formatter = StripAnsiFormatter("%(message)s")

    # Create a log record with ANSI color
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="\033[31mRed text\033[0m",
        args=(),
        exc_info=None,
    )

    result = formatter.format(record)
    assert result == "Red text"


def test_task_progress_manager_initialization(DebugGymLoggerTest):
    # Test that TaskProgressManager initializes correctly
    logger = DebugGymLoggerTest("test_logger")
    problems = ["problem1", "problem2"]
    manager = TaskProgressManager(problems, logger=logger)

    # Check that tasks were added for each problem
    assert len(manager._tasks) == 2
    assert "problem1" in manager._tasks
    assert "problem2" in manager._tasks

    # Check that progress tasks were created
    assert len(manager._progress_task_ids) == 2
    assert "problem1" in manager._progress_task_ids
    assert "problem2" in manager._progress_task_ids


def test_task_progress_manager_advance(DebugGymLoggerTest):
    # Test that TaskProgressManager.advance updates task state correctly
    logger = DebugGymLoggerTest("test_logger")
    problems = ["problem1"]
    manager = TaskProgressManager(problems, logger=logger)

    # Initial state
    assert manager._tasks["problem1"].step == 0
    assert manager._tasks["problem1"].status == "pending"

    # Create a progress update
    update = TaskProgress(
        problem_id="problem1",
        step=5,
        total_steps=10,
        score=50,
        max_score=100,
        status="running",
    )

    # Apply the update
    with patch.object(manager.progress, "update") as mock_update:
        manager.advance(update)

        # Check that the task was updated
        assert manager._tasks["problem1"].step == 5
        assert manager._tasks["problem1"].status == "running"
        assert manager._tasks["problem1"].score == 50

        # Check that the progress was updated
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        assert kwargs["completed"] == 5
        assert kwargs["status"] == "running"


def test_group_tasks_by_status_basic(DebugGymLoggerTest):
    # Test that group_tasks_by_status groups tasks correctly by their status
    logger = DebugGymLoggerTest("test_logger")
    problems = ["p1", "p2", "p3", "p4"]
    manager = TaskProgressManager(problems, logger=logger)
    # Set up tasks with different statuses
    updates = [
        TaskProgress("p1", 1, 10, 10, 100, "resolved"),
        TaskProgress("p2", 2, 10, 20, 100, "running"),
        TaskProgress("p3", 0, 10, 0, 100, "pending"),
        TaskProgress("p4", 0, 10, 0, 100, "error"),
    ]
    for update in updates:
        manager.advance(update)
    grouped = manager.group_tasks_by_status()
    assert grouped["resolved"] == ["p1"]
    assert grouped["running"] == ["p2"]
    assert grouped["pending"] == ["p3"]
    assert grouped["error"] == ["p4"]
    # All other statuses should be empty lists
    for status in TaskProgress.statuses():
        if status not in ["resolved", "running", "pending", "error"]:
            assert grouped[status] == []


def test_group_tasks_by_status_with_unknown_status(DebugGymLoggerTest):
    # Test that a task with an unknown status is grouped under "pending"
    logger = DebugGymLoggerTest("test_logger")
    problems = ["p1"]
    manager = TaskProgressManager(problems, logger=logger)
    # Manually set an unknown status
    manager._tasks["p1"].status = "not-a-status"
    grouped = manager.group_tasks_by_status()
    # Should be grouped under "pending"
    assert grouped["pending"] == ["p1"]
    # All other statuses should be empty
    for status in TaskProgress.statuses():
        if status != "pending":
            assert grouped[status] == []


def test_group_tasks_by_status_multiple_tasks_same_status():
    # Test that multiple tasks with the same status are grouped together
    problems = ["p1", "p2", "p3"]
    manager = TaskProgressManager(problems)
    updates = [
        TaskProgress("p1", 1, 10, 10, 100, "running"),
        TaskProgress("p2", 2, 10, 20, 100, "running"),
        TaskProgress("p3", 0, 10, 0, 100, "pending"),
    ]
    for update in updates:
        manager.advance(update)
    grouped = manager.group_tasks_by_status()
    assert set(grouped["running"]) == {"p1", "p2"}
    assert grouped["pending"] == ["p3"]
    # All other statuses should be empty
    for status in TaskProgress.statuses():
        if status not in ["running", "pending"]:
            assert grouped[status] == []


def test_task_progress_manager_get_task_stats(DebugGymLoggerTest):
    # Test that TaskProgressManager.get_task_stats returns correct stats
    logger = DebugGymLoggerTest("test_logger")
    problems = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]
    manager = TaskProgressManager(problems, max_display=5, logger=logger)

    # Set up tasks with different statuses
    updates = [
        TaskProgress("p1", 10, 10, 100, 100, "resolved"),
        TaskProgress("p2", 5, 10, 50, 100, "unresolved"),
        TaskProgress("p3", 3, 10, 30, 100, "running"),
        TaskProgress("p4", 3, 10, 30, 100, "running"),
        TaskProgress("p5", 0, 10, 0, 100, "pending"),
        TaskProgress("p6", 0, 10, 0, 100, "skip-resolved"),
        TaskProgress("p7", 0, 10, 0, 100, "skip-unresolved"),
        TaskProgress("p8", 0, 10, 0, 100, "error"),
    ]

    for update in updates:
        manager.advance(update)

    # Get stats and check they're correct
    stats = manager.get_task_stats()
    assert stats["total"] == 8
    assert stats["pending"] == 1
    assert stats["running"] == 2
    assert stats["resolved"] == 1
    assert stats["unresolved"] == 1
    assert stats["skip-resolved"] == 1
    assert stats["skip-unresolved"] == 1
    assert stats["error"] == 1


@pytest.mark.parametrize(
    "all_tasks,expected_title",
    [(True, "Tasks:"), (False, "In progress (max-display 2):")],
)
def test_task_progress_manager_panel_title(all_tasks, expected_title):
    # Test that the panel title is set correctly based on all_tasks flag
    problems = ["p1", "p2", "p3", "p4"]  # 4 problems
    manager = TaskProgressManager(problems, max_display=2)  # but only show 2

    title = manager._get_tasks_panel_title(all_tasks)
    assert title == expected_title


def test_log_with_color_calls_logger_info_with_escaped_message():
    mock_logger = MagicMock(spec=DebugGymLogger)
    message = "Hello <world>"
    color = "blue"

    log_with_color(mock_logger, message, color)

    # The message should be escaped and wrapped in color tags
    expected_msg = f"[{color}]{escape(message)}[/{color}]"
    mock_logger.info.assert_called_once()
    args, kwargs = mock_logger.info.call_args
    assert args[0] == expected_msg
    assert kwargs["extra"] == {"already_escaped": True}


def test_log_with_color_handles_different_colors():
    mock_logger = MagicMock(spec=DebugGymLogger)
    message = "Test message"
    for color in ["red", "green", "yellow"]:
        log_with_color(mock_logger, message, color)
        expected_msg = f"[{color}]{escape(message)}[/{color}]"
        mock_logger.info.assert_called_with(
            expected_msg, extra={"already_escaped": True}
        )
        mock_logger.info.reset_mock()


def test_log_with_color_escapes_special_characters():
    mock_logger = MagicMock(spec=DebugGymLogger)
    message = "Special [chars] <here>"
    color = "magenta"
    log_with_color(mock_logger, message, color)
    expected_msg = f"[{color}]{escape(message)}[/{color}]"
    mock_logger.info.assert_called_once_with(
        expected_msg, extra={"already_escaped": True}
    )


def test_debuggymlogger_log_queue_worker(DebugGymLoggerTest):
    DebugGymLoggerTest.set_as_worker()
    logger = DebugGymLoggerTest("test_worker_logger")
    # Clear the queue before test
    while not DebugGymLoggerTest.LOG_QUEUE.empty():
        DebugGymLoggerTest.LOG_QUEUE.get_nowait()
    logger.info("Worker log message")
    # Should be in LOG_QUEUE
    record = DebugGymLoggerTest.LOG_QUEUE.get(timeout=1)
    assert record.msg == "Worker log message"


def test_debuggymlogger_report_progress(DebugGymLoggerTest):
    logger = DebugGymLoggerTest("test_progress_logger")
    # Clear the queue before test
    while not DebugGymLoggerTest.PROGRESS_QUEUE.empty():
        DebugGymLoggerTest.PROGRESS_QUEUE.get_nowait()
    logger.report_progress(
        problem_id="prob1",
        step=1,
        total_steps=10,
        score=5,
        max_score=10,
        status="running",
    )
    progress = DebugGymLoggerTest.PROGRESS_QUEUE.get(timeout=1)
    assert progress.problem_id == "prob1"
    assert progress.step == 1
    assert progress.status == "running"


def test_debuggymlogger_set_as_worker_resets(DebugGymLoggerTest):
    logger = DebugGymLoggerTest("test_reset_logger")
    assert DebugGymLoggerTest.is_main
    assert logger.is_main
    # Set as worker
    DebugGymLoggerTest.set_as_worker()
    assert DebugGymLoggerTest.is_worker
    assert logger.is_worker
    another_logger = DebugGymLoggerTest("test_reset_logger")
    assert another_logger.is_worker


def test_debuggymlogger_rich_progress_raises_in_worker(DebugGymLoggerTest):
    DebugGymLoggerTest.set_as_worker()
    logger = DebugGymLoggerTest("test_rich_progress_logger")
    with pytest.raises(RuntimeError):
        with logger.rich_progress(["p1", "p2"]):
            pass


def test_log_file_path_absolute(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    problem_id = "prob1"
    result = log_file_path(log_dir, problem_id)
    assert result == (log_dir / "prob1.log").absolute()
    assert result.is_absolute()


def test_log_file_path_relative(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    problem_id = "prob2"
    # Change cwd to tmp_path for relative path calculation
    monkeypatch.chdir(tmp_path)
    result = log_file_path(log_dir, problem_id, relative=True)
    assert result == Path("logs") / "prob2.log"


def test_log_file_path_relative_outside_cwd(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    problem_id = "prob5"
    result = log_file_path(log_dir, problem_id, relative=True)
    assert result == log_dir / "prob5.log"


def test_dump_task_status_creates_json_file(tmp_path, DebugGymLoggerTest):
    logdir = tmp_path / "logdir"
    logdir.mkdir()
    logger = DebugGymLoggerTest("test_logger")
    problems = ["problem1", "problem2"]
    manager = TaskProgressManager(problems, logger=logger)

    task = TaskProgress(
        problem_id="problem1",
        step=3,
        total_steps=10,
        score=7,
        max_score=10,
        status="unresolved",
        logdir=str(logdir),
    )
    # Should create status.json in logdir
    manager.dump_task_status(task)
    status_path = logdir / "status.json"
    assert status_path.exists()
    # Check contents
    with open(status_path, "r") as f:
        data = json.load(f)
    assert data["problem_id"] == "problem1"
    assert data["step"] == 3
    assert data["status"] == "unresolved"
    assert data["logdir"] == str(logdir)

    assert data == asdict(task)
