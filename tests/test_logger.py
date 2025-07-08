import logging
from unittest.mock import MagicMock, patch

import pytest
from rich.markup import escape

from debug_gym.logger import (
    DebugGymLogger,
    StatusColumn,
    StripAnsiFormatter,
    TaskProgress,
    TaskProgressManager,
    log_with_color,
)


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


def test_status_column_render():
    # Test the StatusColumn renders correctly for different task states
    column = StatusColumn()

    # Mock tasks in different states
    running_task = MagicMock()
    running_task.finished = False

    failed_task = MagicMock()
    failed_task.finished = True
    failed_task.fields = {"status": "failed"}

    completed_task = MagicMock()
    completed_task.finished = True
    completed_task.fields = {"status": "done"}

    # Test rendering for failed task
    failed_result = column.render(failed_task)
    assert failed_result.plain == "✗"
    assert "red" in failed_result.style

    # Test rendering for completed task
    completed_result = column.render(completed_task)
    assert completed_result.plain == "✓"
    assert "green" in completed_result.style


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


def test_task_progress_manager_initialization():
    # Test that TaskProgressManager initializes correctly
    problems = ["problem1", "problem2"]
    manager = TaskProgressManager(problems)

    # Check that tasks were added for each problem
    assert len(manager._tasks) == 2
    assert "problem1" in manager._tasks
    assert "problem2" in manager._tasks

    # Check that progress tasks were created
    assert len(manager._progress_task_ids) == 2
    assert "problem1" in manager._progress_task_ids
    assert "problem2" in manager._progress_task_ids


def test_task_progress_manager_advance():
    # Test that TaskProgressManager.advance updates task state correctly
    problems = ["problem1"]
    manager = TaskProgressManager(problems)

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


def test_task_progress_manager_get_task_stats():
    # Test that TaskProgressManager.get_task_stats returns correct stats
    problems = ["p1", "p2", "p3", "p4"]
    manager = TaskProgressManager(problems, max_display=5)

    # Set up tasks with different statuses
    updates = [
        TaskProgress("p1", 10, 10, 100, 100, "done"),
        TaskProgress("p2", 5, 10, 50, 100, "failed"),
        TaskProgress("p3", 3, 10, 30, 100, "running"),
        TaskProgress("p4", 0, 10, 0, 100, "pending"),
    ]

    for update in updates:
        manager.advance(update)

    # Get stats and check they're correct
    stats = manager.get_task_stats()
    assert stats["total"] == 4
    assert stats["pending"] == 1
    assert stats["running"] == 1
    assert stats["completed"] == 1
    assert stats["failed"] == 1


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


def test_debuggymlogger_log_queue_worker():
    DebugGymLogger.set_as_worker()
    logger = DebugGymLogger("test_worker_logger")
    # Clear the queue before test
    while not DebugGymLogger.LOG_QUEUE.empty():
        DebugGymLogger.LOG_QUEUE.get_nowait()
    logger.info("Worker log message")
    # Should be in LOG_QUEUE
    record = DebugGymLogger.LOG_QUEUE.get(timeout=1)
    assert record.msg == "Worker log message"


def test_debuggymlogger_report_progress():
    logger = DebugGymLogger("test_progress_logger")
    # Clear the queue before test
    while not DebugGymLogger.PROGRESS_QUEUE.empty():
        DebugGymLogger.PROGRESS_QUEUE.get_nowait()
    logger.report_progress(
        problem_id="prob1",
        step=1,
        total_steps=10,
        score=5,
        max_score=10,
        status="running",
    )
    progress = DebugGymLogger.PROGRESS_QUEUE.get(timeout=1)
    assert progress.problem_id == "prob1"
    assert progress.step == 1
    assert progress.status == "running"


def test_debuggymlogger_set_as_worker_resets():
    logger = DebugGymLogger("test_reset_logger")
    assert not DebugGymLogger._is_worker
    assert not logger._is_worker
    # Set as worker
    DebugGymLogger.set_as_worker()
    assert DebugGymLogger._is_worker
    assert logger._is_worker
    another_logger = DebugGymLogger("test_reset_logger")
    assert another_logger._is_worker


def test_debuggymlogger_rich_progress_raises_in_worker():
    DebugGymLogger.set_as_worker()
    logger = DebugGymLogger("test_rich_progress_logger")
    with pytest.raises(RuntimeError):
        with logger.rich_progress(["p1", "p2"]):
            pass
    DebugGymLogger._is_worker = False
