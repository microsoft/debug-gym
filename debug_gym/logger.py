import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from rich.live import Live
from rich.logging import RichHandler
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, Task, TextColumn
from rich.table import Table
from rich.text import Text

from debug_gym.utils import strip_ansi


class StripAnsiFormatter(logging.Formatter):

    def format(self, record):
        msg = super().format(record)
        return strip_ansi(msg)


@dataclass(slots=True)  # Slitly faster / memory efficient when using slots
class TaskProgress:
    """Data class to communicate task progress information."""

    problem_id: str
    step: int
    total_steps: int  # Total steps for the problem considering early stopping
    score: int
    max_score: int
    status: str
    logfile: str = ""

    @property
    def completed(self) -> bool:
        """Check if the task is completed based on its status."""
        return self.status in (
            "resolved",
            "unresolved",
            "skip-resolved",
            "skip-unresolved",
            "error",
        )

    @classmethod
    def statuses(cls):
        """Return the valid statuses for tasks."""
        return (
            "running",
            "pending",
            "resolved",
            "unresolved",
            "skip-resolved",
            "skip-unresolved",
            "error",
        )

    @classmethod
    def marker(cls, status: str) -> str:
        """Return a marker for the task based on its status."""
        if status == "resolved":
            return "✓"
        elif status == "unresolved":
            return "✗"
        elif status == "skip-resolved":
            return "✓"
        elif status == "skip-unresolved":
            return "✗"
        elif status == "error":
            return "!"
        elif status == "running":
            return "⠋"
        elif status == "pending":
            return "⠋"
        raise ValueError(f"Unknown task status: `{status}`. ")

    @classmethod
    def color(cls, status: str) -> str:
        """Return the color for a given status."""
        if status == "resolved":
            return "green"
        elif status == "unresolved":
            return "red"
        elif status == "skip-resolved":
            return "yellow"
        elif status == "skip-unresolved":
            return "yellow"
        elif status == "error":
            return "red"
        elif status == "running":
            return "blue"
        elif status == "pending":
            return "yellow"
        raise ValueError(f"Unknown task status: `{status}`. ")


class StatusColumn(SpinnerColumn):
    """Custom status column. magenta ! when error,
    yellow spinner pending, blue spinner running,
    green ✓ when resolved, red ✗ when unresolved,
    yellow ✓ when skip-resolved, yellow ✗ when skip-unresolved."""

    def __init__(self, spinner_name: str = "dots", speed: float = 1.0):
        super().__init__(spinner_name=spinner_name, speed=speed)

    def render(self, task: Task):
        status = task.fields["status"]
        if status == "running":
            text = super().render(task)
            text.style = TaskProgress.color(status)
            return text

        # If task is not running, return a static text based on status.
        return Text(
            TaskProgress.marker(status),
            style=TaskProgress.color(status),
        )


class TaskProgressManager:
    """Manages task progress for multiple tasks in a Rich Progress widget.
    It manages the visibility of tasks based on their status and the
    maximum number of tasks to display at once. If there are more tasks
    than the maximum display count, it shows running tasks first,
    then completed tasks, and hides the rest."""

    def __init__(self, problems, max_display: int = 10) -> None:
        self.max_display = max_display
        self._tasks: Dict[str, TaskProgress] = {}
        self._progress_task_ids = {}  # Maps problem IDs to Rich task IDs

        self.progress = Progress(
            StatusColumn(),
            TextColumn("[progress.description]{task.description:<20}"),
            TextColumn("[blue]{task.fields[logfile]}[/blue]"),
            TextColumn("Step: [green]{task.completed:<4}[/green]  "),
            TextColumn(
                "Score: [green]{task.fields[score]:>3}/{task.fields[max_score]:<3}[/green]"
            ),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )
        self._tasks_panel = Panel(
            self.progress,
            title=self._get_tasks_panel_title(),
            title_align="left",
            border_style="green",
            padding=(1, 1),
        )

        for problem in problems:
            self.add_task(problem)

        self.refresh_progress()

    def add_task(self, task_id: str, total_steps: int = 1) -> int:
        """Add a new task to the progress manager and return its task ID."""
        task = TaskProgress(
            problem_id=task_id,
            step=0,
            total_steps=total_steps,
            score=0,
            max_score=0,
            status="pending",
            logfile="",
        )
        self._tasks[task_id] = task
        pid = self.progress.add_task(
            task.problem_id,
            status=task.status,
            completed=task.step,
            total=task.total_steps,
            score=task.score,
            max_score=task.max_score,
            logfile=task.logfile,
        )
        self._progress_task_ids[task.problem_id] = pid
        return pid

    def advance(self, progress_update: TaskProgress) -> None:
        """Advance the task progress based on the provided update."""
        task = self._tasks.get(str(progress_update.problem_id))
        if task:
            task.step = progress_update.step
            task.total_steps = progress_update.total_steps
            task.score = progress_update.score
            task.max_score = progress_update.max_score
            task.status = progress_update.status
            task.logfile = progress_update.logfile
            pid = self._progress_task_ids.get(task.problem_id)
            if pid is not None:
                self.progress.update(
                    pid,
                    completed=task.step,
                    total=task.total_steps,
                    status=task.status,
                    score=task.score,
                    max_score=task.max_score,
                    logfile=task.logfile,
                )

    def refresh_progress(self, all_tasks: bool = False):
        """Refresh the progress display, updating task visibility and panel title.
        If `all_tasks` is True, show all tasks regardless of the max-display."""
        # Update panel title
        self._tasks_panel.title = self._get_tasks_panel_title(all_tasks)

        visible_tasks = self._tasks if all_tasks else self._visible_tasks()
        # Set visibility for each task
        for task_id, task in self._tasks.items():
            pid = self._progress_task_ids.get(task_id)
            if pid is not None:
                is_visible = task_id in visible_tasks
                self.progress.update(pid, visible=is_visible)
        self.progress.refresh()

    def _visible_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get visible tasks limited to the maximum display count,
        showing pending/running tasks first, then completed tasks.

        Returns a dictionary mapping task IDs to their corresponding
        task data for visible tasks only."""
        # Get task IDs for pending, then completed tasks
        pending = []
        completed = []
        for tid, task in self._tasks.items():
            if task.completed:
                completed.append(tid)
            else:
                pending.append(tid)
        # Limit to max_display tasks, showing pending first
        visible_task_ids = (pending + completed)[: self.max_display]
        # Return the actual task data for the visible tasks
        return {tid: self._tasks[tid] for tid in visible_task_ids}

    def _get_tasks_panel_title(self, all_tasks=False):
        """Helper method to get the appropriate title for the tasks panel."""
        if all_tasks or self.max_display >= len(self._tasks):
            return "Tasks:"
        else:
            return f"In progress (max-display {self.max_display}):"

    def group_tasks_by_status(self) -> Dict[str, list[TaskProgress]]:
        """Group tasks by their current status."""
        tasks_per_status = {status: [] for status in TaskProgress.statuses()}
        # Count each task by its current status
        for task in self._tasks.values():
            # Make sure we have a valid status, defaulting to "pending" if unknown
            status = (
                task.status if task.status in TaskProgress.statuses() else "pending"
            )
            tasks_per_status[status].append(task.problem_id)
        return tasks_per_status

    def get_task_stats(self):
        """Get statistics about tasks: total, pending, running,
        resolved, unresolved, error, skip-resolved, skip-unresolved."""
        # Create a dictionary to count tasks by status
        status_counts = {s: len(t) for s, t in self.group_tasks_by_status().items()}
        # Calculate total (should match len(self._tasks) but this is more robust)
        status_counts["total"] = sum(status_counts.values())
        return status_counts


class OverallProgressContext:
    """Context manager for overall progress tracking across multiple tasks.
    It manages the Rich Progress display, task progress updates, and handles
    the background thread for listening to progress updates from worker processes.
    This class is designed to be used in a 'with' context from the main process."""

    def __init__(
        self,
        problems,
        max_display: int,
        live: Live,
        progress_queue: mp.Queue,
        logger: logging.Logger,
    ):
        self.problems = problems
        self.max_display = max_display
        self._live = live
        self.progress_queue = progress_queue
        self.logger = logger
        self.overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )
        self.total = len(problems)
        self.completed = 0
        self._overall_task = self.overall_progress.add_task(
            "Overall",  # Placeholder description, will be set by _refresh
            total=self.total,
        )
        self.tasks_progress = TaskProgressManager(problems, max_display)
        # Initialize table and panels once
        self._table = Table(show_header=False, show_edge=False)
        self._overall_panel = Panel(
            self.overall_progress,
            title=f"Overall ({self.total} tasks)",
            title_align="left",
            border_style="green",
            padding=(1, 0),
        )

        # Initialize table with panels
        self._table.add_row(self._overall_panel)
        self._table.add_row(self.tasks_progress._tasks_panel)
        self.progress_table = Padding(self._table, (1, 0, 0, 0))
        self.refresh_progress()

        # background thread for progress updates
        self._stop_event = threading.Event()
        self._listener_thread: threading.Thread = threading.Thread(
            target=self._status_listener, daemon=True
        )
        self._listener_thread.start()

    def advance(self, progress_update: TaskProgress):
        """Advance the progress for a specific task based on the provided update. Sets
        task as completed if its status is completed (e.g. early stopping)."""
        self.logger.debug(
            f"Advancing progress for problem {progress_update.problem_id}: "
            f"step {progress_update.step}"
        )
        # Update the task progress
        self.tasks_progress.advance(progress_update)
        # Update overall progress completion
        if progress_update.completed:
            self.completed += 1
            log_with_color(
                self.logger,
                f"{TaskProgress.marker(progress_update.status)} {progress_update.status}: "
                f" task {progress_update.problem_id}.",
                TaskProgress.color(progress_update.status),
            )

    def close(self):
        """Stop the listener thread and wait until it exits."""
        self._stop_event.set()
        self._listener_thread.join()
        self.logger.debug("Status listener thread exiting...")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def refresh_progress(self, all_tasks: bool = False):
        """Refresh the progress display, updating overall and tasks progress."""
        # Update overall progress with new stats
        stats = self.tasks_progress.get_task_stats()
        stats_text = (
            f"running: [blue]{stats['running']}[/blue] | "
            f"pending: [yellow]{stats['pending']}[/yellow] | "
            f"resolved: [green]{stats['resolved'] + stats['skip-resolved']}[/green] | "
            f"unresolved: [red]{stats['unresolved'] + stats['skip-unresolved']}[/red] | "
            f"error: [red]{stats['error']}[/red]"
        )
        self.overall_progress.update(
            self._overall_task,
            description=stats_text,
            completed=self.completed,
        )
        self.overall_progress.refresh()
        # Update panel content
        self.tasks_progress.refresh_progress(all_tasks=all_tasks)

    def status_report(self):
        """Prints the current progress table and tasks grouped by status."""
        self._live.console.print(self.progress_table)
        grouped_tasks = self.tasks_progress.group_tasks_by_status()
        self._live.console.print("\n")
        for status, tasks in grouped_tasks.items():
            if tasks:
                self._live.console.print(Padding(f"{status}: {tasks}", (0, 1)))
        self._live.console.print("\n")

    def _status_listener(self):
        self.logger.debug("Starting status listener thread...")
        while not self._stop_event.is_set():
            try:
                progress_update = self.progress_queue.get(timeout=0.1)
                self.advance(progress_update)
                self.refresh_progress()
            except queue.Empty:
                continue
            except EOFError:  # queue closed
                break
        # Final refresh to ensure UI is up-to-date
        self.refresh_progress(all_tasks=True)
        self.logger.debug("Status listener thread exiting...")


class DebugGymLogger(logging.Logger):
    """A multiprocess friendly logger that integrates with Rich for progress reporting.
    Multiprocess workers can use this logger to log messages and report progress via
    shared queues, which the main process processes and displays in a Rich UI."""

    LOG_QUEUE = mp.Queue(maxsize=10000)
    PROGRESS_QUEUE = mp.Queue(maxsize=10000)
    _is_worker = False
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of the logger is created (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(DebugGymLogger, cls).__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(
        self,
        name: str,
        log_dir: str | None = None,
        level: str | int = logging.INFO,
        mode: str = "a",
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return  # Already initialized

        super().__init__(name)
        # If var env "DEBUG_GYM_DEBUG" is set, turn on debug mode
        if os.environ.get("DEBUG_GYM_DEBUG"):
            level = logging.DEBUG

        # Prevent the log messages from being propagated to the root logger
        self.propagate = False

        self.setLevel(level)  # Set logger level, might be overridden by file handler
        self.log_file = None  # File handler for logging to a file

        # Placeholders for rich live, log listener thread, and stop event
        # Will be initialized if the logger is the main process logger
        self._live = None  # rich live context manager for updating the UI
        self.no_live = False  # Flag to disable live updates
        self._log_listener_stop_event = None  # Event to stop the log listener thread
        self._log_listener_thread = None  # Thread to process logs from workers
        if not self._is_worker:
            self._initialize_main_logger(level)
        if log_dir:
            self._initialize_file_handler(name, log_dir, mode)

        self._initialized = True

    def _initialize_main_logger(self, level):
        self._live = Live(transient=True, refresh_per_second=2)
        rich_handler = RichHandler(
            console=self._live.console,
            show_time=False,
            rich_tracebacks=True,
            markup=True,
        )
        rich_handler.setFormatter(logging.Formatter("🐸 [%(name)-12s]: %(message)s"))
        rich_handler.setLevel(level)
        self.addHandler(rich_handler)

        # Start log listener thread
        self._log_listener_stop_event = threading.Event()
        self._log_listener_thread = threading.Thread(
            target=self._log_listener, daemon=True
        )
        self._log_listener_thread.start()

    def _initialize_file_handler(self, name: str, log_dir: str, mode: str):
        self.setLevel(logging.DEBUG)  # Ensure logger operates at DEBUG level
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = (log_dir / f"{name}.log").absolute()
        fh = logging.FileHandler(self.log_file, mode=mode)
        formatter = StripAnsiFormatter("%(asctime)s %(levelname)-8s %(message)s")
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        self.addHandler(fh)
        self.info(f"Logging `{name}` to file: {self.log_file}")

    def handle(self, record):
        """Handle a log record. If this is a worker process,
        log to their own handlers (ex.: a file) and put the
        record into the log queue for the main process to display
        logs through Rich."""
        if self._is_worker:
            self.LOG_QUEUE.put(record)
        super().handle(record)

    def _log_listener(self):
        if not self._is_worker:
            while not self._log_listener_stop_event.is_set():
                try:
                    record = self.LOG_QUEUE.get(timeout=0.1)
                    # escape the message to avoid issues with Rich
                    # logger.info(f"[green]{escape(text)}[/green]", extra={"already_escaped": True})
                    if not getattr(record, "already_escaped", False):
                        record.msg = escape(record.msg)
                    super().handle(record)
                except queue.Empty:
                    continue
                except EOFError:
                    break

    def close(self):
        if self._log_listener_thread is not None:
            self._log_listener_stop_event.set()
            self._log_listener_thread.join()

    def __del__(self):
        self.close()

    @classmethod
    def set_as_worker(cls):
        """Set the logger as a worker logger, which means it will put logs and
        progress updates to the queues, letting the main process handle them."""
        cls._is_worker = True

    def set_no_live(self):
        """Set the logger to not use the Rich Live display."""
        self.no_live = True

    @contextmanager
    def rich_progress(self, problems, max_display: int = 10, final_report: bool = True):
        """Create a Rich progress bar for the given problems. To be used in a 'with' context.
        The context manager yields a `rich.Progress` but allows rich_progress to close
        the progress bar when the context is exited, ensuring that the UI is updated
        correctly and the thread is joined properly. Only the main process can manage
        the Rich UI, so this method raises an error if called in a worker process."""
        if self._is_worker:
            raise RuntimeError("Cannot use rich_progress in worker processes.")
        ctx = OverallProgressContext(
            problems, max_display, self._live, self.PROGRESS_QUEUE, self
        )
        # Start the Live display
        with self._live:
            # Update the live display with the progress table
            self._live.update(ctx.progress_table)
            if self.no_live:
                # Stop live updates after the first update
                self._live.stop()
            try:
                # Yield the context object itself so the caller can access both
                # the table and progress objects
                yield ctx
            finally:  # executed when the caller leaves the with-block
                # Close the context manager, which will stop the listener thread
                ctx.close()  # sets _stop_event and join()s the thread
        if final_report:
            ctx.status_report()

    def report_progress(
        self,
        problem_id: str,
        step: int,
        total_steps: int,
        score: int,
        max_score: int,
        status: str,
        max_attempts: int = 5,
    ) -> None:
        """Send a progress update to the shared queue for the main process to handle.
        This method is used by worker processes to report their progress."""

        # Get log file path relative to the current working directory if it exists.
        logfile = ""
        if self.log_file:
            logfile = rf"\[{self.log_file.relative_to(os.getcwd())}]"

        progress_update = TaskProgress(
            problem_id=problem_id,
            step=step,
            total_steps=total_steps,
            score=score,
            max_score=max_score,
            status=status,
            logfile=logfile,
        )
        # Put in queue with backoff strategy
        for attempt in range(max_attempts):
            try:
                self.PROGRESS_QUEUE.put(progress_update, timeout=1.0)
                self.debug(f"Reported progress: {progress_update}")
                return
            except queue.Full:
                # If queue is full, wait with exponential backoff
                if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                    time.sleep(0.1 * (2**attempt))
        self.warning(
            f"Failed to report progress for {problem_id} after {max_attempts} attempts"
        )


def log_with_color(logger: DebugGymLogger, message: str, color: str):
    """Log a message with a specific color, escape it for Rich, and mark it
    as already escaped for DebugGymLogger so it won't be escaped again."""
    logger.info(
        f"[{color}]{escape(message)}[/{color}]",
        extra={"already_escaped": True},
    )
