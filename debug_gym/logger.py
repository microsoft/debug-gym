import json
import logging
import multiprocessing as mp
import os
import queue
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from rich.live import Live
from rich.logging import RichHandler
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


class StripAnsiFormatter(logging.Formatter):

    def format(self, record):
        msg = super().format(record)
        return re.sub(r"\x1B[@-_][0-?]*[ -/]*[@-~]", "", msg)


@dataclass(slots=True)  # Slitly faster / memory efficient when using slots
class TaskProgress:
    """Data class to communicate task progress information."""

    problem_id: str
    step: int
    total_steps: int  # Total steps for the problem considering early stopping
    score: int
    max_score: int | None
    status: str
    logdir: str = ""

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

    @property
    def log_file_path(self) -> str:
        return (
            str(
                log_file_path(
                    self.logdir,
                    "debug_gym",
                    relative=True,
                )
            )
            if self.logdir
            else ""
        )


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


class ScoreColumn(TextColumn):
    """Custom column for displaying score with proper None handling."""

    def __init__(self):
        super().__init__("")

    def render(self, task: Task):
        score = task.fields.get("score", 0)
        max_score = task.fields.get("max_score")
        max_score_str = str(max_score) if max_score is not None else "?"
        return Text(f"Score: {score:>3}/{max_score_str:>3}", style="green")


def log_file_path(log_dir, problem_id, relative=False) -> Path:
    """Return the path to the log file for a given problem. If `relative` is True,
    it returns a relative path from the current working directory. If the log_dir
    is not a subdir of the cwd, returns the absolute path."""
    logfile = (Path(log_dir) / f"{problem_id}.log").absolute()
    if relative:
        try:
            logfile = logfile.relative_to(os.getcwd())
        except ValueError:
            # If the log_dir is not a subdir of the cwd, return the absolute path
            pass
    return logfile


def status_json_path(log_dir, problem_id) -> Path:
    """Return the path to the status.json file for a given problem."""
    return Path(log_dir) / f"{problem_id}_status.json"


def load_previous_run_status(log_dir: str, problem_id: str) -> TaskProgress | None:
    """Load the previous run progress from a JSON file."""
    status_path = status_json_path(log_dir, problem_id)
    if status_path.exists():
        with open(status_path, "r") as f:
            data = json.load(f)
            return TaskProgress(**data)
    return None


class TaskProgressManager:
    """Manages task progress for multiple tasks in a Rich Progress widget.
    It manages the visibility of tasks based on their status and the
    maximum number of tasks to display at once. If there are more tasks
    than the maximum display count, it shows running tasks first,
    then completed tasks, and hides the rest."""

    def __init__(
        self, problems, max_display: int = 10, logger: "DebugGymLogger" = None
    ) -> None:
        self.max_display = max_display
        self.logger = logger
        self._tasks: Dict[str, TaskProgress] = {}
        self._progress_task_ids = {}  # Maps problem IDs to Rich task IDs

        self.progress = Progress(
            StatusColumn(),
            TextColumn("[progress.description]{task.description:<20}"),
            TextColumn("[blue]{task.fields[logfile]}[/blue]"),
            TextColumn("Step: [green]{task.completed:<4}[/green]  "),
            ScoreColumn(),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
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
            max_score=None,
            status="pending",
            logdir="",
        )
        self._tasks[task_id] = task
        pid = self.progress.add_task(
            task.problem_id,
            status=task.status,
            completed=task.step,
            total=task.total_steps,
            score=task.score,
            max_score=task.max_score,
            logfile=task.log_file_path,
        )
        self._progress_task_ids[task.problem_id] = pid
        return pid

    def advance(self, progress_update: TaskProgress) -> None:
        """Advance the task progress based on the provided update."""
        task = self._tasks.get(str(progress_update.problem_id))
        if task:
            # Update the cached task (TaskProgress) with the new progress
            task.step = progress_update.step
            task.total_steps = progress_update.total_steps
            task.score = progress_update.score
            task.max_score = progress_update.max_score
            task.status = progress_update.status
            task.logdir = progress_update.logdir
            # Log and dump final status
            if progress_update.completed:
                log_with_color(
                    self.logger,
                    f"{TaskProgress.marker(task.status)} {task.status}: "
                    f" task {task.problem_id}.",
                    TaskProgress.color(task.status),
                )
                if task.status not in ["skip-resolved", "skip-unresolved"]:
                    self.dump_task_status(task)
            # Update the Rich task
            pid = self._progress_task_ids.get(task.problem_id)
            if pid is not None:
                self.progress.update(
                    pid,
                    completed=task.step,
                    total=task.total_steps,
                    status=task.status,
                    score=task.score,
                    max_score=task.max_score,
                    logfile=task.log_file_path,
                )

    def dump_task_status(self, task: TaskProgress):
        if not task.logdir:
            self.logger.warning(
                f"No logdir set for task {task.problem_id}. "
                "Skipping task status dump."
            )
            return
        status_path = status_json_path(task.logdir, task.problem_id)
        task_dict = asdict(task)
        self.logger.debug(f"Dumping task status to JSON: {status_path}")
        with open(status_path, "w") as f:
            json.dump(task_dict, f)

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

    def _visible_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get visible tasks limited to the maximum display count,
        showing running tasks first, then pending, then failed ones, then completed tasks.

        Returns a dictionary mapping task IDs to their corresponding
        task data for visible tasks only."""
        # Get task IDs for running, pending, failed, and completed tasks
        running = []
        pending = []
        failed = []
        completed = []
        for tid, task in self._tasks.items():
            if task.status == "running":
                running.append(tid)
            elif task.status == "pending":
                pending.append(tid)
            elif task.status in ("error", "unresolved"):
                failed.append(tid)
            elif task.completed:
                completed.append(tid)
        # Limit to max_display tasks, showing running first, then pending, then failed, then completed
        visible_task_ids = (running + pending + failed + completed)[: self.max_display]
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
        logger: "DebugGymLogger",
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
            TimeElapsedColumn(),
            expand=True,
        )
        self.total = len(problems)
        self.completed = 0
        self._overall_task = self.overall_progress.add_task(
            "Overall",  # Placeholder description, will be set by _refresh
            total=self.total,
        )
        self.tasks_progress = TaskProgressManager(problems, max_display, logger)
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
        self.completed += 1 if progress_update.completed else 0

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
        resolved = stats["resolved"] + stats["skip-resolved"]
        unresolved = stats["unresolved"] + stats["skip-unresolved"] + stats["error"]
        total_so_far = (resolved + unresolved) or 1  # Avoid division by zero
        perf_so_far = f"{100. * float(resolved) / (total_so_far):.2f}%"
        stats_text = (
            f"running: [blue]{stats['running']}[/blue] | "
            f"pending: [yellow]{stats['pending']}[/yellow] | "
            f"resolved: [green]{stats['resolved'] + stats['skip-resolved']}[/green] | "
            f"unresolved: [red]{stats['unresolved'] + stats['skip-unresolved']}[/red] | "
            f"error: [red]{stats['error']}[/red] | "
            f"acc: [green]{perf_so_far}[/green]"
        )
        self.overall_progress.update(
            self._overall_task,
            description=stats_text,
            completed=self.completed,
        )
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
        last_refresh = time.time()
        refresh_interval = 0.1  # Refresh UI every 100ms instead of after every update

        while not self._stop_event.is_set():
            try:
                progress_update = self.progress_queue.get(timeout=0.1)
                self.advance(progress_update)

                # Only refresh UI periodically to avoid overwhelming the display
                current_time = time.time()
                if current_time - last_refresh >= refresh_interval:
                    self.refresh_progress()
                    last_refresh = current_time

            except queue.Empty:
                # Refresh UI even when no updates to ensure latest state is shown
                current_time = time.time()
                if current_time - last_refresh >= refresh_interval:
                    self.refresh_progress()
                    last_refresh = current_time
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
    PROGRESS_QUEUE = mp.Queue(maxsize=30000)  # macOS has semaphore limit ~32767
    _is_worker = False
    _main_process_logger = None

    @classmethod
    def is_worker(cls):
        return cls._is_worker

    @classmethod
    def is_main(cls):
        return not cls._is_worker

    @classmethod
    def set_as_worker(cls):
        """Set the logger as a worker logger, which means it will put logs and
        progress updates to the queues, letting the main process handle them."""
        cls._is_worker = True

    def __init__(
        self,
        name: str,
        log_dir: str | None = None,
        level: str | int = logging.INFO,
        mode: str = "a",
    ):
        super().__init__(name)
        # If var env "DEBUG_GYM_DEBUG" is set, turn on debug mode
        if os.environ.get("DEBUG_GYM_DEBUG", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            level = logging.DEBUG

        # Prevent the log messages from being propagated to the root logger
        self.propagate = False

        super().setLevel(level)  # Set initial logger level
        if DebugGymLogger._main_process_logger is not None:
            self._is_worker = True

        # Placeholders for rich live, log listener thread, and stop event
        # Will be initialized if the logger is the main process logger
        self._live = None  # rich live context manager for updating the UI
        self.no_live = False  # Flag to disable live updates
        self._log_listener_stop_event = None  # Event to stop the log listener thread
        self._log_listener_thread = None  # Thread to process logs from workers
        self._rich_handler = None  # Handler for Rich console output
        if self.is_main():
            self._initialize_main_logger(level)
            DebugGymLogger._main_process_logger = self

        self.log_file = None  # File handler for logging to a file
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:  # Directory to store log files
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._initialize_file_handler(name, mode)
            self.info(f"Logging to directory: {self.log_dir}")

    def _initialize_main_logger(self, level):
        self._live = Live(transient=True, refresh_per_second=2)
        self._rich_handler = RichHandler(
            console=self._live.console,
            show_time=False,
            rich_tracebacks=True,
            markup=False,
        )
        self._rich_handler.setFormatter(
            logging.Formatter("🐸 [%(name)-12s]: %(message)s")
        )
        self._rich_handler.setLevel(level)
        self.addHandler(self._rich_handler)

        # Start log listener thread
        self._log_listener_stop_event = threading.Event()
        self._log_listener_thread = threading.Thread(
            target=self._log_listener, daemon=True
        )
        self._log_listener_thread.start()

    def _initialize_file_handler(self, name: str, mode: str):
        super().setLevel(logging.DEBUG)  # Ensure logger operates at DEBUG level
        self.log_file = log_file_path(self.log_dir, "debug_gym")
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
        if self.is_worker():
            self.LOG_QUEUE.put(record)
        super().handle(record)

    def _log_listener(self):
        if self.is_main():
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

    def setLevel(self, level: int | str) -> None:
        """Set the logging level for console output.

        When a file handler is configured, the logger's internal level remains at
        DEBUG to ensure all messages are saved to the log file. This method only
        changes the level of the rich_handler (console output), allowing users to
        control what is displayed to stdout without affecting the file logs.

        Args:
            level: The logging level to set. Can be an integer (e.g., logging.INFO,
                logging.DEBUG) or a string (e.g., 'INFO', 'DEBUG').
        """
        if self._rich_handler is not None:
            self._rich_handler.setLevel(level)
        else:
            # Fallback for when rich_handler is not initialized (e.g., worker process)
            super().setLevel(level)

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
        if self.is_worker():
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
        max_score: int | None,
        status: str,
        max_attempts: int = 5,
    ) -> None:
        """Send a progress update to the shared queue for the main process to handle.
        This method is used by worker processes to report their progress."""
        progress_update = TaskProgress(
            problem_id=problem_id,
            step=step,
            total_steps=total_steps,
            score=score,
            max_score=max_score,
            status=status,
            logdir=str(self.log_dir) if self.log_dir else "",
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


def log_with_color(
    logger: DebugGymLogger, message: str, color: str, level=logging.INFO
):
    """Log a message with a specific color, escape it for Rich, and mark it667
    as already escaped for DebugGymLogger so it won't be escaped again."""
    logger.log(
        level,
        f"[{color}]{escape(message)}[/{color}]",
        extra={"already_escaped": True, "markup": True},
    )
