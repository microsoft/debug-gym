import logging
import multiprocessing as mp
import os
import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.queues import Queue as MPQueue
from pathlib import Path
from typing import Any, Dict

from rich.live import Live
from rich.logging import RichHandler
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from debug_gym.utils import strip_ansi


class StripAnsiFormatter(logging.Formatter):

    def format(self, record):
        msg = super().format(record)
        return strip_ansi(msg)


@dataclass(slots=True)  # Slitly faster / memory efficient when using slots
class TaskProgress:
    problem_id: str
    step: int
    total_steps: int  # Total steps for the problem considering early stopping
    score: int
    max_score: int
    status: str

    @property
    def completed(self) -> bool:
        """Check if the task is completed based on its status."""
        return self.status in ["done", "failed"]


class StatusColumn(SpinnerColumn):
    """Custom status column. Spinner while task is running,
    green ✓ when completed or red ✗ when failed."""

    def __init__(self, spinner_name: str = "dots", speed: float = 1.0):
        super().__init__(spinner_name=spinner_name, speed=speed)

    def render(self, task):
        if task.finished:
            if task.fields.get("status") == "failed":
                return Text("✗", style="red")
            return Text("✓", style="green")
        return super().render(task)


class TaskProgressManager:
    """Stores per-task data and renders a Progress widget."""

    def __init__(self, problems, max_display: int = 10) -> None:
        self._max_display = max_display
        self._tasks: Dict[str, TaskProgress] = {}
        self._progress = Progress(
            StatusColumn(),
            TextColumn("[progress.description]{task.description:<20}"),
            TextColumn("Step: [green]{task.completed}[/green]"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )
        self._progress_task_ids = {}
        for problem in problems:
            self.add_task(problem)

    def add_task(self, task_id: str, total_steps: int = 1) -> int:
        task = TaskProgress(
            problem_id=task_id,
            step=0,
            total_steps=total_steps,
            score=0,
            max_score=0,
            status="pending",
        )
        self._tasks[task_id] = task
        pid = self._progress.add_task(
            task.problem_id, completed=task.step, total=task.total_steps
        )
        self._progress_task_ids[task.problem_id] = pid
        return pid

    def advance(self, progress_update: TaskProgress) -> None:
        task = self._tasks.get(str(progress_update.problem_id))
        if task:
            task.step = progress_update.step
            task.total_steps = progress_update.total_steps
            task.score = progress_update.score
            task.max_score = progress_update.max_score
            task.status = progress_update.status
            pid = self._progress_task_ids.get(task.problem_id)
            if pid is not None:
                self._progress.update(
                    pid,
                    completed=task.step,
                    total=task.total_steps,
                    status=task.status,
                )

    def _visible(self) -> Dict[str, Dict[str, Any]]:
        """Limits the number of tasks to self._max_display,
        showing pending tasks first."""
        # Get task IDs for pending, then completed tasks
        pending = [tid for tid, t in self._tasks.items() if not t.completed]
        completed = [tid for tid, t in self._tasks.items() if t.completed]
        # Limit to max_display tasks, showing pending first
        visible_task_ids = (pending + completed)[: self._max_display]
        # Return the actual task data for the visible tasks
        return {tid: self._tasks[tid] for tid in visible_task_ids}

    def render(self, *, all_tasks: bool = False) -> Progress:
        tasks = self._tasks if all_tasks else self._visible()

        # Clear the progress bar
        for task_id in list(self._progress.task_ids):
            self._progress.remove_task(task_id)
            self._progress_task_ids.pop(task_id, None)

        # Re-add tasks ordered by status
        for task_id, task in tasks.items():
            pid = self._progress.add_task(
                task.problem_id,
                total=task.total_steps,
                completed=task.step,
                status=task.status,
            )
            # Rich doesn't set the task as completed when adding it
            self._progress.update(
                pid,
                completed=task.step,
            )
            self._progress_task_ids[task.problem_id] = pid
        return self._progress


class OverallProgressContext:
    def __init__(
        self,
        problems,
        max_display: int,
        live: Live,
        progress_queue: MPQueue,
        logger: logging.Logger,
    ):
        self.problems = problems
        self.max_display = max_display
        self._live = live
        self.progress_queue = progress_queue
        self.logger = logger
        self.progress = Progress(  # TODO: rename to overall_progress
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )
        self.total = len(problems)
        self.completed = 0
        self._overall_task = self.progress.add_task(
            "Overall",  # Placeholder description, will be set by _refresh
            total=self.total,
        )
        self.tasks_progress = TaskProgressManager(problems, max_display)
        self.progress_table = self._refresh()

        # background thread for progress updates
        self._stop_event = threading.Event()
        self._listener_thread: threading.Thread = threading.Thread(
            target=self._status_listener, daemon=True
        )
        self._listener_thread.start()

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return self.completed >= self.total

    def advance(self, progress_update):
        self.logger.debug(
            f"Advancing progress for problem {progress_update.problem_id}: "
            f"step {progress_update.step}"
        )
        # Update the task progress
        self.tasks_progress.advance(progress_update)
        # Update overall progress
        self.completed += 1 if progress_update.status in ["done", "failed"] else 0
        self._refresh()

    def close(self):
        """Stop the listener thread and wait until it exits."""
        self._stop_event.set()
        self._listener_thread.join()
        self.logger.debug("Status listener thread exiting...")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _make_table(self, all_tasks=False) -> Table:
        tbl = Table(show_header=False, show_edge=False)
        tbl.add_row(
            Panel(
                self.progress,
                title=f"Overall ({self.total} tasks)",
                title_align="left",
                border_style="green",
                padding=(1, 0),
            )
        )
        if self.total <= self.max_display:
            per_task_title = "In progress:"
        else:
            per_task_title = f"In progress (max display {self.max_display}):"
        tbl.add_row(
            Panel(
                self.tasks_progress.render(all_tasks=all_tasks),
                title=per_task_title,
                title_align="left",
                border_style="green",
                padding=(1, 1),
            )
        )
        return Padding(tbl, (1, 0, 0, 0))

    def _refresh(self, all_tasks: bool = False):
        # Get updated stats
        stats = self.get_task_stats()
        stats_text = (
            f"Running: [blue]{stats['running']}[/blue] | "
            f"Pending: [yellow]{stats['pending']}[/yellow] | "
            f"Completed: [green]{stats['completed']}[/green] | "
            f"Failed: [red]{stats['failed']}[/red]"
        )
        # Update overall progress with new stats
        self.progress.update(
            self._overall_task,
            description=stats_text,
            completed=self.completed,
        )
        self.progress_table = self._make_table(all_tasks=all_tasks)
        self._live.update(self.progress_table)
        return self.progress_table

    def _status_listener(self):
        self.logger.debug("Starting status listener thread...")
        while not self._stop_event.is_set():
            try:
                progress_update = self.progress_queue.get(timeout=0.1)
                self.logger.info(f"Received progress update: {progress_update}")
                self.advance(progress_update)
                self._refresh()
            except queue.Empty:
                continue
            except EOFError:  # queue closed
                break
        self.logger.debug("Status listener thread exiting...")

    def get_task_stats(self):
        """Get statistics about tasks: total, pending, completed, failed."""
        tasks = self.tasks_progress._tasks.values()
        total = len(tasks)
        completed = sum(1 for t in tasks if t.status == "done")
        failed = sum(1 for t in tasks if t.status == "failed")
        running = sum(1 for t in tasks if t.status == "running")
        pending = total - completed - failed - running
        return {
            "total": total,
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
        }


class DebugGymLogger(logging.Logger):
    """A multiprocess friendly logger that integrates with Rich for progress reporting.
    Multiprocess workers can use this logger to log messages and report progress via
    shared queues, which the main process processes and displays in a Rich UI."""

    LOG_QUEUE = mp.Queue()
    PROGRESS_QUEUE = mp.Queue()
    _is_worker = False

    def __init__(
        self,
        name: str,
        log_dir: str | None = None,
        level: str | int = logging.INFO,
        mode: str = "a",
    ):
        super().__init__(name)

        # If var env "DEBUG_GYM_DEBUG" is set, turn on debug mode
        if os.environ.get("DEBUG_GYM_DEBUG"):
            level = logging.DEBUG

        self.level = level
        self.setLevel(self.level)
        self._live = None  # rich live context manager for updating the UI
        if not self._is_worker:
            self._live = Live()
            rich_handler = RichHandler(
                console=self._live.console,
                show_time=False,
                rich_tracebacks=True,
                markup=True,
            )
            rich_handler.setFormatter(
                logging.Formatter("🐸 [%(name)-12s]: %(message)s")
            )
            rich_handler.setLevel(self.level)
            self.addHandler(rich_handler)

            # Start log listener thread
            self._log_listener_stop_event = threading.Event()
            self._log_listener_thread = threading.Thread(
                target=self._log_listener, daemon=True
            )
            self._log_listener_thread.start()

        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            self.log_file = (log_dir / f"{name}.log").absolute()
            fh = logging.FileHandler(self.log_file, mode=mode)
            formatter = StripAnsiFormatter("%(asctime)s %(levelname)-8s %(message)s")
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            self.addHandler(fh)

    def handle(self, record):
        """Handle a log record. If this is a worker process,
        log to their own handlers (ex.: a file) and put the
        record into the log queue for the main process to display
        logs through Rich."""
        if self._is_worker:
            self.LOG_QUEUE.put(record)
        super().handle(record)

    def _log_listener(self):
        while not self._log_listener_stop_event.is_set():
            try:
                record = self.LOG_QUEUE.get(timeout=0.1)
                super().handle(record)
            except queue.Empty:
                continue
            except EOFError:
                break

    def close(self):
        if self._log_listener_thread:
            self._log_listener_stop_event.set()
            self._log_listener_thread.join()

    def __del__(self):
        self.close()

    @classmethod
    def set_as_worker(cls):
        """Set the logger as a worker logger, which means it will put logs and
        progress updates to the queues, letting the main process handle them."""
        cls._is_worker = True

    @contextmanager
    def rich_progress(self, problems, max_display: int = 10):
        """Create a Rich progress bar for the given problems. To be used in a 'with' context.
        The context manager yields a `rich.Progress` but allows rich_progress to close
        the progress bar when the context is exited, ensuring that the UI is updated
        correctly and the thread is joined properly."""
        if self._is_worker:
            raise RuntimeError("Cannot use rich_progress in worker processes.")
        ctx = OverallProgressContext(
            problems, max_display, self._live, self.PROGRESS_QUEUE, self
        )
        # Start the Live display
        with self._live:
            # Update the live display with the progress table
            self._live.update(ctx.progress_table)

            try:
                # Yield the context object itself so the caller can access both
                # the table and progress objects
                yield ctx
            finally:  # executed when the caller leaves the with-block
                ctx.close()  # sets _stop_event and join()s the thread

    def report_progress(
        self,
        problem_id: str,
        step: int,
        total_steps: int,
        score: int,
        max_score: int,
        status: str,
    ) -> None:
        """Send a progress update to the shared queue."""
        progress_update = TaskProgress(
            problem_id=problem_id,
            step=step,
            total_steps=total_steps,
            score=score,
            max_score=max_score,
            status=status,
        )
        self.PROGRESS_QUEUE.put(progress_update)
        self.debug(f"Reported progress: {progress_update}")
