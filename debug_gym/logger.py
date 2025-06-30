import atexit
import logging
import multiprocessing as mp
import os
import queue
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener
from multiprocessing.queues import Queue as MPQueue  # real Queue class for typing
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich.live import Live
from rich.logging import RichHandler
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
class ProgressUpdate:
    problem_id: str
    step: int
    total_steps: int  # Total steps for the problem considering early stopping
    score: int
    max_score: int
    status: str


class StatusColumn(SpinnerColumn):
    """Custom status column. Spinner while task is running,
    green ✓ when completed or red ✗ when failed."""

    def __init__(self, spinner_name: str = "dots", speed: float = 1.0):
        super().__init__(spinner_name=spinner_name, speed=speed)

    def render(self, task):
        # TODO: implement a red ✗ for failed tasks
        return Text("✓", style="green") if task.finished else super().render(task)


class TaskProgressManager:
    """Stores per-task data and renders a Progress widget."""

    def __init__(self, max_display: int = 10) -> None:
        self._max_display = max_display
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def add_task(self, task_id: str, total_steps: int = 1) -> int:
        self._tasks[task_id] = dict(
            id=task_id,
            total=total_steps,  # Placeholder. Will be updated on advance().
            completed=0.0,
            finished=False,
        )
        return task_id

    def advance(self, progress_update: ProgressUpdate) -> None:
        task = self._tasks.get(str(progress_update.problem_id))
        if task:
            task["total"] = max(task["total"], progress_update.max_score)
            task["completed"] = min(
                task["completed"] + progress_update.step, task["total"]
            )
            task["finished"] = task["completed"] >= task["total"]

    @property
    def total_completed(self) -> float:
        return sum(t["completed"] for t in self._tasks.values())

    @property
    def total_work(self) -> float:
        return sum(t["total"] for t in self._tasks.values())

    @property
    def all_finished(self) -> bool:
        return all(t["finished"] for t in self._tasks.values())

    def _visible(self) -> Dict[str, Dict[str, Any]]:
        """Limits the number of tasks to self._max_display,
        showing pending tasks first."""
        # Get task IDs for pending, then completed tasks
        pend = [t_id for t_id, t in self._tasks.items() if not t["finished"]]
        done = [t_id for t_id, t in self._tasks.items() if t["finished"]]
        # Limit to max_display tasks, showing pending first
        visible_task_ids = (pend + done)[: self._max_display]
        # Return the actual task data for the visible tasks
        return {t_id: self._tasks[t_id] for t_id in visible_task_ids}

    def render(self, *, all_tasks: bool = False) -> Progress:
        progress = Progress(
            StatusColumn(),
            TextColumn("[progress.description]{task.description:<20}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )
        tasks = self._tasks if all_tasks else self._visible()
        for task in tasks.values():
            pid = progress.add_task(
                task["id"], total=task["total"], completed=task["completed"]
            )
            progress.update(pid, completed=task["completed"])
        return progress


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
        self._overall_task = self.progress.add_task("Overall", total=self.total)
        self.tasks_progress = TaskProgressManager(max_display)
        for problem in problems:
            self.tasks_progress.add_task(problem)

        self.progress_table = self._make_table(
            self.progress, self.tasks_progress.render(all_tasks=False)
        )

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
        self.completed += 1 if progress_update.status in ["done", "failed"] else 0
        self.progress.update(
            self._overall_task,
            completed=self.completed,
        )
        self.tasks_progress.advance(progress_update)

    def close(self):
        """Stop the listener thread and wait until it exits."""
        self._stop_event.set()
        self._listener_thread.join()
        self.logger.debug("Status listener thread exiting...")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def _make_table(overall: Progress, jobs: Progress) -> Table:
        tbl = Table(show_header=False, show_edge=False)
        tbl.add_row(
            Panel(overall, title="Overall", border_style="green", padding=(1, 0))
        )
        tbl.add_row(Panel(jobs, title="Problems", border_style="green", padding=(1, 1)))
        return tbl

    def _refresh(self, all_tasks: bool = False):
        self._live.update(
            self._make_table(
                self.progress, self.tasks_progress.render(all_tasks=all_tasks)
            )
        )

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
        if self._is_worker:
            self.LOG_QUEUE.put(record)
        else:
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
        # self._overall_progress = OverallProgressContext(
        #     problems, max_display, self._live, self.PROGRESS_QUEUE, self
        # )
        # return self._overall_progress.progress
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
        progress_update = ProgressUpdate(
            problem_id=problem_id,
            step=step,
            total_steps=total_steps,
            score=score,
            max_score=max_score,
            status=status,
        )
        self.PROGRESS_QUEUE.put(progress_update)
        self.debug(f"Reported progress: {progress_update}")
