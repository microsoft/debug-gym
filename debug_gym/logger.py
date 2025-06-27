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


@dataclass(slots=True)  # Slitly faster / memory efficient when using slots
class ProgressUpdate:
    problem_id: int
    step: int
    score: int
    max_score: int
    status: str


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
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True,
        )
        self.total = len(problems)
        self.completed = 0
        self._overall_task = self.progress.add_task("Overall", total=self.total)

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
        self.completed += 1
        self.progress.update(
            self._overall_task,
            completed=self.completed,
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

    @staticmethod
    def _make_table(overall: Progress, jobs: Progress) -> Table:
        tbl = Table(show_header=False, show_edge=False)
        tbl.add_row(
            Panel(overall, title="Overall", border_style="green", padding=(1, 0))
        )
        # tbl.add_row(Panel(jobs, title="Problems", border_style="green", padding=(1, 1)))
        return tbl

    def _refresh(self, all_tasks: bool = False):
        self._live.update(self._make_table(self.progress, None))

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


class StripAnsiFormatter(logging.Formatter):

    def format(self, record):
        msg = super().format(record)
        return strip_ansi(msg)


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

        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            self.log_file = (log_dir / f"{name}.log").absolute()
            fh = logging.FileHandler(self.log_file, mode=mode)
            formatter = StripAnsiFormatter("%(asctime)s %(levelname)-8s %(message)s")
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            self.addHandler(fh)

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
        # Start drawing the Rich bar
        with ctx.progress:
            try:
                yield ctx.progress  # what the caller binds to progress_bar
            finally:  # executed when the caller leaves the with-block
                ctx.close()  # sets _stop_event and join()s the thread

    def report_progress(
        self, problem_id: int, step: int, score: int, max_score: int, status: str
    ) -> None:
        """Send a progress update to the shared queue."""
        progress_update = ProgressUpdate(
            problem_id=problem_id,
            step=step,
            score=score,
            max_score=max_score,
            status=status,
        )
        self.PROGRESS_QUEUE.put(progress_update)
        self.debug(f"Reported progress: {progress_update}")
