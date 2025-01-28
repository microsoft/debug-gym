import logging
import re
from pathlib import Path

from rich.console import Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Column


class ProgressHandler(RichHandler):
    def __init__(self, progress, task_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_id = task_id
        self.progress = progress

    def emit(self, record):
        if self.task_id not in self.progress.task_ids:
            return

        # Strip color codes from the log message
        message = re.sub(r"\x1b\[[0-9;]*m", "", self.format(record))
        message = message.replace("\n", "\\n").replace("\r", "\\r")
        # Truncate message to 80 characters
        message = message[:80] + "..." if len(message) > 80 else message
        self.progress.update(self.task_id, log=message)


class FroggyLogger(logging.Logger):
    task_progress = Progress(
        TimeElapsedColumn(),
        BarColumn(bar_width=10),
        TextColumn("{task.description}"),
        TextColumn(
            "{task.fields[log]}", table_column=Column(no_wrap=True)  # , width=80)
        ),
    )
    overall_progress = Progress(
        TextColumn("🐸"),
        TimeElapsedColumn(),
        BarColumn(),
        TextColumn("{task.description}"),
    )
    progress_group = Group(
        Panel(task_progress, title="Workers"),
        overall_progress,
    )

    def __init__(
        self,
        name: str,
        log_dir: str | None = None,
        verbose: bool = False,
        mode: str = "a",
    ):
        super().__init__(name)
        self.setLevel(logging.DEBUG)
        self.task_id = self.task_progress.add_task(
            f"\\[{name}]:", log="Starting task..."
        )

        ph = ProgressHandler(self.task_progress, self.task_id)
        ph.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter("%(levelname)-8s %(message)s")
        ph.setFormatter(formatter)
        self.addHandler(ph)

        console = logging.StreamHandler()
        formatter = logging.Formatter("🐸 [%(name)-12s]: %(levelname)-8s %(message)s")
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG if verbose else logging.INFO)
        self.addHandler(console)

        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            self.log_file = log_dir / f"{name}.log"
            fh = logging.FileHandler(self.log_file, mode=mode)
            formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)
            self.addHandler(fh)

        # Prevent the log messages from being propagated to the root logger
        self.propagate = False

    def tqdm(self, iterable, total=None, desc=None, unit="it", ncols=80):
        total = len(iterable) if total is None else total
        desc = "" if desc is None else desc
        unit = "" if unit is None else unit
        self.task_progress.update(self.task_id, total=total)
        for i, item in enumerate(iterable):
            self.task_progress.update(self.task_id, advance=1)
            yield item

    def stop(self, remove: bool = False):
        self.task_progress.stop(self.task_id)
        if remove:
            self.task_progress.remove_task(self.task_id)
