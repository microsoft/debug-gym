import logging
import os
from pathlib import Path

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class FroggyLogger(logging.Logger):

    def __init__(
        self,
        name: str,
        log_dir: str | None = None,
        level: str | int = logging.INFO,
        mode: str = "a",
    ):
        super().__init__(name)
        self.setLevel(logging.DEBUG)

        # If var env "FROGGY_DEBUG" is set, turn on debug modeé
        if os.environ.get("FROGGY_DEBUG"):
            level = logging.DEBUG

        console = TqdmLoggingHandler()
        formatter = logging.Formatter("🐸 [%(name)-12s]: %(levelname)-8s %(message)s")
        console.setFormatter(formatter)
        console.setLevel(level)
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

    def tqdm(self, iterable, desc=None, *args, **kwargs):
        desc = desc or f"  [{self.name:12s}]"
        kwargs.pop("leave", None)
        yield from tqdm(iterable, desc=desc, *args, **kwargs, leave=False)
