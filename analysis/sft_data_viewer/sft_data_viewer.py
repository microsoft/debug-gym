import argparse
import atexit
import errno
import json
import logging
import math
import os
import stat
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.security import safe_join
from werkzeug.utils import secure_filename

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5001
DEFAULT_SAFE_ROOT = Path.cwd().resolve()
DEFAULT_TRUSTED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
ALLOWED_SUFFIXES = {".jsonl"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
app.config["SAFE_ROOT"] = (
    Path(os.environ.get("SFT_DATA_VIEWER_SAFE_ROOT", DEFAULT_SAFE_ROOT))
    .expanduser()
    .resolve()
)
app.config["TRUSTED_HOSTS"] = list(DEFAULT_TRUSTED_HOSTS)

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@dataclass(frozen=True)
class LoadedFileState:
    descriptor: int
    filepath: Path
    filename: str
    total_records: int
    owned_upload: bool


current_file_state: LoadedFileState | None = None
current_file_lock = threading.Lock()
records_per_page = 10


class SafePathError(ValueError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def configure_safe_root(raw_root: str | Path) -> Path:
    try:
        safe_root = Path(raw_root).expanduser().resolve(strict=True)
        root_stat = safe_root.stat()
    except (OSError, RuntimeError) as exc:
        raise SafePathError("The configured safe root is unavailable", 500) from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise SafePathError("The configured safe root is unavailable", 500)
    app.config["SAFE_ROOT"] = safe_root
    app.config["SAFE_ROOT_IDENTITY"] = (root_stat.st_dev, root_stat.st_ino)
    return safe_root


def _safe_path_message(status_code: int) -> str:
    return {
        400: "Invalid path",
        403: "Access denied",
        404: "File not found",
        409: "File changed while it was being loaded",
        500: "File access is unavailable",
    }.get(status_code, "Unable to access file")


def _safe_candidate(
    raw_path: str,
    *,
    allowed_suffixes: set[str] | None = None,
) -> tuple[Path, Path]:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise SafePathError("A path is required", 400)
    if "\\" in raw_path:
        raise SafePathError("Invalid path", 400)

    safe_root = Path(app.config["SAFE_ROOT"])
    joined = safe_join(str(safe_root), raw_path)
    if joined is None:
        raise SafePathError("Access denied", 403)
    candidate = Path(joined)
    try:
        relative = candidate.relative_to(safe_root)
    except ValueError as exc:
        raise SafePathError("Access denied", 403) from exc
    if (
        allowed_suffixes is not None
        and candidate.suffix.lower() not in allowed_suffixes
    ):
        raise SafePathError("Invalid file type", 400)
    return candidate, relative


def _is_symbolic_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def open_confined_server_file(raw_path: str) -> int:
    """Open one regular, single-link file beneath the pinned safe root."""
    candidate, relative = _safe_candidate(
        raw_path,
        allowed_suffixes=ALLOWED_SUFFIXES,
    )
    if not relative.parts:
        raise SafePathError("File not found", 404)
    safe_root = Path(app.config["SAFE_ROOT"])
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)

    if os.open in os.supports_dir_fd and no_follow:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow
        root_descriptor = None
        parent_descriptor = None
        try:
            root_descriptor = os.open(safe_root, directory_flags)
            root_stat = os.fstat(root_descriptor)
            if (root_stat.st_dev, root_stat.st_ino) != app.config.get(
                "SAFE_ROOT_IDENTITY"
            ):
                raise SafePathError("The configured safe root changed", 409)
            parent_descriptor = root_descriptor
            root_descriptor = None
            for part in relative.parts[:-1]:
                next_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=parent_descriptor,
                )
                os.close(parent_descriptor)
                parent_descriptor = next_descriptor
            descriptor = os.open(
                relative.parts[-1],
                flags | no_follow,
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError as exc:
            raise SafePathError("File not found", 404) from exc
        except NotADirectoryError as exc:
            raise SafePathError("File not found", 404) from exc
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK}:
                raise SafePathError("Access denied", 403) from exc
            raise
        finally:
            if parent_descriptor is not None:
                os.close(parent_descriptor)
            if root_descriptor is not None:
                os.close(root_descriptor)
    else:
        current = safe_root
        for part in relative.parts:
            current /= part
            if _is_symbolic_link(current):
                raise SafePathError("Access denied", 403)
        try:
            descriptor = os.open(candidate, flags | no_follow)
        except FileNotFoundError as exc:
            raise SafePathError("File not found", 404) from exc
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK}:
                raise SafePathError("Access denied", 403) from exc
            raise
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(safe_root)
            path_stat = os.stat(resolved, follow_symlinks=False)
            descriptor_stat = os.fstat(descriptor)
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
                path_stat.st_dev,
                path_stat.st_ino,
            ):
                raise SafePathError("File changed while it was being loaded", 409)
        except (OSError, RuntimeError, ValueError):
            os.close(descriptor)
            raise

    descriptor_stat = os.fstat(descriptor)
    if not stat.S_ISREG(descriptor_stat.st_mode):
        os.close(descriptor)
        raise SafePathError("File not found", 404)
    if descriptor_stat.st_nlink != 1:
        os.close(descriptor)
        raise SafePathError("Files with multiple hard links are not allowed", 403)
    return descriptor


configure_safe_root(app.config["SAFE_ROOT"])


@app.before_request
def reject_cross_origin_mutations():
    if request.method != "POST":
        return None
    origin = request.headers.get("Origin")
    if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
        return jsonify({"error": "Origin not allowed"}), 403
    return None


def save_uploaded_file(file_storage) -> tuple[int, Path, str]:
    """Save an upload through an exclusive descriptor with a generated name."""
    filename = secure_filename(file_storage.filename)
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise SafePathError("Please upload a JSONL file", 400)

    configured_root = Path(app.config["UPLOAD_FOLDER"]).absolute()
    if configured_root.is_symlink():
        raise OSError("Upload folder must not be a symbolic link")
    configured_root.mkdir(parents=True, exist_ok=True)
    upload_root = configured_root.resolve(strict=True)
    filepath = upload_root / f".debug-gym-upload-{uuid.uuid4().hex}.jsonl"
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    descriptor = os.open(filepath, flags, 0o600)
    completed = False
    try:
        with os.fdopen(os.dup(descriptor), "wb") as destination:
            file_storage.save(destination)
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode) or descriptor_stat.st_nlink != 1:
            raise OSError("Upload destination is not a regular file")
        if descriptor_stat.st_size > app.config["MAX_CONTENT_LENGTH"]:
            raise OSError("Uploaded file exceeds the size limit")
        os.lseek(descriptor, 0, os.SEEK_SET)
        completed = True
        return descriptor, filepath, filename
    finally:
        if not completed:
            os.close(descriptor)
            filepath.unlink(missing_ok=True)


@contextmanager
def open_descriptor(descriptor: int):
    duplicate = os.dup(descriptor)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with os.fdopen(duplicate, "r", encoding="utf-8") as file:
            duplicate = None
            yield file
    finally:
        if duplicate is not None:
            os.close(duplicate)


@contextmanager
def open_current_state():
    with current_file_lock:
        if current_file_state is None:
            raise RuntimeError("No file loaded")
        with open_descriptor(current_file_state.descriptor) as file:
            yield current_file_state, file


def _dispose_state(state: LoadedFileState) -> None:
    os.close(state.descriptor)
    if state.owned_upload:
        state.filepath.unlink(missing_ok=True)


def replace_current_file(
    descriptor: int,
    filepath: Path,
    filename: str,
    total_records: int,
    owned_upload: bool,
) -> None:
    global current_file_state
    replacement = LoadedFileState(
        descriptor=descriptor,
        filepath=filepath,
        filename=filename,
        total_records=total_records,
        owned_upload=owned_upload,
    )
    with current_file_lock:
        previous_state = current_file_state
        current_file_state = None
        if previous_state is not None:
            _dispose_state(previous_state)
        current_file_state = replacement


def clear_current_file() -> None:
    global current_file_state
    with current_file_lock:
        previous_state = current_file_state
        current_file_state = None
        if previous_state is not None:
            _dispose_state(previous_state)


def _clear_current_file_at_exit() -> None:
    try:
        clear_current_file()
    except OSError:
        logging.exception("Failed to clean up an uploaded SFT file during shutdown")


atexit.register(_clear_current_file_at_exit)


def to_pretty_json(value):
    """Convert Python object to pretty-printed JSON string with minimal whitespace"""
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).strip()


# Add custom filters and globals to Jinja2
app.jinja_env.filters["tojson_pretty"] = to_pretty_json
app.jinja_env.globals.update(min=min, max=max)


def count_jsonl_lines(descriptor):
    """Count total lines in JSONL file efficiently"""
    with open_descriptor(descriptor) as f:
        return sum(1 for _ in f)


def load_jsonl_page(file, page=0, per_page=10):
    """Load a specific page of records from JSONL file"""
    records = []
    start_idx = page * per_page
    end_idx = start_idx + per_page

    for i, line in enumerate(file):
        if i >= end_idx:
            break
        if i >= start_idx:
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError:
                logging.warning("Skipping invalid JSONL record at line %s", i)
    return records


def get_shuffled_indices(total_records, page=0, per_page=10, seed=None):
    """Generate shuffled indices for a specific page"""
    import random

    if seed is not None:
        random.seed(seed)

    # Create a list of all indices and shuffle them
    all_indices = list(range(total_records))
    random.shuffle(all_indices)

    # Get the requested page of shuffled indices
    start_idx = page * per_page
    end_idx = start_idx + per_page
    return all_indices[start_idx:end_idx]


def load_records_by_indices(file, indices):
    """Load specific records by their line indices"""
    records = []
    indices_set = set(indices)

    for i, line in enumerate(file):
        if i in indices_set:
            try:
                record = json.loads(line.strip())
                records.append((i, record))  # Keep original index
            except json.JSONDecodeError:
                logging.warning("Skipping invalid JSONL record at line %s", i)

        # Early exit if we've found all records we need
        if len(records) == len(indices):
            break

    # Sort records to match the order of indices
    index_to_record = {idx: record for idx, record in records}
    return [
        (idx, index_to_record.get(idx)) for idx in indices if idx in index_to_record
    ]


def load_single_record(file, record_idx):
    """Load a single record by index from JSONL file"""
    for i, line in enumerate(file):
        if i == record_idx:
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError:
                logging.warning("Invalid JSONL record at line %s", record_idx)
                return None
    return None


@app.route("/")
def index():
    try:
        state_context = open_current_state()
        with state_context as (state, file):
            page = request.args.get("page", 0, type=int)
            shuffle = request.args.get("shuffle", "false").lower() == "true"
            total_pages = math.ceil(state.total_records / records_per_page)

            if shuffle:
                shuffled_indices = get_shuffled_indices(
                    state.total_records,
                    page,
                    records_per_page,
                )
                records = load_records_by_indices(file, shuffled_indices)
            else:
                normal_records = load_jsonl_page(file, page, records_per_page)
                records = [
                    (page * records_per_page + i, record)
                    for i, record in enumerate(normal_records)
                ]
    except RuntimeError:
        return redirect(url_for("file_upload"))
    except (OSError, UnicodeError):
        logging.exception("Failed to load an SFT data page")
        return render_template("upload.html", error="Unable to load file"), 500

    # Process records for display (extract key info)
    processed_records = []
    for original_idx, record in records:
        record_idx = original_idx

        # Extract key metadata
        satisfied_criteria = record.get("satisfied_criteria", False)
        has_satisfied_criteria = (
            len(satisfied_criteria) > 0
            if isinstance(satisfied_criteria, list)
            else bool(satisfied_criteria)
        )
        criteria_count = (
            len(satisfied_criteria)
            if isinstance(satisfied_criteria, list)
            else (1 if satisfied_criteria else 0)
        )

        metadata = {
            "index": record_idx,
            "problem": record.get("problem", "N/A"),
            "run_id": record.get("run_id", "N/A"),
            "satisfied_criteria": satisfied_criteria,
            "has_satisfied_criteria": has_satisfied_criteria,
            "criteria_count": criteria_count,
            "truncated": record.get("truncated", False),
            "tokens": record.get("#tokens", 0),
            "messages_count": len(record.get("messages", [])),
            "tools_count": len(record.get("tools", [])) if record.get("tools") else 0,
        }

        # Extract conversation preview (first few messages)
        messages = record.get("messages", [])
        conversation_preview = []
        for msg in messages[:3]:  # Show first 3 messages as preview
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate content for preview
            if len(content) > 200:
                content = content[:200] + "..."
            conversation_preview.append({"role": role, "content": content})

        processed_records.append(
            {
                "metadata": metadata,
                "conversation_preview": conversation_preview,
                "has_more_messages": len(messages) > 3,
            }
        )

    return render_template(
        "index.html",
        processed_records=processed_records,
        current_page=page,
        total_pages=total_pages,
        total_records=state.total_records,
        current_file=state.filename,
        records_per_page=records_per_page,
        is_shuffled=shuffle,
    )


@app.route("/upload", methods=["GET", "POST"])
def file_upload():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", error="No file selected")

        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", error="No file selected")

        if file:
            descriptor = None
            filepath = None

            try:
                descriptor, filepath, filename = save_uploaded_file(file)
                total_records = count_jsonl_lines(descriptor)
                replace_current_file(
                    descriptor,
                    filepath,
                    filename,
                    total_records,
                    True,
                )
                descriptor = None
                filepath = None
                return redirect(url_for("index"))
            except SafePathError as exc:
                return (
                    render_template(
                        "upload.html",
                        error=_safe_path_message(exc.status_code),
                    ),
                    exc.status_code,
                )
            except (OSError, UnicodeError):
                logging.exception("Failed to load uploaded SFT data")
                return render_template(
                    "upload.html",
                    error="Unable to load file. For large files, use the "
                    "'Load from Server' option instead.",
                )
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                if filepath is not None:
                    filepath.unlink(missing_ok=True)
        else:
            return render_template("upload.html", error="Please upload a JSONL file")

    return render_template("upload.html")


@app.route("/load_file", methods=["POST"])
def load_file():
    """Load a file from the server filesystem"""
    descriptor = None
    try:
        raw_path = request.form.get("filepath", "").strip()
        filepath, _ = _safe_candidate(
            raw_path,
            allowed_suffixes=ALLOWED_SUFFIXES,
        )
        descriptor = open_confined_server_file(raw_path)
        total_records = count_jsonl_lines(descriptor)
        replace_current_file(
            descriptor,
            filepath,
            PurePosixPath(raw_path).name,
            total_records,
            False,
        )
        descriptor = None
        return redirect(url_for("index"))
    except SafePathError as exc:
        return (
            render_template(
                "upload.html",
                error=_safe_path_message(exc.status_code),
            ),
            exc.status_code,
        )
    except (OSError, UnicodeError):
        logging.exception("Failed to load SFT data")
        return render_template("upload.html", error="Unable to load file"), 500
    finally:
        if descriptor is not None:
            os.close(descriptor)


@app.route("/record/<int:record_idx>")
def view_record(record_idx):
    """View a single record in detail"""
    try:
        with open_current_state() as (state, file):
            if record_idx < 0 or record_idx >= state.total_records:
                return jsonify({"error": "Record not found"}), 404
            record = load_single_record(file, record_idx)
    except RuntimeError:
        return redirect(url_for("file_upload"))
    except (OSError, UnicodeError):
        logging.exception("Failed to load an SFT record")
        return jsonify({"error": "Unable to load record"}), 500
    if record is None:
        return jsonify({"error": "Record not found"}), 404

    # Extract metadata
    satisfied_criteria = record.get("satisfied_criteria", [])
    metadata = {
        "index": record_idx,
        "problem": record.get("problem", "N/A"),
        "run_id": record.get("run_id", "N/A"),
        "satisfied_criteria": satisfied_criteria,
        "satisfied_criteria_list": (
            satisfied_criteria if isinstance(satisfied_criteria, list) else []
        ),
        "has_satisfied_criteria": (
            len(satisfied_criteria) > 0
            if isinstance(satisfied_criteria, list)
            else bool(satisfied_criteria)
        ),
        "truncated": record.get("truncated", False),
        "tokens": record.get("#tokens", 0),
        "messages_count": len(record.get("messages", [])),
        "tools_count": len(record.get("tools", [])) if record.get("tools") else 0,
    }

    # Process messages for better display
    messages = record.get("messages", [])
    tools = record.get("tools", [])

    # No need to process tool calls - we'll handle JSON formatting in the frontend

    return render_template(
        "record_detail.html",
        record=record,
        metadata=metadata,
        messages=messages,
        tools=tools,
        record_idx=record_idx,
        total_records=state.total_records,
        current_file=state.filename,
    )


@app.route("/api/record/<int:record_idx>")
def get_record_api(record_idx):
    """API endpoint to get record data as JSON"""
    try:
        with open_current_state() as (state, file):
            if record_idx < 0 or record_idx >= state.total_records:
                return jsonify({"error": "Record not found"}), 404
            record = load_single_record(file, record_idx)
    except RuntimeError:
        return jsonify({"error": "No file loaded"}), 400
    except (OSError, UnicodeError):
        logging.exception("Failed to load an SFT record")
        return jsonify({"error": "Unable to load record"}), 500
    if record is None:
        return jsonify({"error": "Record not found"}), 404

    return jsonify(record)


@app.route("/statistics")
def statistics():
    """Show statistics about the loaded dataset"""
    try:
        state_context = open_current_state()
        with state_context as (state, file):
            use_all_records = state.total_records <= 2000
            if use_all_records:
                sample_records = []
                for line in file:
                    try:
                        record = json.loads(line.strip())
                        sample_records.append(record)
                    except json.JSONDecodeError:
                        continue
                sample_size = len(sample_records)
            else:
                sample_size = 2000
                step = state.total_records / sample_size
                sample_indices = {int(index * step) for index in range(sample_size)}
                sample_records = []
                for index, line in enumerate(file):
                    if index in sample_indices:
                        try:
                            record = json.loads(line.strip())
                            sample_records.append(record)
                        except json.JSONDecodeError:
                            continue
                sample_size = len(sample_records)
    except RuntimeError:
        return redirect(url_for("file_upload"))
    except (OSError, UnicodeError):
        logging.exception("Failed to calculate SFT statistics")
        return render_template("upload.html", error="Unable to load file"), 500

    # Collect statistics
    stats = {
        "total_records": state.total_records,
        "sample_size": sample_size,
        "using_all_records": use_all_records,
        "message_counts": [],
        "token_counts": [],
        "satisfied_criteria_count": 0,
        "truncated_count": 0,
        "role_counts": {},
        "criteria_distribution": {},  # Track different criteria combinations
        "all_criteria": set(),  # Track all unique criteria seen
        "criteria_counts": {},  # Count how often each criterion appears
    }

    for record in sample_records:
        # Message and token counts
        messages = record.get("messages", [])
        stats["message_counts"].append(len(messages))
        stats["token_counts"].append(record.get("#tokens", 0))

        # Criteria and truncation analysis
        satisfied_criteria = record.get("satisfied_criteria", [])
        if isinstance(satisfied_criteria, list):
            # Track criteria combinations
            criteria_key = (
                tuple(sorted(satisfied_criteria))
                if satisfied_criteria
                else ("no_criteria",)
            )
            stats["criteria_distribution"][criteria_key] = (
                stats["criteria_distribution"].get(criteria_key, 0) + 1
            )

            # Track all unique criteria
            stats["all_criteria"].update(satisfied_criteria)

            # Count individual criteria occurrences
            for criterion in satisfied_criteria:
                stats["criteria_counts"][criterion] = (
                    stats["criteria_counts"].get(criterion, 0) + 1
                )

            # Count as successful if there are satisfied criteria (can be refined later)
            if len(satisfied_criteria) > 0:
                stats["satisfied_criteria_count"] += 1
        elif satisfied_criteria:  # Handle boolean case for backward compatibility
            stats["satisfied_criteria_count"] += 1
            stats["criteria_distribution"][("legacy_boolean",)] = (
                stats["criteria_distribution"].get(("legacy_boolean",), 0) + 1
            )
        if record.get("truncated", False):
            stats["truncated_count"] += 1

        # Role statistics
        for msg in messages:
            role = msg.get("role", "unknown")
            stats["role_counts"][role] = stats["role_counts"].get(role, 0) + 1

    # Calculate averages and percentages
    if stats["message_counts"]:
        stats["avg_messages"] = sum(stats["message_counts"]) / len(
            stats["message_counts"]
        )
        stats["max_messages"] = max(stats["message_counts"])
        stats["min_messages"] = min(stats["message_counts"])

    if stats["token_counts"]:
        stats["avg_tokens"] = sum(stats["token_counts"]) / len(stats["token_counts"])
        stats["max_tokens"] = max(stats["token_counts"])
        stats["min_tokens"] = min(stats["token_counts"])

    stats["satisfied_criteria_percent"] = (
        (stats["satisfied_criteria_count"] / sample_size * 100)
        if sample_size > 0
        else 0
    )
    stats["truncated_percent"] = (
        (stats["truncated_count"] / sample_size * 100) if sample_size > 0 else 0
    )

    # Process criteria statistics for template
    stats["all_criteria_list"] = sorted(list(stats["all_criteria"]))
    stats["criteria_combinations"] = []
    for criteria_tuple, count in sorted(
        stats["criteria_distribution"].items(), key=lambda x: x[1], reverse=True
    ):
        criteria_list = list(criteria_tuple)
        percentage = (count / sample_size * 100) if sample_size > 0 else 0
        stats["criteria_combinations"].append(
            {
                "criteria": criteria_list,
                "count": count,
                "percentage": percentage,
                "criteria_text": (
                    ", ".join(criteria_list)
                    if criteria_list != ["no_criteria"]
                    else "No criteria satisfied"
                ),
            }
        )

    # Individual criteria statistics
    stats["individual_criteria"] = []
    for criterion, count in sorted(
        stats["criteria_counts"].items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / sample_size * 100) if sample_size > 0 else 0
        stats["individual_criteria"].append(
            {"name": criterion, "count": count, "percentage": percentage}
        )

    return render_template("statistics.html", stats=stats, current_file=state.filename)


@app.route("/change_file", methods=["POST"])
def change_file():
    clear_current_file()
    return redirect(url_for("file_upload"))


@app.errorhandler(413)
def too_large(e):
    return (
        render_template(
            "upload.html",
            error="File too large! The uploaded file exceeds the maximum size limit. Please use the 'Load from Server' option for large files.",
        ),
        413,
    )


def main():
    parser = argparse.ArgumentParser(description="View SFT JSONL data")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=int(os.environ.get("SFT_DATA_VIEWER_PORT", DEFAULT_PORT)),
    )
    parser.add_argument(
        "--safe-root",
        type=Path,
        default=app.config["SAFE_ROOT"],
        help="Only JSONL files below this directory may be loaded from the server",
    )
    parser.add_argument(
        "--trusted-host",
        action="append",
        help="Exact Host header accepted by the viewer",
    )
    args = parser.parse_args()

    try:
        configure_safe_root(args.safe_root)
    except SafePathError:
        parser.error("safe root is unavailable")
    if args.trusted_host is not None:
        app.config["TRUSTED_HOSTS"] = args.trusted_host
    app.run(host=DEFAULT_HOST, port=args.port)


if __name__ == "__main__":
    main()
