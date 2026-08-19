import argparse
import errno
import hashlib
import json
import logging
import os
import re
import shlex
import stat
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.security import safe_join
from werkzeug.utils import secure_filename

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5050
DEFAULT_SAFE_ROOT = Path.cwd().resolve()
DEFAULT_TRUSTED_HOSTS = ("127.0.0.1", "localhost", "[::1]")
ALLOWED_SUFFIXES = {".json", ".jsonl"}


def _configured_origins() -> tuple[str, ...]:
    value = os.environ.get("JSON_LOG_VIEWER_ALLOWED_ORIGINS", "")
    return tuple(
        origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()
    )


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["SAFE_ROOT"] = (
    Path(os.environ.get("JSON_LOG_VIEWER_SAFE_ROOT", DEFAULT_SAFE_ROOT))
    .expanduser()
    .resolve()
)
app.config["ALLOWED_ORIGINS"] = _configured_origins()
app.config["TRUSTED_HOSTS"] = list(DEFAULT_TRUSTED_HOSTS)

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@dataclass(frozen=True)
class LoadedJsonState:
    data: dict | list
    filename: str


loaded_state: LoadedJsonState | None = None
loaded_state_lock = threading.Lock()


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


def get_loaded_state() -> LoadedJsonState | None:
    with loaded_state_lock:
        return loaded_state


def replace_loaded_state(content: dict | list, filename: str) -> None:
    global loaded_state
    state = LoadedJsonState(data=content, filename=filename)
    with loaded_state_lock:
        loaded_state = state


def clear_loaded_state() -> None:
    global loaded_state
    with loaded_state_lock:
        loaded_state = None


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


def open_confined_directory(raw_path: str) -> tuple[int | None, Path, Path]:
    """Open a directory beneath the pinned safe root for one listing request."""
    candidate, relative = _safe_candidate(raw_path)
    safe_root = Path(app.config["SAFE_ROOT"])
    no_follow = getattr(os, "O_NOFOLLOW", 0)

    if os.open in os.supports_dir_fd and no_follow:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow
        descriptor = None
        completed = False
        try:
            descriptor = os.open(safe_root, directory_flags)
            root_stat = os.fstat(descriptor)
            if (root_stat.st_dev, root_stat.st_ino) != app.config.get(
                "SAFE_ROOT_IDENTITY"
            ):
                raise SafePathError("The configured safe root changed", 409)
            for part in relative.parts:
                next_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
            completed = True
            return descriptor, candidate, relative
        except FileNotFoundError as exc:
            raise SafePathError("File not found", 404) from exc
        except NotADirectoryError as exc:
            raise SafePathError("Invalid directory", 400) from exc
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK}:
                raise SafePathError("Access denied", 403) from exc
            raise
        finally:
            if descriptor is not None and not completed:
                os.close(descriptor)

    current = safe_root
    for part in relative.parts:
        current /= part
        if _is_symbolic_link(current):
            raise SafePathError("Access denied", 403)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(safe_root)
        root_stat = safe_root.stat()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SafePathError("Invalid directory", 400) from exc
    if (root_stat.st_dev, root_stat.st_ino) != app.config.get("SAFE_ROOT_IDENTITY"):
        raise SafePathError("The configured safe root changed", 409)
    if not resolved.is_dir():
        raise SafePathError("Invalid directory", 400)
    return None, resolved, relative


configure_safe_root(app.config["SAFE_ROOT"])


def save_uploaded_json(file_storage) -> tuple[str, dict | list]:
    """Save an upload to an exclusive server-generated path and parse it in-place."""
    filename = secure_filename(file_storage.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise SafePathError("Invalid file type", 400)

    configured_root = Path(app.config["UPLOAD_FOLDER"]).absolute()
    if configured_root.is_symlink():
        raise OSError("Upload folder must not be a symbolic link")
    configured_root.mkdir(parents=True, exist_ok=True)
    upload_root = configured_root.resolve(strict=True)
    filepath = upload_root / f"{uuid.uuid4().hex}{suffix}"
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    descriptor = os.open(filepath, flags, 0o600)
    try:
        with os.fdopen(os.dup(descriptor), "wb") as destination:
            file_storage.save(destination)
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode) or descriptor_stat.st_nlink != 1:
            raise OSError("Upload destination is not a regular file")
        with os.fdopen(os.dup(descriptor), "rb") as source:
            source.seek(0)
            content = source.read(app.config["MAX_CONTENT_LENGTH"] + 1)
        if len(content) > app.config["MAX_CONTENT_LENGTH"]:
            raise OSError("Uploaded file exceeds the size limit")
        return filename, json.loads(content.decode("utf-8"))
    finally:
        os.close(descriptor)
        filepath.unlink(missing_ok=True)


@app.before_request
def reject_untrusted_cross_origin_loads():
    if request.method != "POST":
        return None
    origin = request.headers.get("Origin")
    if not origin:
        return None
    normalized_origin = origin.rstrip("/") if origin else None
    same_origin = request.host_url.rstrip("/")
    if normalized_origin == same_origin:
        return None
    if request.endpoint == "load_file_from_path" and normalized_origin in set(
        app.config.get("ALLOWED_ORIGINS", ())
    ):
        return None
    return jsonify({"error": "Origin not allowed"}), 403


@app.after_request
def add_configured_cors_header(response):
    origin = request.headers.get("Origin")
    if (
        request.endpoint == "load_file_from_path"
        and origin
        and origin.rstrip("/") in set(app.config.get("ALLOWED_ORIGINS", ()))
    ):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "POST"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers.add("Vary", "Origin")
    return response


ACTION_COLOR_PALETTE = [
    "#2b2d42",
    "#43616f",
    "#6c91a1",
    "#99bbad",
    "#c4d6b0",
    "#f3d5a9",
    "#e09f70",
    "#c26d51",
    "#8f4f3f",
    "#663d3c",
    "#3f3a37",
    "#756d54",
    "#a5907e",
    "#d6b69f",
    "#f4c095",
    "#c39a8d",
    "#926c7f",
    "#5f4b66",
    "#3f3351",
    "#2d1e2f",
    "#3a5a78",
    "#567f89",
    "#7aa6a6",
    "#a5c4b8",
    "#d0d8b7",
    "#f6e5b5",
    "#e7b98a",
    "#c48f6a",
    "#a86f5c",
    "#7f4d46",
    "#553a3d",
    "#392e34",
    "#594f4f",
    "#867b6f",
    "#b7a99a",
    "#e1c9b3",
]


def to_pretty_json(value):
    return json.dumps(value, sort_keys=True, indent=4, separators=(",", ": "))


app.jinja_env.filters["tojson_pretty"] = to_pretty_json


def sanitize_action_name(name: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", name.lower())
    sanitized = sanitized.strip("-")
    return sanitized or "action"


def make_unique_class(name: str, used_classes):
    base_class = f"action-{sanitize_action_name(name)}"
    candidate = base_class
    counter = 2
    while candidate in used_classes:
        candidate = f"{base_class}-{counter}"
        counter += 1
    used_classes.add(candidate)
    return candidate


def pick_text_color(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#ffffff"
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "#000000" if brightness > 155 else "#ffffff"


def hsl_to_hex(h: float, s: float, l: float) -> str:
    h = h % 1.0

    def hue_to_rgb(p, q, t):
        t = t % 1.0
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1 / 3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1 / 3)

    def clamp_rgb(value: float) -> int:
        return max(0, min(255, int(round(value * 255))))

    return "#{:02x}{:02x}{:02x}".format(clamp_rgb(r), clamp_rgb(g), clamp_rgb(b))


def color_for_index(index: int) -> str:
    if index < len(ACTION_COLOR_PALETTE):
        return ACTION_COLOR_PALETTE[index]

    generated_index = index - len(ACTION_COLOR_PALETTE)

    hue = (generated_index * 0.29 + 0.1) % 1.0
    saturation_options = [0.58, 0.46, 0.36]
    lightness_options = [0.42, 0.56, 0.68]

    saturation = saturation_options[generated_index % len(saturation_options)]
    lightness = lightness_options[
        (generated_index // len(saturation_options)) % len(lightness_options)
    ]

    return hsl_to_hex(hue, saturation, lightness)


def color_for_key(key: str, fallback_index: int) -> str:
    if not key:
        return color_for_index(fallback_index)

    normalized = key.lower().strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()

    hue_raw = int.from_bytes(digest[0:2], "big") / 65535.0
    hue = (hue_raw + fallback_index * 0.19) % 1.0

    saturation_options = [0.55, 0.42, 0.48, 0.35]
    lightness_options = [0.40, 0.55, 0.68, 0.48]

    saturation = saturation_options[digest[2] % len(saturation_options)]
    lightness = lightness_options[digest[3] % len(lightness_options)]

    return hsl_to_hex(hue, saturation, lightness)


def unique_preserve_order(values):
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def classify_bash_command(command: str) -> str | None:
    if not command:
        return None

    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        tokens = command.strip().split()

    if not tokens:
        return None

    connectors = {
        "&&",
        "||",
        "|",
        ";",
        "then",
        "do",
        "fi",
        "elif",
        "else",
        "in",
        "(",
        ")",
        "{",
        "}",
    }
    skip_tokens = {"sudo", "env", "bash", "sh"}

    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1

        if not token:
            continue

        token_lower = token.lower()

        if token_lower in connectors:
            continue

        if token_lower in skip_tokens:
            continue

        if token.startswith("-"):
            continue

        if "=" in token and not token.startswith("--"):
            continue

        cleaned = os.path.basename(token)
        if not cleaned:
            continue

        cleaned_lower = cleaned.lower()
        if cleaned_lower in skip_tokens or cleaned_lower in connectors:
            continue

        if cleaned_lower in {"python", "python3", "python2"}:
            if index < len(tokens) and tokens[index] == "-m":
                module_index = index + 1
                if module_index < len(tokens):
                    module = os.path.basename(tokens[module_index])
                    if module:
                        return module.lower()
            return cleaned_lower

        if " " in cleaned:
            nested_classification = classify_bash_command(cleaned)
            if nested_classification:
                return nested_classification
            return cleaned.split()[0].lower()

        return cleaned_lower

    return None


def get_bash_detail(command: str) -> tuple[str, str]:
    classification = classify_bash_command(command)

    if classification:
        return (f"bash::{classification}", f"bash: {classification}")

    if command is None:
        return ("bash::no-command", "bash: (no command)")

    stripped = command.strip()
    if not stripped:
        return ("bash::empty", "bash: (empty command)")

    return ("bash::other", "bash: other")


@app.route("/")
def index():
    state = get_loaded_state()
    if state is None:
        return redirect(url_for("file_upload"))
    data = state.data

    # Pass metadata to the template
    metadata = {
        "problem": data["problem"],
        "config": data["config"],
        "uuid": data["uuid"],
        "success": data["success"],
    }
    total_steps = len(data["log"])

    # Extract action types per step, capturing bash command details when available
    step_actions = []
    step_entries = []
    bash_detail_entries = []
    display_name_overrides = {
        "no_action": "No Action",
        "unknown": "Unknown Action",
    }
    for idx, step in enumerate(data["log"]):
        action = step.get("action")
        action_name = "no_action"
        command_text = None

        is_initial_step = idx == 0 and (
            step.get("system_message") is not None
            or step.get("problem_message") is not None
        )

        if is_initial_step:
            action_name = "initial_state"
            display_name_overrides[action_name] = "Initial State"
        elif isinstance(action, dict):
            action_name = action.get("name") or "unknown"
            arguments = action.get("arguments")
            if isinstance(arguments, dict) and "command" in arguments:
                raw_command = arguments.get("command")
                if raw_command is not None:
                    command_text = str(raw_command)
        elif action is None:
            action_name = "no_action"
        else:
            action_name = str(action)

        step_actions.append(action_name)

        bash_key = None
        if action_name == "bash" and command_text is not None:
            bash_key, bash_label = get_bash_detail(command_text)
            bash_detail_entries.append((bash_key, bash_label))

        step_entries.append(
            {
                "index": idx,
                "action_name": action_name,
                "bash_key": bash_key,
            }
        )

    # Gather declared tools (if present) for consistent styling
    declared_tool_names = []
    for tool in data.get("tools", []):
        tool_name = None
        if isinstance(tool, dict):
            if tool.get("type") == "function":
                tool_name = tool.get("function", {}).get("name")
            else:
                tool_name = tool.get("name")
        elif isinstance(tool, str):
            tool_name = tool
        if tool_name:
            declared_tool_names.append(tool_name)

    declared_tool_names = unique_preserve_order(declared_tool_names)
    base_keys = unique_preserve_order(declared_tool_names + step_actions)

    used_classes = set()
    base_action_styles = []
    base_action_style_map = {}
    for idx, key in enumerate(base_keys):
        palette_color = color_for_key(key, idx)
        css_class = make_unique_class(key, used_classes)
        display_name = display_name_overrides.get(key, key)
        style_entry = {
            "key": key,
            "display_name": display_name,
            "class": css_class,
            "background": palette_color,
            "text_color": pick_text_color(palette_color),
        }
        base_action_styles.append(style_entry)
        base_action_style_map[key] = style_entry

    bash_detail_map = {}
    for key, display_name in bash_detail_entries:
        if key not in bash_detail_map:
            bash_detail_map[key] = display_name

    detailed_keys = unique_preserve_order(base_keys + list(bash_detail_map.keys()))
    display_name_map = {key: display_name_overrides.get(key, key) for key in base_keys}
    display_name_map.update(bash_detail_map)

    detailed_action_style_map = {
        key: dict(style_entry) for key, style_entry in base_action_style_map.items()
    }
    used_detailed_classes = set(
        style_entry["class"] for style_entry in base_action_styles
    )
    color_index = len(base_action_styles)

    for key in detailed_keys:
        if key in detailed_action_style_map:
            continue
        palette_color = color_for_key(key, color_index)
        css_class = make_unique_class(key, used_detailed_classes)
        style_entry = {
            "key": key,
            "display_name": display_name_map.get(key, key),
            "class": css_class,
            "background": palette_color,
            "text_color": pick_text_color(palette_color),
        }
        detailed_action_style_map[key] = style_entry
        color_index += 1

    detailed_action_styles = []
    seen_keys = set()
    for key in detailed_keys:
        if key in seen_keys:
            continue
        style_entry = detailed_action_style_map.get(key)
        if style_entry:
            detailed_action_styles.append(style_entry)
            seen_keys.add(key)

    combined_action_styles = []
    seen_classes = set()
    for style_entry in base_action_styles + detailed_action_styles:
        css_class = style_entry["class"]
        if css_class in seen_classes:
            continue
        combined_action_styles.append(style_entry)
        seen_classes.add(css_class)

    steps = []
    for entry in step_entries:
        base_style = base_action_style_map.get(entry["action_name"])
        detailed_key = entry["bash_key"] or entry["action_name"]
        detailed_style = detailed_action_style_map.get(detailed_key, base_style)

        base_class = base_style["class"] if base_style else "action-unknown"
        detailed_class = detailed_style["class"] if detailed_style else base_class

        steps.append(
            {
                "index": entry["index"],
                "base_class": base_class,
                "detailed_class": detailed_class,
                "base_label": (
                    base_style["display_name"] if base_style else entry["action_name"]
                ),
                "detailed_label": (
                    detailed_style["display_name"]
                    if detailed_style
                    else entry["action_name"]
                ),
            }
        )

    return render_template(
        "index.html",
        metadata=metadata,
        total_steps=total_steps,
        current_file=state.filename,
        steps=steps,
        base_action_styles=base_action_styles,
        detailed_action_styles=detailed_action_styles,
        combined_action_styles=combined_action_styles,
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
            try:
                filename, content = save_uploaded_json(file)
                replace_loaded_state(content, filename)
                return redirect(url_for("index"))
            except SafePathError as exc:
                return (
                    render_template(
                        "upload.html",
                        error=_safe_path_message(exc.status_code),
                    ),
                    exc.status_code,
                )
            except json.JSONDecodeError:
                return render_template("upload.html", error="Invalid JSON file")
            except (OSError, UnicodeError):
                logging.exception("Failed to load uploaded trajectory")
                return render_template("upload.html", error="Unable to load file")
        else:
            return render_template(
                "upload.html", error="Please upload a JSON or JSONL file"
            )

    return render_template("upload.html")


@app.route("/load_from_cwd/<filename>", methods=["POST"])
def load_from_cwd(filename):
    filename = secure_filename(filename)
    descriptor = None
    try:
        descriptor = open_confined_server_file(filename)
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as source:
            content = json.load(source)
        replace_loaded_state(content, PurePosixPath(filename).name)
        return redirect(url_for("index"))
    except SafePathError as exc:
        return (
            render_template(
                "upload.html",
                error=_safe_path_message(exc.status_code),
            ),
            exc.status_code,
        )
    except json.JSONDecodeError:
        return render_template("upload.html", error="Invalid JSON file")
    except (OSError, UnicodeError):
        logging.exception("Failed to load trajectory")
        return render_template("upload.html", error="Unable to load file"), 500
    finally:
        if descriptor is not None:
            os.close(descriptor)


@app.route("/browse_directory")
def browse_directory():
    """Browse directory contents via AJAX"""
    directory_descriptor = None
    try:
        directory_descriptor, path, relative_path = open_confined_directory(
            request.args.get("path", ".")
        )
        current_path = relative_path.as_posix() if relative_path.parts else "."
        items = []

        # Add parent directory if not at root
        if relative_path.parts:
            parent_relative = relative_path.parent
            items.append(
                {
                    "name": "..",
                    "path": (
                        parent_relative.as_posix() if parent_relative.parts else "."
                    ),
                    "type": "directory",
                    "is_parent": True,
                }
            )

        # List directory contents
        scan_target = directory_descriptor if directory_descriptor is not None else path
        with os.scandir(scan_target) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                try:
                    if "\\" in entry.name or entry.is_symlink():
                        continue
                    item_relative_path = relative_path / entry.name
                    item_relative = item_relative_path.as_posix()
                    if entry.is_dir(follow_symlinks=False):
                        items.append(
                            {
                                "name": entry.name,
                                "path": item_relative,
                                "type": "directory",
                                "is_parent": False,
                            }
                        )
                    elif (
                        entry.is_file(follow_symlinks=False)
                        and PurePosixPath(entry.name).suffix.lower() in ALLOWED_SUFFIXES
                    ):
                        items.append(
                            {
                                "name": entry.name,
                                "path": item_relative,
                                "type": "file",
                                "is_parent": False,
                            }
                        )
                except OSError:
                    continue

        return jsonify({"current_path": current_path, "items": items})
    except SafePathError as exc:
        return jsonify({"error": _safe_path_message(exc.status_code)}), exc.status_code
    except (OSError, PermissionError):
        logging.exception("Error while browsing directory")
        return jsonify({"error": "Permission denied"}), 403
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)


@app.route("/load_file_from_path", methods=["POST"])
def load_file_from_path():
    """Load a JSON file from a specific path"""
    descriptor = None
    try:
        request_data = request.get_json(silent=True) or request.form
        raw_path = request_data.get("path", "")
        descriptor = open_confined_server_file(raw_path)
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as source:
            content = json.load(source)
        replace_loaded_state(content, PurePosixPath(raw_path).name)
        return jsonify({"success": True, "redirect": url_for("index")})

    except SafePathError as exc:
        return jsonify({"error": _safe_path_message(exc.status_code)}), exc.status_code
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON file"}), 400
    except (OSError, UnicodeError):
        logging.exception("Failed to load trajectory")
        return jsonify({"error": "Unable to load file"}), 500
    finally:
        if descriptor is not None:
            os.close(descriptor)


@app.route("/get_step/<int:step_id>")
def get_step(step_id):
    state = get_loaded_state()
    if state is None:
        return jsonify({"error": "No file loaded"}), 400
    data = state.data

    # Return the specific step data as JSON
    if 0 <= step_id < len(data["log"]):
        step = data["log"][step_id]
        return jsonify(step)
    return jsonify({"error": "Step not found"}), 404


@app.route("/statistics")
def statistics():
    state = get_loaded_state()
    if state is None:
        return redirect(url_for("file_upload"))
    data = state.data

    # Collect action statistics
    action_counts = {}
    total_actions = 0

    for step in data["log"]:
        if step.get("action") and step["action"] is not None:
            action_name = step["action"].get("name", "unknown")
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            total_actions += 1

    # Calculate percentages and sort by count
    statistics_data = []
    for action_name, count in sorted(
        action_counts.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        statistics_data.append(
            {"name": action_name, "count": count, "percentage": round(percentage, 1)}
        )

    # Pass metadata to template
    metadata = {
        "problem": data["problem"],
        "config": data["config"],
        "uuid": data["uuid"],
        "success": data["success"],
    }

    return render_template(
        "statistics.html",
        metadata=metadata,
        statistics_data=statistics_data,
        total_actions=total_actions,
        total_steps=len(data["log"]),
        current_file=state.filename,
    )


@app.route("/change_file", methods=["POST"])
def change_file():
    clear_loaded_state()
    return redirect(url_for("file_upload"))


def main():
    parser = argparse.ArgumentParser(description="View debug-gym trajectory logs")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=int(os.environ.get("JSON_LOG_VIEWER_PORT", DEFAULT_PORT)),
    )
    parser.add_argument(
        "--safe-root",
        type=Path,
        default=app.config["SAFE_ROOT"],
        help="Only files below this directory may be browsed or loaded",
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        help="Exact origin allowed to use the cross-origin load integration",
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
    if args.allowed_origin is not None:
        app.config["ALLOWED_ORIGINS"] = tuple(
            origin.rstrip("/") for origin in args.allowed_origin
        )
    if args.trusted_host is not None:
        app.config["TRUSTED_HOSTS"] = args.trusted_host
    app.run(host=DEFAULT_HOST, port=args.port)


if __name__ == "__main__":
    main()
