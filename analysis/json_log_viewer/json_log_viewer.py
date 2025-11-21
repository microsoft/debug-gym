import hashlib
import json
import os
import re
import shlex

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Define a safe root directory for browsing
SAFE_ROOT = os.getcwd()

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global variable to store loaded data
data = None
current_file = None


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
    global data
    if data is None:
        return redirect(url_for("file_upload"))

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
    for idx, step in enumerate(data["log"]):
        action = step.get("action")
        action_name = "no_action"
        command_text = None

        if isinstance(action, dict):
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
        if action_name == "bash":
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
        style_entry = {
            "key": key,
            "display_name": key,
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
    display_name_map = {key: key for key in base_keys}
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
        current_file=current_file,
        steps=steps,
        base_action_styles=base_action_styles,
        detailed_action_styles=detailed_action_styles,
        combined_action_styles=combined_action_styles,
    )


@app.route("/upload", methods=["GET", "POST"])
def file_upload():
    global data, current_file

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", error="No file selected")

        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", error="No file selected")

        if file and (
            file.filename.endswith(".json") or file.filename.endswith(".jsonl")
        ):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                current_file = filename
                return redirect(url_for("index"))
            except json.JSONDecodeError:
                return render_template("upload.html", error="Invalid JSON file")
            except Exception as e:
                return render_template(
                    "upload.html", error=f"Error loading file: {str(e)}"
                )
        else:
            return render_template(
                "upload.html", error="Please upload a JSON or JSONL file"
            )

    return render_template("upload.html")


@app.route("/load_from_cwd/<filename>")
def load_from_cwd(filename):
    global data, current_file

    # Sanitize filename to prevent directory traversal
    filename = secure_filename(filename)

    # Check if file exists and has valid extension
    if not (filename.endswith(".json") or filename.endswith(".jsonl")):
        return render_template("upload.html", error="Invalid file type")

    if not os.path.exists(filename):
        return render_template("upload.html", error="File not found")

    try:
        with open(filename, "r") as f:
            data = json.load(f)
        current_file = filename
        return redirect(url_for("index"))
    except json.JSONDecodeError:
        return render_template("upload.html", error="Invalid JSON file")
    except Exception as e:
        return render_template("upload.html", error=f"Error loading file: {str(e)}")


@app.route("/browse_directory")
def browse_directory():
    """Browse directory contents via AJAX"""
    path = request.args.get("path", SAFE_ROOT)

    # Sanitize path to prevent directory traversal attacks
    try:
        path = os.path.abspath(path)
        # Ensure path is within SAFE_ROOT
        if not path.startswith(SAFE_ROOT):
            return (
                jsonify({"error": "Access denied: Path outside allowed directory"}),
                403,
            )
        if not os.path.exists(path) or not os.path.isdir(path):
            return jsonify({"error": "Invalid directory"}), 400
    except (OSError, ValueError):
        return jsonify({"error": "Invalid path"}), 400

    try:
        items = []

        # Add parent directory if not at root
        if path != os.path.dirname(path):  # Not at root
            parent_path = os.path.dirname(path)
            items.append(
                {
                    "name": "..",
                    "path": parent_path,
                    "type": "directory",
                    "is_parent": True,
                }
            )

        # List directory contents
        for item in sorted(os.listdir(path)):
            item_path = os.path.join(path, item)
            try:
                if os.path.isdir(item_path):
                    items.append(
                        {
                            "name": item,
                            "path": item_path,
                            "type": "directory",
                            "is_parent": False,
                        }
                    )
                elif item.endswith((".jsonl")):
                    items.append(
                        {
                            "name": item,
                            "path": item_path,
                            "type": "file",
                            "is_parent": False,
                        }
                    )
            except (OSError, PermissionError):
                # Skip items we can't access
                continue

        return jsonify({"current_path": path, "items": items})

    except (OSError, PermissionError) as e:
        return jsonify({"error": f"Permission denied: {str(e)}"}), 403


@app.route("/load_file_from_path")
def load_file_from_path():
    """Load a JSON file from a specific path"""
    global data, current_file

    filepath = request.args.get("path")
    if not filepath:
        return jsonify({"error": "No file path provided"}), 400

    try:
        filepath = os.path.abspath(filepath)
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            return jsonify({"error": "File not found"}), 404

        if not filepath.endswith(".jsonl"):
            return jsonify({"error": "Invalid file type"}), 400

        with open(filepath, "r") as f:
            data = json.load(f)

        current_file = os.path.basename(filepath)
        return jsonify({"success": True, "redirect": url_for("index")})

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON file"}), 400
    except Exception as e:
        return jsonify({"error": f"Error loading file: {str(e)}"}), 500


@app.route("/get_step/<int:step_id>")
def get_step(step_id):
    global data
    if data is None:
        return jsonify({"error": "No file loaded"}), 400

    # Return the specific step data as JSON
    if 0 <= step_id < len(data["log"]):
        step = data["log"][step_id]
        return jsonify(step)
    return jsonify({"error": "Step not found"}), 404


@app.route("/statistics")
def statistics():
    global data
    if data is None:
        return redirect(url_for("file_upload"))

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
        current_file=current_file,
    )


@app.route("/change_file")
def change_file():
    return redirect(url_for("file_upload"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
