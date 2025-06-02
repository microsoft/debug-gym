import json
import os

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global variable to store loaded data
data = None
current_file = None


def to_pretty_json(value):
    return json.dumps(value, sort_keys=True, indent=4, separators=(",", ": "))


app.jinja_env.filters["tojson_pretty"] = to_pretty_json


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

    # Extract action types for each step for color-coding
    step_actions = []
    for step in data["log"]:
        if step.get("action") and step["action"] is not None:
            action_name = step["action"].get("name", "unknown")
        else:
            action_name = "no_action"
        step_actions.append(action_name)

    return render_template(
        "index.html",
        metadata=metadata,
        total_steps=total_steps,
        current_file=current_file,
        step_actions=step_actions,
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
