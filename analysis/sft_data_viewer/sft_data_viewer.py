import json
import math
import os

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = (
    5 * 1024 * 1024 * 1024
)  # 5GB max file size for large JSONL files

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global variables to store loaded data info
current_file_path = None
current_file_name = None
total_records = 0
records_per_page = 10


def to_pretty_json(value):
    """Convert Python object to pretty-printed JSON string with minimal whitespace"""
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).strip()


# Add custom filters and globals to Jinja2
app.jinja_env.filters["tojson_pretty"] = to_pretty_json
app.jinja_env.globals.update(min=min, max=max)


def count_jsonl_lines(filepath):
    """Count total lines in JSONL file efficiently"""
    try:
        with open(filepath, "r") as f:
            count = sum(1 for _ in f)
        return count
    except Exception as e:
        print(f"Error counting lines: {e}")
        return 0


def load_jsonl_page(filepath, page=0, per_page=10):
    """Load a specific page of records from JSONL file"""
    records = []
    start_idx = page * per_page
    end_idx = start_idx + per_page

    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if i >= end_idx:
                    break
                if i >= start_idx:
                    try:
                        record = json.loads(line.strip())
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i}: {e}")
                        continue
        return records
    except Exception as e:
        print(f"Error loading page: {e}")
        return []


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


def load_records_by_indices(filepath, indices):
    """Load specific records by their line indices"""
    records = []
    indices_set = set(indices)

    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if i in indices_set:
                    try:
                        record = json.loads(line.strip())
                        records.append((i, record))  # Keep original index
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i}: {e}")
                        continue

                # Early exit if we've found all records we need
                if len(records) == len(indices):
                    break
    except Exception as e:
        print(f"Error loading records by indices: {e}")
        return []

    # Sort records to match the order of indices
    index_to_record = {idx: record for idx, record in records}
    return [
        (idx, index_to_record.get(idx)) for idx in indices if idx in index_to_record
    ]


def load_single_record(filepath, record_idx):
    """Load a single record by index from JSONL file"""
    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if i == record_idx:
                    try:
                        return json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"Error parsing record {record_idx}: {e}")
                        return None
        return None
    except Exception as e:
        print(f"Error loading record: {e}")
        return None


@app.route("/")
def index():
    global current_file_path, current_file_name, total_records

    if current_file_path is None:
        return redirect(url_for("file_upload"))

    # Get pagination and shuffle parameters
    page = request.args.get("page", 0, type=int)
    shuffle = request.args.get("shuffle", "false").lower() == "true"

    # Calculate pagination info
    total_pages = math.ceil(total_records / records_per_page)

    if shuffle:
        # Generate shuffled indices for this page
        shuffled_indices = get_shuffled_indices(total_records, page, records_per_page)
        # Load records by their shuffled indices
        indexed_records = load_records_by_indices(current_file_path, shuffled_indices)
        records = [(idx, record) for idx, record in indexed_records]
    else:
        # Load records normally
        normal_records = load_jsonl_page(current_file_path, page, records_per_page)
        records = [
            (page * records_per_page + i, record)
            for i, record in enumerate(normal_records)
        ]

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
        total_records=total_records,
        current_file=current_file_name,
        records_per_page=records_per_page,
        is_shuffled=shuffle,
    )


@app.route("/upload", methods=["GET", "POST"])
def file_upload():
    global current_file_path, current_file_name, total_records

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("upload.html", error="No file selected")

        file = request.files["file"]
        if file.filename == "":
            return render_template("upload.html", error="No file selected")

        if file and file.filename.endswith(".jsonl"):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            try:
                file.save(filepath)
                # Count total records
                total_records = count_jsonl_lines(filepath)
                current_file_path = filepath
                current_file_name = filename
                return redirect(url_for("index"))
            except Exception as e:
                # Clean up the file if it was partially saved
                if os.path.exists(filepath):
                    os.remove(filepath)
                return render_template(
                    "upload.html",
                    error=f"Error loading file: {str(e)}. File may be too large - try using the 'Load from Server' option instead.",
                )
        else:
            return render_template("upload.html", error="Please upload a JSONL file")

    return render_template("upload.html")


@app.route("/load_file", methods=["POST"])
def load_file():
    """Load a file from the server filesystem"""
    global current_file_path, current_file_name, total_records

    filepath = request.form.get("filepath", "").strip()
    if not filepath:
        return render_template("upload.html", error="Please provide a file path")

    if not os.path.exists(filepath):
        return render_template("upload.html", error=f"File not found: {filepath}")

    if not filepath.endswith(".jsonl"):
        return render_template("upload.html", error="File must be a JSONL file")

    try:
        # Count total records
        total_records = count_jsonl_lines(filepath)
        current_file_path = filepath
        current_file_name = os.path.basename(filepath)
        return redirect(url_for("index"))
    except Exception as e:
        return render_template("upload.html", error=f"Error loading file: {str(e)}")


@app.route("/record/<int:record_idx>")
def view_record(record_idx):
    """View a single record in detail"""
    global current_file_path

    if current_file_path is None:
        return redirect(url_for("file_upload"))

    if record_idx < 0 or record_idx >= total_records:
        return jsonify({"error": "Record not found"}), 404

    record = load_single_record(current_file_path, record_idx)
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
        total_records=total_records,
        current_file=current_file_name,
    )


@app.route("/api/record/<int:record_idx>")
def get_record_api(record_idx):
    """API endpoint to get record data as JSON"""
    global current_file_path

    if current_file_path is None:
        return jsonify({"error": "No file loaded"}), 400

    if record_idx < 0 or record_idx >= total_records:
        return jsonify({"error": "Record not found"}), 404

    record = load_single_record(current_file_path, record_idx)
    if record is None:
        return jsonify({"error": "Record not found"}), 404

    return jsonify(record)


@app.route("/statistics")
def statistics():
    """Show statistics about the loaded dataset"""
    global current_file_path, total_records

    if current_file_path is None:
        return redirect(url_for("file_upload"))

    # Decide whether to sample or use all records based on dataset size
    use_all_records = total_records <= 2000  # Use all records for datasets under 2k

    if use_all_records:
        # Load all records for accurate statistics
        sample_records = []
        with open(current_file_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    sample_records.append(record)
                except json.JSONDecodeError:
                    continue
        sample_size = len(sample_records)
    else:
        # For larger datasets, sample evenly distributed records
        sample_size = 2000  # Sample 2k records for better performance

        # Create evenly distributed sample indices
        step = total_records / sample_size
        sample_indices = [int(i * step) for i in range(sample_size)]

        sample_records = []
        with open(current_file_path, "r") as f:
            for i, line in enumerate(f):
                if i in sample_indices:
                    try:
                        record = json.loads(line.strip())
                        sample_records.append(record)
                    except json.JSONDecodeError:
                        continue
        sample_size = len(sample_records)

    # Collect statistics
    stats = {
        "total_records": total_records,
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

    return render_template(
        "statistics.html", stats=stats, current_file=current_file_name
    )


@app.route("/change_file")
def change_file():
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
