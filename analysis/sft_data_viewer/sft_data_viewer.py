import json
import math
import os
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 * 1024  # 5GB max file size for large JSONL files
# Configure allowed directories for security - prevents path traversal attacks
# Add any directories you want to allow file access to
app.config["ALLOWED_DIRECTORIES"] = [
    os.path.abspath(os.path.expanduser("~/data")),  # User's data directory
    os.path.abspath("data"),  # Project data directory
    os.path.abspath("uploads"),  # Upload directory
    # Add more directories as needed, e.g.:
    # os.path.abspath("/path/to/your/datasets"),
]

# Create uploads directory if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global variables to store loaded data info
current_file_path = None
current_file_name = None
total_records = 0
records_per_page = 10


def is_safe_path(filepath):
    """
    Validate that the file path is safe and within allowed directories.
    Prevents path traversal attacks and restricts access to authorized directories.
    
    Args:
        filepath (str): The file path to validate
        
    Returns:
        bool: True if the path is safe, False otherwise
    """
    try:
        # Basic input validation
        if not filepath or not isinstance(filepath, str):
            return False
        
        # First, normalize the path but don't do any file system operations yet
        try:
            resolved_path = os.path.abspath(os.path.expanduser(filepath))
        except (OSError, ValueError):
            return False
        
        # Check if file has .jsonl extension (case insensitive) BEFORE any file operations
        if not resolved_path.lower().endswith('.jsonl'):
            return False
        
        # Check if the path is within any of the allowed directories FIRST
        # This prevents any file system operations on unauthorized paths
        is_within_allowed = False
        for allowed_dir in app.config["ALLOWED_DIRECTORIES"]:
            try:
                # Use pathlib for robust path comparison
                allowed_path = Path(allowed_dir).resolve()
                candidate_path = Path(resolved_path)  # Don't resolve yet to avoid file system access
                
                # Try to resolve only after checking it's theoretically within bounds
                try:
                    candidate_resolved = candidate_path.resolve()
                    # Check if the file is within the allowed directory (including subdirectories)
                    candidate_resolved.relative_to(allowed_path)
                    is_within_allowed = True
                    resolved_path = str(candidate_resolved)  # Use the safely resolved path
                    break
                except ValueError:
                    # File is not within this allowed directory, continue checking others
                    continue
                except OSError:
                    # Path resolution failed, continue to next directory
                    continue
            except (OSError, ValueError):
                # Handle cases where allowed directory path cannot be resolved
                continue
        
        if not is_within_allowed:
            return False
        
        # Now that we've validated the path is within allowed directories,
        # we can safely check if file exists and is a file (not a directory or symlink)
        try:
            if not os.path.isfile(resolved_path) or os.path.islink(resolved_path):
                return False
        except (OSError, ValueError):
            return False
        
        return True
    except (OSError, ValueError, TypeError):
        return False


def safe_remove_file(filepath):
    """
    Safely remove a file, ensuring it's within our allowed directories.
    Used for cleanup purposes.
    
    Args:
        filepath (str): The file path to remove
        
    Returns:
        bool: True if file was removed or doesn't exist, False if removal failed
    """
    try:
        # Only proceed if it's a valid path string
        if not filepath or not isinstance(filepath, str):
            return False
        
        # Get the upload directory first
        upload_dir = os.path.abspath(app.config["UPLOAD_FOLDER"])
        
        # Resolve the absolute path but validate first
        try:
            resolved_path = os.path.abspath(filepath)
        except (OSError, ValueError):
            return False
        
        # Check if the file is within the upload directory BEFORE any file operations
        try:
            Path(resolved_path).relative_to(Path(upload_dir))
        except ValueError:
            # File is not within upload directory, don't remove
            return False
        except (OSError, ValueError):
            # Path validation failed
            return False
        
        # Now that we've validated the path is safe, check if file exists and remove it
        try:
            if os.path.isfile(resolved_path):
                os.remove(resolved_path)
        except (OSError, ValueError):
            return False
        
        return True
    except (OSError, ValueError, TypeError):
        return False


def to_pretty_json(value):
    """Convert Python object to pretty-printed JSON string with minimal whitespace"""
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).strip()


# Add custom filters and globals to Jinja2
app.jinja_env.filters["tojson_pretty"] = to_pretty_json
app.jinja_env.globals.update(min=min, max=max)


def count_jsonl_lines(filepath):
    """Count total lines in JSONL file efficiently"""
    # Validate file path for security
    if not is_safe_path(filepath):
        print(f"Security error: Access denied to file path: {filepath}")
        return 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        return count
    except Exception as e:
        print(f"Error counting lines: {e}")
        return 0


def load_jsonl_page(filepath, page=0, per_page=10):
    """Load a specific page of records from JSONL file"""
    # Validate file path for security
    if not is_safe_path(filepath):
        print(f"Security error: Access denied to file path: {filepath}")
        return []
    
    records = []
    start_idx = page * per_page
    end_idx = start_idx + per_page
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
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


def load_single_record(filepath, record_idx):
    """Load a single record by index from JSONL file"""
    # Validate file path for security
    if not is_safe_path(filepath):
        print(f"Security error: Access denied to file path: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
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
    
    if current_file_path is None or not is_safe_path(current_file_path):
        # Reset invalid file path
        current_file_path = None
        current_file_name = None
        total_records = 0
        return redirect(url_for("file_upload"))
    
    # Get pagination parameters
    page = request.args.get('page', 0, type=int)
    
    # Validate page number
    if page < 0:
        page = 0
    
    # Load records for current page
    records = load_jsonl_page(current_file_path, page, records_per_page)
    
    # Calculate pagination info
    total_pages = math.ceil(total_records / records_per_page)
    
    # Process records for display (extract key info)
    processed_records = []
    for i, record in enumerate(records):
        record_idx = page * records_per_page + i
        
        # Extract key metadata
        metadata = {
            'index': record_idx,
            'problem': record.get('problem', 'N/A'),
            'run_id': record.get('run_id', 'N/A'),
            'satisfied_criteria': record.get('satisfied_criteria', False),
            'truncated': record.get('truncated', False),
            'tokens': record.get('#tokens', 0),
            'messages_count': len(record.get('messages', [])),
            'tools_count': len(record.get('tools', [])) if record.get('tools') else 0,
        }
        
        # Extract conversation preview (first few messages)
        messages = record.get('messages', [])
        conversation_preview = []
        for msg in messages[:3]:  # Show first 3 messages as preview
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            # Truncate content for preview
            if len(content) > 200:
                content = content[:200] + "..."
            conversation_preview.append({'role': role, 'content': content})
        
        processed_records.append({
            'metadata': metadata,
            'conversation_preview': conversation_preview,
            'has_more_messages': len(messages) > 3
        })
    
    return render_template(
        "index.html",
        processed_records=processed_records,
        current_page=page,
        total_pages=total_pages,
        total_records=total_records,
        current_file=current_file_name,
        records_per_page=records_per_page
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
            # Additional validation: ensure filename is not empty after securing
            if not filename or not filename.endswith(".jsonl"):
                return render_template("upload.html", error="Invalid filename. Please use a valid .jsonl file.")
            
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            
            try:
                file.save(filepath)
                # Validate the saved file path
                if not is_safe_path(filepath):
                    # Clean up the file if validation fails
                    safe_remove_file(filepath)
                    return render_template("upload.html", error="Security validation failed for uploaded file.")
                
                # Count total records
                total_records = count_jsonl_lines(filepath)
                if total_records == 0:
                    # Clean up the file if it's empty or unreadable
                    safe_remove_file(filepath)
                    return render_template("upload.html", error="Uploaded file appears to be empty or could not be read.")
                
                current_file_path = os.path.abspath(filepath)  # Store the absolute path
                current_file_name = filename
                return redirect(url_for("index"))
            except Exception as e:
                # Clean up the file if it was partially saved
                safe_remove_file(filepath)
                return render_template(
                    "upload.html", error=f"Error loading file: {str(e)}. File may be too large - try using the 'Load from Server' option instead."
                )
        else:
            return render_template(
                "upload.html", error="Please upload a JSONL file"
            )

    return render_template("upload.html")


@app.route("/load_file", methods=["POST"])
def load_file():
    """Load a file from the server filesystem"""
    global current_file_path, current_file_name, total_records
    
    filepath = request.form.get("filepath", "").strip()
    if not filepath:
        return render_template("upload.html", error="Please provide a file path")
    
    # Validate the file path for security
    if not is_safe_path(filepath):
        allowed_dirs = ", ".join(app.config["ALLOWED_DIRECTORIES"])
        return render_template("upload.html", 
                             error=f"Access denied. File must be a .jsonl file within allowed directories: {allowed_dirs}")
    
    try:
        # Count total records
        total_records = count_jsonl_lines(filepath)
        if total_records == 0:
            return render_template("upload.html", error="File appears to be empty or could not be read")
        
        current_file_path = os.path.abspath(filepath)  # Store the absolute path
        current_file_name = os.path.basename(os.path.abspath(filepath))  # Safe basename extraction
        return redirect(url_for("index"))
    except Exception as e:
        return render_template("upload.html", error=f"Error loading file: {str(e)}")


@app.route("/record/<int:record_idx>")
def view_record(record_idx):
    """View a single record in detail"""
    global current_file_path
    
    if current_file_path is None or not is_safe_path(current_file_path):
        # Reset invalid file path
        current_file_path = None
        return redirect(url_for("file_upload"))
    
    # Validate record index
    if not isinstance(record_idx, int) or record_idx < 0 or record_idx >= total_records:
        return jsonify({"error": "Invalid record index"}), 404
    
    record = load_single_record(current_file_path, record_idx)
    if record is None:
        return jsonify({"error": "Record not found"}), 404
    
    # Extract metadata
    satisfied_criteria = record.get('satisfied_criteria', [])
    metadata = {
        'index': record_idx,
        'problem': record.get('problem', 'N/A'),
        'run_id': record.get('run_id', 'N/A'),
        'satisfied_criteria': satisfied_criteria,
        'satisfied_criteria_list': satisfied_criteria if isinstance(satisfied_criteria, list) else [],
        'has_satisfied_criteria': len(satisfied_criteria) > 0 if isinstance(satisfied_criteria, list) else bool(satisfied_criteria),
        'truncated': record.get('truncated', False),
        'tokens': record.get('#tokens', 0),
        'messages_count': len(record.get('messages', [])),
        'tools_count': len(record.get('tools', [])) if record.get('tools') else 0,
    }
    
    # Process messages for better display
    messages = record.get('messages', [])
    tools = record.get('tools', [])
    
    # No need to process tool calls - we'll handle JSON formatting in the frontend
    
    return render_template(
        "record_detail.html",
        record=record,
        metadata=metadata,
        messages=messages,
        tools=tools,
        record_idx=record_idx,
        total_records=total_records,
        current_file=current_file_name
    )


@app.route("/api/record/<int:record_idx>")
def get_record_api(record_idx):
    """API endpoint to get record data as JSON"""
    global current_file_path
    
    if current_file_path is None or not is_safe_path(current_file_path):
        # Reset invalid file path
        current_file_path = None
        return jsonify({"error": "No file loaded"}), 400
    
    # Validate record index
    if not isinstance(record_idx, int) or record_idx < 0 or record_idx >= total_records:
        return jsonify({"error": "Invalid record index"}), 404
    
    record = load_single_record(current_file_path, record_idx)
    if record is None:
        return jsonify({"error": "Record not found"}), 404
    
    return jsonify(record)


@app.route("/statistics")
def statistics():
    """Show statistics about the loaded dataset"""
    global current_file_path, total_records
    
    if current_file_path is None or not is_safe_path(current_file_path):
        # Reset invalid file path
        current_file_path = None
        return redirect(url_for("file_upload"))
    
    # Sample some records to gather statistics
    sample_size = min(100, total_records)
    sample_records = load_jsonl_page(current_file_path, 0, sample_size)
    
    # Collect statistics
    stats = {
        'total_records': total_records,
        'sample_size': len(sample_records),
        'problems': {},
        'message_counts': [],
        'token_counts': [],
        'satisfied_criteria_count': 0,
        'truncated_count': 0,
        'role_counts': {},
        'criteria_distribution': {},  # Track different criteria combinations
        'all_criteria': set(),  # Track all unique criteria seen
        'criteria_counts': {},  # Count how often each criterion appears
    }
    
    for record in sample_records:
        # Problem statistics
        problem = record.get('problem', 'N/A')
        stats['problems'][problem] = stats['problems'].get(problem, 0) + 1
        
        # Message and token counts
        messages = record.get('messages', [])
        stats['message_counts'].append(len(messages))
        stats['token_counts'].append(record.get('#tokens', 0))
        
        # Criteria and truncation analysis
        satisfied_criteria = record.get('satisfied_criteria', [])
        if isinstance(satisfied_criteria, list):
            # Track criteria combinations
            criteria_key = tuple(sorted(satisfied_criteria)) if satisfied_criteria else ('no_criteria',)
            stats['criteria_distribution'][criteria_key] = stats['criteria_distribution'].get(criteria_key, 0) + 1
            
            # Track all unique criteria
            stats['all_criteria'].update(satisfied_criteria)
            
            # Count individual criteria occurrences
            for criterion in satisfied_criteria:
                stats['criteria_counts'][criterion] = stats['criteria_counts'].get(criterion, 0) + 1
            
            # Count as successful if there are satisfied criteria (can be refined later)
            if len(satisfied_criteria) > 0:
                stats['satisfied_criteria_count'] += 1
        elif satisfied_criteria:  # Handle boolean case for backward compatibility
            stats['satisfied_criteria_count'] += 1
            stats['criteria_distribution'][('legacy_boolean',)] = stats['criteria_distribution'].get(('legacy_boolean',), 0) + 1
        if record.get('truncated', False):
            stats['truncated_count'] += 1
        
        # Role statistics
        for msg in messages:
            role = msg.get('role', 'unknown')
            stats['role_counts'][role] = stats['role_counts'].get(role, 0) + 1
    
    # Calculate averages and percentages
    if stats['message_counts']:
        stats['avg_messages'] = sum(stats['message_counts']) / len(stats['message_counts'])
        stats['max_messages'] = max(stats['message_counts'])
        stats['min_messages'] = min(stats['message_counts'])
    
    if stats['token_counts']:
        stats['avg_tokens'] = sum(stats['token_counts']) / len(stats['token_counts'])
        stats['max_tokens'] = max(stats['token_counts'])
        stats['min_tokens'] = min(stats['token_counts'])
    
    stats['satisfied_criteria_percent'] = (stats['satisfied_criteria_count'] / sample_size * 100) if sample_size > 0 else 0
    stats['truncated_percent'] = (stats['truncated_count'] / sample_size * 100) if sample_size > 0 else 0
    
    # Sort problems by count for easier template rendering
    stats['top_problems'] = sorted(stats['problems'].items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Process criteria statistics for template
    stats['all_criteria_list'] = sorted(list(stats['all_criteria']))
    stats['criteria_combinations'] = []
    for criteria_tuple, count in sorted(stats['criteria_distribution'].items(), key=lambda x: x[1], reverse=True):
        criteria_list = list(criteria_tuple)
        percentage = (count / sample_size * 100) if sample_size > 0 else 0
        stats['criteria_combinations'].append({
            'criteria': criteria_list,
            'count': count,
            'percentage': percentage,
            'criteria_text': ', '.join(criteria_list) if criteria_list != ['no_criteria'] else 'No criteria satisfied'
        })
    
    # Individual criteria statistics
    stats['individual_criteria'] = []
    for criterion, count in sorted(stats['criteria_counts'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / sample_size * 100) if sample_size > 0 else 0
        stats['individual_criteria'].append({
            'name': criterion,
            'count': count,
            'percentage': percentage
        })
    
    return render_template(
        "statistics.html",
        stats=stats,
        current_file=current_file_name
    )


@app.route("/change_file")
def change_file():
    return redirect(url_for("file_upload"))


@app.errorhandler(413)
def too_large(e):
    return render_template("upload.html", error="File too large! The uploaded file exceeds the maximum size limit. Please use the 'Load from Server' option for large files."), 413


# Security headers middleware
@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    # Only serve over HTTPS in production
    # response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response


if __name__ == "__main__":
    # SECURITY NOTE: For production deployment, consider adding:
    # - CSRF protection (Flask-WTF)
    # - Session security (secure cookies, session timeout)
    # - Rate limiting (Flask-Limiter)
    # - HTTPS enforcement
    # - Authentication/authorization if needed
    app.run(host="0.0.0.0", port=5001)
