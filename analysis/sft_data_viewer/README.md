# SFT Data Viewer

A web-based viewer for Supervised Fine-Tuning (SFT) data stored in JSONL format. This viewer is designed to handle large JSONL files efficiently by using pagination and streaming techniques.

## Features

- **Large File Support**: Handles large JSONL files without loading everything into memory
- **Pagination**: Browse through records with configurable page sizes
- **Detailed View**: View individual records with full conversation details
- **Statistics Dashboard**: Analyze dataset statistics including message counts, token usage, and success rates
- **File Upload**: Upload files via web interface or load directly from server filesystem
- **Conversation Display**: Clean, color-coded display of conversation messages with different roles

## Data Format

The viewer expects JSONL files where each line contains a JSON object with the following structure:

```json
{
  "messages": [
    {
      "role": "system|user|assistant|tool",
      "content": "message content",
      "tool_calls": [...],  // optional
      "tool_call_id": "...", // optional
      "name": "..."          // optional
    }
  ],
  "tools": [...],              // optional array of available tools
  "problem": "problem description",
  "run_id": "unique_identifier",
  "satisfied_criteria": true|false,
  "truncated": true|false,
  "#tokens": 1234
}
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the viewer:
```bash
python sft_data_viewer.py
```

3. Open your browser and go to: `http://localhost:5001`

## Usage

### Loading Files

You have two options to load JSONL files:

1. **Upload via Web Interface**: Use the file upload form on the main page
2. **Load from Server**: Enter the full file path to load files directly from the server

### Navigation

- **Main Page**: Shows paginated list of records with key metadata and conversation previews
- **Record Detail**: Click on any record to view the full conversation and metadata
- **Statistics**: View dataset statistics including role distribution, success rates, and token usage

### Features

- **Pagination**: Navigate through large datasets efficiently
- **Search**: Browse by record index or use pagination controls
- **Raw JSON**: View the raw JSON data for any record
- **Tools Display**: See available tools for each conversation
- **Message Roles**: Color-coded display for different message roles (system, user, assistant, tool)

## Configuration

- **Port**: Default is 5001, change in the `app.run()` call
- **Records per Page**: Default is 10, modify `records_per_page` variable
- **Max File Size**: Default is 1GB, adjust `MAX_CONTENT_LENGTH` setting

## Performance

The viewer is optimized for large files:
- Streams JSONL files line by line
- Only loads requested pages into memory
- Efficient line counting for pagination
- Background loading for statistics

## Comparison with json_log_viewer

This SFT data viewer is based on the `json_log_viewer` but adapted for:
- JSONL format instead of single JSON files
- Conversation-based data structure
- Large file handling with pagination
- Different metadata fields specific to SFT data
- Tool call visualization
- Multi-message conversation flows

## Example Usage

1. Start the server: `python sft_data_viewer.py`
2. Go to `http://localhost:5001`
3. Load your JSONL file (e.g., `/path/to/d2_full_truncated_30k_sep1.jsonl`)
4. Browse records, view conversations, and analyze statistics
