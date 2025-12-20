# JSON Log Viewer

A Flask-based web viewer for debug-gym trajectory JSON files. Visualize agent exploration sessions with step-by-step action breakdowns.

## Installation

```bash
cd analysis/json_log_viewer
pip install -r requirements.txt
```

## Usage

Start the server:

```bash
python json_log_viewer.py -p 5050
```

Then open http://127.0.0.1:5050 in your browser.

### Loading Trajectories

You can load trajectory files in several ways:

1. **Upload**: Click "Upload" and select a JSON file
2. **Browse**: Click "Browse Files" to navigate your filesystem
3. **API**: Load programmatically via `GET /load_file_from_path?path=/path/to/trajectory.json`

### Integration with Gray Tree Frog

The viewer supports CORS requests, allowing Gray Tree Frog's lineage visualization to open trajectories directly. When viewing the lineage graph, click "View trajectory" on any discovery to open its exploration session.

## Features

- Step-by-step trajectory visualization
- Color-coded action types (bash, view, edit, etc.)
- Detailed bash command classification
- Statistics view showing action distribution
- Keyboard navigation between steps
