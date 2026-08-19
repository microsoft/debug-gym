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
python json_log_viewer.py -p 5050 --safe-root /path/to/trajectories
```

Then open http://127.0.0.1:5050 in your browser.

The viewer binds to `127.0.0.1` and confines server-side browsing and loading to
the current directory by default. Set `--safe-root` (or
`JSON_LOG_VIEWER_SAFE_ROOT`) to the narrowest directory containing the logs you
need. API and browser paths are relative to that root. Symbolic links, hard
links, traversal, and files opened through a replaced ancestor are rejected.
The application always binds to loopback; direct non-loopback serving is not
supported.

If remote access is required, place an authenticated, single-user TLS reverse
proxy or SSH tunnel in front of the loopback listener. The proxy must restrict
access to one trusted operator and either rewrite `Host` to a loopback hostname
or pass its exact hostname through `--trusted-host`. `--trusted-host` validates
routing input only; it is not authentication.

### Loading Trajectories

You can load trajectory files in several ways:

1. **Upload**: Click "Upload" and select a JSON file
2. **Browse**: Click "Browse Files" to navigate within the configured safe root
3. **API**: Load an in-root file with
   `POST /load_file_from_path` and JSON body `{"path":"trajectory.json"}`

Uploads are limited to 16 MiB, parsed through an exclusive temporary file, and
deleted immediately after parsing. Loaded data and its display name are
published as one thread-safe snapshot.

### Integration with Gray Tree Frog

Cross-origin requests are disabled by default. To allow Gray Tree Frog's
lineage visualization to open trajectories, configure its exact origin:

```bash
python json_log_viewer.py \
  --safe-root /path/to/trajectories \
  --allowed-origin https://gray-tree-frog.example
```

Repeat `--allowed-origin` for multiple trusted origins, or set a comma-separated
`JSON_LOG_VIEWER_ALLOWED_ORIGINS` value. Wildcard origins are not supported.
The viewer accepts only loopback `Host` headers by default. A local authenticated
proxy may add its exact hostname with `--trusted-host`.

## Features

- Step-by-step trajectory visualization
- Color-coded action types (bash, view, edit, etc.)
- Detailed bash command classification
- Statistics view showing action distribution
- Keyboard navigation between steps
