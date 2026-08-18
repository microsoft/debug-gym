# SFT Data Viewer

A web-based tool for viewing and analyzing Supervised Fine-Tuning (SFT) conversation data in JSONL format.

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python sft_data_viewer.py --safe-root /path/to/sft-data
```

3. Open `http://127.0.0.1:5001` in your browser

4. Upload a JSONL file to view conversation trajectories with:
   - Message-by-message navigation
   - Success/failure indicators  
   - Random shuffle for diverse sampling
   - Dataset statistics and analysis

The viewer binds to `127.0.0.1` and limits **Load from Server** to `.jsonl`
files under the current directory by default. Configure the narrowest suitable
root with `--safe-root` or `SFT_DATA_VIEWER_SAFE_ROOT`; traversal, sibling-prefix
paths, symbolic links, hard links, and mutable-ancestor escapes are rejected.
Server paths are relative to the configured root. Uploaded files continue to
work independently of the server-side safe root.

The application always binds to loopback; direct non-loopback serving is not
supported. For remote access, use an authenticated, single-user TLS reverse
proxy or SSH tunnel to the loopback listener. The proxy must either rewrite
`Host` to a loopback hostname or pass its exact hostname through
`--trusted-host`. `--trusted-host` is routing validation, not authentication.

Uploads are limited to 64 MiB. The viewer retains at most the active uploaded
file and deletes viewer-owned uploads on replacement, **Change File**, and
orderly shutdown. Large datasets should be placed below `--safe-root` and loaded
from the server instead; server-owned source files are never deleted.

## Data Format

Expects JSONL files with conversation objects containing `messages`, `problem`, `run_id`, `satisfied_criteria`, and token counts.
