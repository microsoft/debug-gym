# SFT Data Viewer

A web-based tool for viewing and analyzing Supervised Fine-Tuning (SFT) conversation data in JSONL format.

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python sft_data_viewer.py
```

3. Open `http://localhost:5001` in your browser

4. Upload a JSONL file to view conversation trajectories with:
   - Message-by-message navigation
   - Success/failure indicators  
   - Random shuffle for diverse sampling
   - Dataset statistics and analysis

## Data Format

Expects JSONL files with conversation objects containing `messages`, `problem`, `run_id`, `satisfied_criteria`, and token counts.
