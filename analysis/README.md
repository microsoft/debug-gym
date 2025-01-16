# Analysis Folder

This folder contains scripts for analyzing data after runing the agent through the Froggy. Below is a brief description of each script:

## Scripts

### gather_results_aider.py
> This script gathers success rate results from all problems in the aider bench.

### comparisons_aider.py
> This script is used to compare different agent performance on the aider bench. It includes the number of success cases, number of rewrites, and number of pdb usages. It generates output in a figure in the `/results` folder and names of problems that are difficult to solve in the `/results` folder.

## Usage

To run the scripts, you need output folders that contains all the problems in.
The output folders should be like following hierachy:
```
output_aider/
├── baseline uuid1/
│   ├── wordy/       # name of problem in aider bench
│   ├── two_bucket/
│   ├── acronym/
│   ├── beer_song/
│   ├── ... 
├── agent uuid1/
│   ├── wordy/
│   ├── two_bucket/
│   ├── acronym/
│   ├── beer_song/
│   ├── ... 
```

## Dependencies

Before running the scripts, ensure you have the necessary dependencies installed. You can install them using pip:

```bash
pip install termcolor matplotlib
```

### Input Instructions

To input agent names and uuids for analysis, use the `parse_folders.jsonl` file. Follow these guidelines:

- For a single agent with multiple runs, list the UUIDs as follows:
    ```json
    ["input uuid 1 here", "input uuid 2 here"]
    ```

- To compare two agents (agent1 and agent2), place their UUID lists in the respective fields:
    ```json
    {
        "baseline": ["agent1 uuid 1", "agent1 uuid 2"],
        "agent": ["agent2 uuid 1", "agent2 uuid 2"]
    }
    ```

To run the analysis code, use the following commands:

```bash
python analysis/gather_results_aider.py
```

```bash
python analysis/comparisons_aider.py
```
