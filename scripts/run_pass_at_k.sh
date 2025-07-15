#!/bin/bash

# This script runs the debugging agent on a set of test cases and evaluates its performance.

# Set the number of test cases to run
K=12
agent='explanation_agent'
problem='burnash__gspread.a8be3b96.lm_rewrite__596cni6x'

# Run the debugging agent on the test cases
for i in $(seq 1 $K); do
    echo "Running case $i..."
    # Run the debugging agent and capture its output
    OUTPUT=$(python scripts/run.py scripts/config_swesmith.yaml --agent $agent -p base.llm_name="qwen3-8b-vllm" -p base.problems=$problem -p base.output_path="exps/qwen_8b/ten_examples_pass_at_12/" -p base.uuid="explanation_qwen8b_$i")
    echo "Output for test case $i:"
    echo "$OUTPUT"
done

# Evaluate the performance of the debugging agent
echo "Evaluating performance..."
# (Insert evaluation code here)