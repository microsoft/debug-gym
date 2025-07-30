#!/bin/bash

# This script runs the debugging agent on a set of test cases and evaluates its performance.

# Set the number of test cases to run
K=12
agent='debug_agent'
problem='pndurette__gTTS.dbcda4f3.lm_rewrite__4m80szt9'

# Run the debugging agent on the test cases
for i in $(seq 1 $K); do
    echo "Running case $i..."
    # Run the debugging agent and capture its output
    python scripts/run.py scripts/config_swesmith.yaml --agent explanation_agent -p base.llm_name="gpt-4o" -p base.output_path="exps/jul29/gpt-4o-fails/explanation_agent" -p base.uuid="explanation_4o_$i" -n 20
    # echo "Output for test case $i:"
    # echo "$OUTPUT"
    sleep 15
done

# Evaluate the performance of the debugging agent
echo "Evaluating performance..."
# (Insert evaluation code here)d