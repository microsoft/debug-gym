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
    python scripts/run.py scripts/config_swesmith.yaml --agent explanation_agent -p base.problems="cantools__cantools.0c6a7871.func_pm_remove_assign__0d3u2d21" -p base.llm_name="gpt-4o" -p base.output_path="exps/jul21-branch/gpt-4o/test_4o_always_fail/" -p base.uuid="explanation_4o_$i"
    # echo "Output for test case $i:"
    # echo "$OUTPUT"
    sleep 60
done

# Evaluate the performance of the debugging agent
echo "Evaluating performance..."
# (Insert evaluation code here)