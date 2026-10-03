#!/bin/bash

# Performance Evaluation Runner for PCNet (DPCF mode)
# Runs all dataset experiments from PCNet/ directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PCNET_DIR="$SCRIPT_DIR/PCNet"

echo "========================================="
echo "PCNet Performance Evaluation (DPCF Mode)"
echo "========================================="

# Run all dataset experiments
for dataset_script in "$PCNET_DIR"/*.sh
do
    if [ -f "$dataset_script" ]; then
        dataset_name=$(basename "$dataset_script")
        echo ""
        echo "=== Running $dataset_name ==="
        bash "$dataset_script"
    fi
done

echo ""
echo "========================================="
echo "All Performance Experiments Completed!"
echo "========================================="