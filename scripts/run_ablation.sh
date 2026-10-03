#!/bin/bash

# Ablation Study Runner for PCNet
# Runs all variant experiments from Ablation/ directory

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ABLATION_DIR="$SCRIPT_DIR/Ablation"

echo "========================================="
echo "PCNet Ablation Study"
echo "========================================="

# Run ablation experiments (DPRF, FPCF, FPRF, MLP)
for variant_script in "$ABLATION_DIR/DPRF.sh" "$ABLATION_DIR/FPCF.sh" "$ABLATION_DIR/FPRF.sh" "$ABLATION_DIR/MLP.sh"
do
    if [ -f "$variant_script" ]; then
        variant_name=$(basename "$variant_script")
        echo ""
        echo "=== Running $variant_name ==="
        bash "$variant_script"
    fi
done

echo ""
echo "========================================="
echo "All Ablation Experiments Completed!"
echo "========================================="