#!/usr/bin/env bash

# Set PYTHONPATH to include the mmdetection directory
export PYTHONPATH="$(pwd)":$PYTHONPATH

# Run the visualization script
python tools/visualize_mask_and_sparse_rcnn_comparison.py

echo "Visualization completed. Results saved in work_dirs/visualization_comparison/" 