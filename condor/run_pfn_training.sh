#!/bin/bash

set -e
set -x

cd $HOME/hodge-dti

#uv sync
#uv pip install -e .

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export GRAPH_CACHE_DIR=/tmp/graph_cache #/scratch/chair_volkamer/michael.backenkoehler/graph_cache
mkdir -p "$GRAPH_CACHE_DIR"

uv run $1 --fold $2 --seed 0 --n-bins $3 --mol-feat $4 --unmasked-weight $5 --property-set-ratio $6
