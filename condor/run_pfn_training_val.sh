#!/bin/bash

set -e
set -x

cd $HOME/assay-ml


export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CACHE_DIR=/scratch/chair_volkamer/michael.backenkoehler/cache
export GRAPH_CACHE_DIR=/tmp/graph_cache
mkdir -p "$GRAPH_CACHE_DIR"
mkdir -p "$CACHE_DIR"

uv run $1 \
  --seed 0 \
  --no-test \
  --num-epochs 100 \
  --fold $2 \
  --n-bins $3 \
  --mol-feat $4 \
  --unmasked-weight $5 \
  --property-set-ratio $6 \
  --norm $7 \
  --dropout $8 \
  --lr $9 \
  --obj ${10} \
  --act ${11}
