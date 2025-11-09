#!/bin/bash

set -e
set -x

cd $HOME/hodge-dti

#uv sync
#uv pip install -e .

export CUBLAS_WORKSPACE_CONFIG=:4096:8
#export JOBLIB_TEMP_FOLDER=/home/michael.backenkoehler/hodge-dti/data/tmp

uv run $1 --fold $2 --seed 0 --n-bins $3 --mol-feat $4 --unmasked-weight $5 --property-set-ratio $6
