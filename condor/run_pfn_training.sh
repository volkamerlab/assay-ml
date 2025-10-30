#!/bin/bash

set -e
set -x

cd $HOME/hodge-dti

#uv sync
#uv pip install -e .

export CUBLAS_WORKSPACE_CONFIG=:4096:8
#export JOBLIB_TEMP_FOLDER=/home/michael.backenkoehler/hodge-dti/data/tmp
uv run $1 --fold $2 --mol-feat $3 --unmasked-weight $4 --property-set-ratio $5 --n-bins $6
