#!/bin/bash

set -e
set -x

cd $HOME/hodge-dti

# uv sync
#uv pip install -e .

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FP_CACHE_DIR=/scratch/chair_volkamer/michael.backenkoehler
mkdir -p $FP_CACHE_DIR
chmod 777 $FP_CACHE_DIR
#export JOBLIB_TEMP_FOLDER=/home/michael.backenkoehler/hodge-dti/data/tmp
uv run $1 --seed $2 --dataset $3 --method $4 --fold $5 --mol-feat $6 --unmasked-weight $7 --property-set-ratio $8 --n_bins $9
