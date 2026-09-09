#!/bin/bash

set -e
set -x

cd $HOME/hodge-dti

#uv sync
#uv pip install -e .

export CUBLAS_WORKSPACE_CONFIG=:4096:8
#export JOBLIB_TEMP_FOLDER=/home/michael.backenkoehler/hodge-dti/data/tmp
ASSAY_ML_IDENT="${3}_allsets_${5}" uv run $1 $2 $3 allsets $4  &
ASSAY_ML_IDENT="${3}_sets_${5}" uv run $1 $2 $3 sets $4  &
ASSAY_ML_IDENT="${3}_ic50_${5}" uv run $1 $2 $3 ic50 $4  &

wait
