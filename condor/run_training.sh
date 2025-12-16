#!/bin/bash

set -e
set -x

cd $HOME/hodge-dti

#uv sync
#uv pip install -e .

export CUBLAS_WORKSPACE_CONFIG=:4096:8
#export JOBLIB_TEMP_FOLDER=/home/michael.backenkoehler/hodge-dti/data/tmp
ASSAY_ML_IDENT="${3}_allsets_${5}" uv run $1 $2 $3 allsets $5 $6
ASSAY_ML_IDENT="${3}_sets_${5}" ASSAY_ML_MODEL_WEIGHTS="data/${3}_allsets_${5}/model${5}.pt" uv run $1 $2 $3 sets $5 $6
ASSAY_ML_IDENT="${3}_ic50_${5}" uv run $1 $2 $3 ic50 $5 $6

