#!/bin/bash

set -x

cd $HOME/hodge-dti

#export JOBLIB_TEMP_FOLDER=/home/michael.backenkoehler/hodge-dti/data/tmp
uv run scripts/fit_flaml_baseline.py $1
