#!/bin/bash

set -e
set -x

cd $HOME/hodge-dti

uv sync
uv pip install -e .

#export JOBLIB_TEMP_FOLDER=/home/michael.backenkoehler/hodge-dti/data/tmp
uv run scripts/fit_flaml_baseline.py
