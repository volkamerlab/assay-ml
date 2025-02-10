#!/bin/bash

set -e
set -x

cd $HOME/hodge-dti

if [ ! -d venv ]; then
    python -m venv venv
fi

source venv/bin/activate

#pip install -r requirements.txt
#pip install -e .

export CUBLAS_WORKSPACE_CONFIG=:4096:8
python $1 $2 $3 $4
