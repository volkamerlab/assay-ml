#!/bin/bash

set -e
set -x

cd $HOME/hodge-dti

if [ ! -d venv ]; then
    python -m venv venv
fi

source venv/bin/activate

pip install -r requirements.txt
pip install -e .

python scripts/train_new.py
