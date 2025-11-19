#!/bin/bash

set -e
set -x

cd $HOME/hodge-dti

#uv sync
#uv pip install -e .

export CUBLAS_WORKSPACE_CONFIG=:4096:8
#export GRAPH_CACHE_DIR=/tmp/graph_cache 
export CACHE_DIR=/scratch/chair_volkamer/michael.backenkoehler/cache
export GRAPH_CACHE_DIR=/scratch/chair_volkamer/michael.backenkoehler/cache/pyg
mkdir -p "$GRAPH_CACHE_DIR"
#rm -rf /scratch/chair_volkamer/michael.backenkoehler/graph_cache/* &
mkdir -p "$CACHE_DIR"
# rm -rf /scratch/chair_volkamer/michael.backenkoehler/cache/chembl/0/{all,allfp}/{train,val,test}

# arguments = $(script) $(fold) $(n_bins) $(mol_feat) $(unmasked_weight) $(physcem) $(dropout) $(lr)
uv run $1 --fold $2 --seed 0 --n-bins $3 --mol-feat $4 --unmasked-weight $5 --property-set-ratio $6 --dropout $7 --lr $8
