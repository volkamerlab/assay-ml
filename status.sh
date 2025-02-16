#!/bin/bash
output_dir=~/hodge-dti/data/output/
output_files=$(find $output_dir -name output.log | sort)
for f in $output_files; do
    echo $f
    for i in 0 1 2 3 4; do
        cur_performance=$(grep "Test Rank Corr:" $f | grep "Fold: $i" | tail -n 1)
        cur_corr=$(echo $cur_performance | awk '{print $NF}')
        cur_epoch=$(echo $cur_performance | awk '{print $5}' | cut -c -3)
        echo "fold=$i epoch=$cur_epoch: $cur_corr"
    done
done
