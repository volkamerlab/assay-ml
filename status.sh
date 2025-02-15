#!/bin/bash
output_dir=~/hodge-dti/data/output/
output_files=$(find $output_dir -name output.log | sort)
for f in $output_files; do
    echo $f
    for i in 0 1 2 3 4; do
        cur_performance=$(grep "test_rank_corr" $f | grep "fold=$i" | tail -n 1)
        cur_corr=$(echo $cur_performance | awk '{print $NF}' | cut -c 16-)
        cur_epoch=$(echo $cur_performance | awk '{print $4}')
        echo "fold=$i $cur_epoch: $cur_corr"
    done
done
