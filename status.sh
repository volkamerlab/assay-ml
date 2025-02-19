#!/bin/bash
output_dir=~/hodge-dti/data/output/
output_files=$(find $output_dir -name output.log | sort)

printf "%-32s %-6s %-5s %-10s %-10s\n" "Dataset" "Epoch" "Fold" "Test Corr" "Finished"
echo "------------------------------------------------------------------"

for f in $output_files; do
    line=$(grep "Test Rank Corr:" $f | tail -n 1)
    dataset=$(echo "$line" | awk '{print $3}')
    epoch=$(echo "$line" | awk '{print $5}')
    fold=$(echo "$line" | awk '{print $7}')
    rank_corr=$(echo "$line" | awk '{print $11}')
    finished=$(grep -q "finished" $f && echo "x" || echo "")
    last_line=$(tail -n 1 "$f")
    if echo "$last_line" | grep -q "Error"; then
        finished="E"
    elif grep -q "finished" "$f"; then
        finished="x"
    else
        finished=""
    fi
    printf "%-32s %-6s %-5s %-10s %-10s\n" "$dataset" "$epoch" "$fold" "$rank_corr" "$finished"
done
