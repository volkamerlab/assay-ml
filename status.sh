#!/bin/bash
output_dir=~/hodge-dti/data/output/
output_files=$(find $output_dir -name output.log | sort)

printf "%-32s %-6s %-5s %-10s %-10s\n" "Dataset" "Epoch" "Fold" "Test Corr" "State"
echo "-----------------------------------------------------------------"

upcoming=""
for f in $output_files; do
    line=$(rg "Test Rank Corr:" $f | tail -n 1)
    dataset=$(echo "$line" | awk '{print $3}')
    epoch=$(echo "$line" | awk '{print $5}')
    fold=$(echo "$line" | awk '{print $7}')
    rank_corr=$(echo "$line" | awk '{print $11}')
    last_line=$(tail -n 1 "$f")
    if echo "$last_line" | rg -q "Error"; then
        state="Error"
    elif rg -q "finished" "$f"; then
        state="Finished"
    else
        state="Running"
    fi
    if [ "$dataset" == "" ]; then
        upcoming="$upcoming $f"
    else
        printf "%-32s %-6s %-5s %-10s %-10s\n" "$dataset" "$epoch" "$fold" "$rank_corr" "$state"
    fi
done

printf "\nOther:\n------\n"
for f in $upcoming; do
    echo $f
    echo " $(tail -n 1 $f)"
done
