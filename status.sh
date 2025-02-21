#!/bin/bash
output_dir=~/hodge-dti/data/output/
output_files=$(find $output_dir -name output.log | sort)

printf "%-32s %-6s %-5s %-10s %-10s\n" "Run    " "Epoch" "Fold" "Test Corr" "State"

upcoming=""
old_fold="-1"
for f in $output_files; do
    line=$(rg "Test Rank Corr:" $f | tail -n 1)
    dataset=$(echo "$line" | awk '{print $3}')
    epoch=$(echo "$line" | awk '{print $5}')
    fold=$(echo "$line" | awk '{print $7}')
    if [ "$old_fold" != "$fold" ]; then
        if [ "$old_fold" == "-1" ]; then
            echo "================================================================="
            old_fold=$fold
        elif [ "$fold" != "" ]; then
            echo "-----------------------------------------------------------------"
            old_fold=$fold
        fi
    fi
    rank_corr=$(echo "$line" | awk '{print $11}')
    last_line=$(tail -n 1 "$f")
    if echo "$last_line" | rg -q "Error"; then
        state="Error"
    elif rg -q "finished" "$f"; then
        state="Finished"
    else
        state="Running"
    fi
    if [ "$dataset" == "" ] && [ "$state" != "Error" ]; then
        upcoming="$upcoming $f"
    else
        if [ "$dataset" == "" ]; then
            fname=$(echo "$f" | cut -d '/' -f 7)
            fname=$(echo "[$fname]")
            printf "%-56s %-10s\n" "$fname" "$state"
        else
            printf "%-32s %-6s %-5s %-10s %-10s\n" "$dataset" "$epoch" "$fold" "$rank_corr" "$state"
        fi
    fi
done

if [ "$upcoming" != "" ]; then
    printf "\nOther:\n------\n"
    for f in $upcoming; do
        echo $f
        echo " $(tail -n 1 $f)"
    done
fi
