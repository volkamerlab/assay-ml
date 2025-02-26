#!/bin/bash
output_dir=data/hpc-data/
remote_dir=/home/michael.backenkoehler/hodge-dti/data/output/
remote=michael.backenkoehler@conduit.cs.uni-saarland.de:$remote_dir

echo "sync files"
rsync -av --exclude='*.pt' --exclude='*.csv' $remote $output_dir > /dev/null

output_files=$(find $output_dir -name output.log | sort)

bar ()
{
    printf "${1}%.0s" $(seq 1 $2)
    printf '\n'
}
bar '━' 63
printf "%-6s %-13s %-9s %-6s %-5s %-10s %-10s\n" "Ident" "Dataset" "Method" "Epoch" "Fold" "Test Corr" "State"
bar '━' 63


upcoming=""
old_fold="-1"
for f in $output_files; do
    line=$(rg "Test Rank Corr:" $f | tail -n 1)
    run_name=$(echo "$line" | awk '{print $3}')
    dataset=$(echo $run_name | cut -d '_' -f 1 | cut -c 2-)
    method=$(echo $run_name | cut -d '_' -f 3)
    ident=$(echo $run_name | cut -d '_' -f 4 | cut -c -4)
    epoch=$(echo "$line" | awk '{print $5}')
    fold=$(echo "$line" | awk '{print $7}')
    if [ "$old_fold" != "$fold" ]; then
        if [ "$old_fold" == "-1" ]; then
            old_fold=$fold
        elif [ "$fold" != "" ]; then
            bar '⎯' 62
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
    if [ "$run_name" == "" ] && [ "$state" != "Error" ]; then
        upcoming="$upcoming $f"
    else
        if [ "$run_name" == "" ]; then
            fname=$(echo "$f" | cut -d '/' -f 7)
            fname=$(echo "[$fname]")
            printf "%-52s %-10s\n" "$fname" "$state"
        else
            printf "%-6s %-13s %-9s %-6s %-5s %-10s %-10s\n" "$ident" "$dataset" "$method" "$epoch" "$fold" "$rank_corr" "$state"
        fi
    fi
done
bar '━' 63

if [ "$upcoming" != "" ]; then
    printf "\nOther:\n⎯⎯⎯⎯⎯⎯\n"
    for f in $upcoming; do
        echo $f
        echo " $(tail -n 1 $f)"
    done
fi
