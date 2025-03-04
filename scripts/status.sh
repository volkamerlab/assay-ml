#!/bin/bash
output_dir=data/hpc-data/
remote_dir=/home/michael.backenkoehler/hodge-dti/data/output/
remote=michael.backenkoehler@conduit.cs.uni-saarland.de:$remote_dir

rsync -av --exclude='*.pt' --exclude='*.csv' $remote $output_dir > /dev/null

output_files=$(find $output_dir -name output.log | sort)
line_length=63

bar ()
{
    printf "${1}%.0s" $(seq 1 $2)
    printf '\n'
}
bar '━' $((${line_length}))
printf "%-6s %-13s %-9s %-6s %-5s %-10s %-10s\n" "Ident" "Dataset" "Method" "Epoch" "Fold" "Test Corr" "State"
bar '━' $((${line_length}))


upcoming=""
old_fold="-1"
old_dataset=""
for f in $output_files; do
    line=$(rg "Test Rank Corr:" $f | tail -n 1)
    run_name=$(basename $(dirname $f))
    dataset=$(echo $run_name | cut -d '_' -f 1)
    method=$(echo $run_name | cut -d '_' -f 3)
    ident=$(echo $run_name | cut -d '_' -f 4 | cut -c -4)
    fold=$(echo $run_name | cut -d '_' -f 2)
    epoch=$(echo "$line" | awk '{print $5}')
    if [ "$old_fold" != "$fold" ] || [ "$old_dataset" != "$dataset" ]; then
        if [ "$fold" != "" ]; then
            bar '―' $line_length
        fi
        old_fold=$fold
        old_dataset=$dataset
    fi
    rank_corr=$(echo "$line" | awk '{print $11}')
    last_line=$(tail -n 1 "$f")
    if echo "$last_line" | rg -q -e "ERROR" -e "assertions"; then
        state="Error"
    elif [ "$last_line" == "" ]; then
        state="Error"
    elif rg -q "finished" "$f"; then
        state="Finished"
    else
        state="Running"
    fi
    if [ "$state" != "Error" ]; then
        printf "%-6s %-13s %-9s %-6s %-5s %-10s %-10s\n" "$ident" "$dataset" "$method" "$epoch" "$fold" "$rank_corr" "$state"
    fi
done
bar '━' $line_length
