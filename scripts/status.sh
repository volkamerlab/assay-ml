#!/bin/bash

output_dir=data/hpc-data/
remote=$MBHPC/hodge-dti/data/output/

rsync -a --exclude='*.pt' --exclude='*.csv' $remote $output_dir  # > /dev/null

output_files=$(find $output_dir -name output.log | sort)
line_length=75

bar () {
    printf "${1}%.0s" $(seq 1 $2)
    printf '\n'
}

print_table () {
    local files=("$@")
    local old_fold="-1"
    local old_dataset=""

    bar '━' $line_length
    printf "%-6s %-10s %-10s %-10s %-9s %-5s %-10s %-10s\n" \
        "Ident" "Dataset" "FP" "Method" "Epoch" "Fold" "Corr" "State"
    bar '━' $line_length

    for f in "${files[@]}"; do
        # grab last test performance line
        line=$(rg "test rank corr:" "$f" | tail -n 1)
        run_name=$(basename $(dirname "$f"))

        run_name="${run_name%_coldtgt}"
        IFS='_' read -r -a parts <<< "$run_name"
        len=${#parts[@]}

        ident="${parts[$((len-1))]:-}"
        method="${parts[$((len-2))]:-}"
        fold="${parts[$((len-3))]:-}"
        fingerprint="${parts[$((len-4))]:-}"

        # Dataset = everything before fingerprint
        if (( len > 4 )); then
            dataset_parts=("${parts[@]:0:len-4}")
            dataset="${dataset_parts[*]}"
            dataset="${dataset// /_}"
        else
            dataset=""
        fi

        ident="${ident:0:4}"   # trim ident
        epoch=$(echo "$line" | awk '{print $4}')
        cur_epoch=$(rg "epoch:" "$f"| tail -n 1 | awk '{print $4}')
        rank_corr=$(echo "$line" | awk '{print $10}')

        if [ "$old_fold" != "$fold" ] || [ "$old_dataset" != "$dataset" ]; then
            if [ "$fold" != "" -a "$old_dataset" != "" ]; then
                bar '―' $line_length
            fi
            old_fold=$fold
            old_dataset=$dataset
        fi

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
            printf "%-6s %-10s %-10s %-10s %-4s/%-4s %-5s %-10s %-10s\n" \
                "$ident" "$dataset" "$fingerprint" "$method" "$epoch" "$cur_epoch" "$fold" "$rank_corr" "$state"
        fi
    done

    bar '━' $line_length
}

# Separate into two groups
normal_runs=()
coldtgt_runs=()

for f in $output_files; do
    run_name=$(basename $(dirname "$f"))
    if [[ "$run_name" == *_coldtgt ]]; then
        coldtgt_runs+=("$f")
    else
        normal_runs+=("$f")
    fi
done

# Print normal runs or cold target runs
if [ "$1" == "-cold" ]; then
    echo "Cold Target Runs"
    print_table "${coldtgt_runs[@]}"
else
    echo "Regular Runs"
    print_table "${normal_runs[@]}"
fi
