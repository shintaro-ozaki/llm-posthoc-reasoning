#!/bin/bash


set -eu
source .venv/bin/activate

project=$(pwd)


qas=(
    mathqa
    hellaswag
)

for qa in "${qas[@]}"; do
    echo "Processing $qa"

    input_file=$project/data/model_input/$qa.jsonl
    output_file=$project/data/annotation_format/$qa.csv
    mkdir -p $(dirname $output_file)

    echo QA: $qa
    echo Input file: $input_file
    echo Output file: $output_file

    python src/create.annotation.format.py \
        --input_file $input_file \
        --output_file $output_file
done
echo Done
