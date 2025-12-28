#!/bin/bash


# set -eu
source .venv/bin/activate

project=$(pwd)

model_names=(
    # Llama-3.3-70B-Instruct
    # Qwen2.5-72B-Instruct
    QwQ-32B
    gpt-oss-20b
    # Phi-4-reasoning
)
qas=(
    mathqa
    hellaswag
)

for model_name in "${model_names[@]}"; do
    for qa in "${qas[@]}"; do
        input_file=$project/data/reasoning_datasets_before_split/${qa}.${model_name}.json
        correct_output_file=$project/data/reasoning_datasets_after_split/${qa}.${model_name}.correct.json
        incorrect_output_file=$project/data/reasoning_datasets_after_split/${qa}.${model_name}.incorrect.json

        mkdir -p "$(dirname "$correct_output_file")"
        mkdir -p "$(dirname "$incorrect_output_file")"

        echo Model: "$model_name"
        echo QA: "$qa"
        echo Input file: "$input_file"
        echo Correct output file: "$correct_output_file"
        echo Incorrect output file: "$incorrect_output_file"

        python src/1130.convert.to.input.py \
            --input_file "$input_file" \
            --correct_output_file "$correct_output_file" \
            --incorrect_output_file "$incorrect_output_file"
    done
done

echo Done
