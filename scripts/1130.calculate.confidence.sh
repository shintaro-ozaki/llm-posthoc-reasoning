#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 2
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --account=is-nlp
#SBATCH --job-name=sample
#SBATCH -o logs/slurm-%x-%j.log

set -eu
source .venv/bin/activate


qa=mathqa
qa=hellaswag

# model_name=Qwen/Qwen3-4B-Instruct-2507
model_name=openai/gpt-oss-20b
# model_name=Qwen/QwQ-32B
model_suffix=${model_name##*/}

# input_file=data/reasoning_datasets_after_split/${qa}.QwQ-32B.correct.json
input_file=data/reasoning_datasets_after_split/${qa}.${model_suffix}.correct.json
output_file=data/model_output/${qa}/${model_suffix}.confidence.jsonl

mkdir -p "$(dirname "${output_file}")"

echo "$model_name"
echo "$input_file"
echo "$output_file"

python src/1130.calculate.confidence.py \
    --model_name "${model_name}" \
    --input_file "${input_file}" \
    --output_file "${output_file}"

echo Done

