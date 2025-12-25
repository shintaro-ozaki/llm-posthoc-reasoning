#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 2
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:6000:1
#SBATCH --account=is-nlp
#SBATCH --job-name=openai.hellaswag
#SBATCH -o logs/slurm-%x-%A_%a.log
#SBATCH --array=0-19

set -eu
source .venv/bin/activate

model_name=openai/gpt-oss-20b
model_suffix=${model_name##*/}

qa=hellaswag
input_file=data/model_input/${qa}.json

# batch settings
BATCH_SIZE=20
SHARD_ID=${SLURM_ARRAY_TASK_ID}

START_IDX=$(( SHARD_ID * BATCH_SIZE ))
END_IDX=$(( START_IDX + BATCH_SIZE ))

output_file=data/reasoning_datasets_before_split/${qa}.${model_suffix}.${START_IDX}.${END_IDX}.json
mkdir -p "$(dirname "${output_file}")"

echo "QA: ${qa}"
echo "Model: ${model_name}"
echo "Input file: ${input_file}"
echo "Output file: ${output_file}"
echo "Shard ID: ${SHARD_ID}"
echo "Start idx: ${START_IDX}"
echo "End idx: ${END_IDX}"

python src/create.reasoning.datasets.py \
    --input_file "${input_file}" \
    --model_name "${model_name}" \
    --output_file "${output_file}" \
    --seed 42 \
    --debug \
    --debug_sample_num 400 \
    --start_idx "${START_IDX}" \
    --end_idx "${END_IDX}"

echo "Done ${START_IDX}-${END_IDX}"
