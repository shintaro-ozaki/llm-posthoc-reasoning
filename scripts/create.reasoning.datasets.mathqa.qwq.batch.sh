# #!/bin/bash
# #SBATCH -p gpu_long
# #SBATCH -c 2
# #SBATCH -t 100:00:00
# #SBATCH --gres=gpu:6000:1
# #SBATCH --account=is-nlp
# #SBATCH --job-name=qwq.mathqa
# #SBATCH -o logs/slurm-%x-%A_%a.log
# #SBATCH --array=0-9

# set -eu
# source .venv/bin/activate

# # model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
# model_name=Qwen/QwQ-32B
# model_suffix=${model_name##*/}

# qa=mathqa
# input_file=data/model_input/${qa}.json

# # batch settings
# BATCH_SIZE=30
# SHARD_ID=${SLURM_ARRAY_TASK_ID}

# START_IDX=$(( SHARD_ID * BATCH_SIZE ))
# END_IDX=$(( START_IDX + BATCH_SIZE ))

# output_file=data/reasoning_datasets_before_split/${qa}.${model_suffix}.${START_IDX}.${END_IDX}.json
# mkdir -p "$(dirname "${output_file}")"

# echo "QA: ${qa}"
# echo "Model: ${model_name}"
# echo "Input file: ${input_file}"
# echo "Output file: ${output_file}"
# echo "Shard ID: ${SHARD_ID}"
# echo "Start idx: ${START_IDX}"
# echo "End idx: ${END_IDX}"

# python src/create.reasoning.datasets.py \
#     --input_file "${input_file}" \
#     --model_name "${model_name}" \
#     --output_file "${output_file}" \
#     --seed 42 \
#     --use_4bit \
#     --debug \
#     --debug_sample_num 400 \
#     --start_idx "${START_IDX}" \
#     --end_idx "${END_IDX}"

# echo "Done ${START_IDX}-${END_IDX}"


#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 2
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:6000:1
#SBATCH -x elm72
#SBATCH --account=is-nlp
#SBATCH --job-name=qwq.mathqa
#SBATCH -o logs/slurm-%x-%A_%a.log
#SBATCH --array=0-3

set -eu
source .venv/bin/activate

# model settings
model_name=Qwen/QwQ-32B
model_suffix=${model_name##*/}

qa=mathqa
input_file=data/model_input/${qa}.json

# batch settings
BATCH_SIZE=25
BASE_START=300
SHARD_ID=${SLURM_ARRAY_TASK_ID}

START_IDX=$(( BASE_START + SHARD_ID * BATCH_SIZE ))
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
    --use_4bit \
    --debug \
    --debug_sample_num 400 \
    --start_idx "${START_IDX}" \
    --end_idx "${END_IDX}"

echo "Done ${START_IDX}-${END_IDX}"
