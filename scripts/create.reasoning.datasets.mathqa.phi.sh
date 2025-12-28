#!/bin/bash
#SBATCH -p gpu_long
#SBATCH -c 2
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:6000:1
#SBATCH --account=is-nlp
#SBATCH --job-name=mathqa
#SBATCH -o logs/slurm-%x-%j.log

set -eu
source .venv/bin/activate

# model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
model_name=openai/gpt-oss-20b
# model_name=Qwen/Qwen3-4B-Instruct-2507
model_name=microsoft/Phi-4-reasoning
# model_name=meta-llama/Llama-3.3-70B-Instruct
# model_name=Qwen/Qwen2.5-72B-Instruct

model_suffix=${model_name##*/}

qa=mathqa
input_file=data/model_input/${qa}.json
output_file=data/reasoning_datasets_before_split/${qa}.${model_suffix}.json
mkdir -p $(dirname ${output_file})

echo QA: ${qa}
echo input_file: ${input_file}
echo output_file: ${output_file}
echo model_name: ${model_name}

python src/create.reasoning.datasets.py \
    --input_file ${input_file} \
    --model_name ${model_name} \
    --output_file ${output_file} \
    --seed 42 \
    --debug \
    --debug_sample_num 10

echo Done
