#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --job-name=0289_qwen3_hidden_states
#SBATCH -o logs/%x-%j.log


set -eu
source .venv/bin/activate

# model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
model_names=(
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    openai/gpt-oss-20b
    # microsoft/Phi-4-reasoning
    # nvidia/Llama-3.1-Nemotron-Nano-8B-v1
)

for model_name in "${model_names[@]}"; do
    model_suffix=${model_name##*/}
    input_file=data/cache_model_output/all.${model_suffix}.seed42.jsonl
    output_dir=data/hidden_states/${model_suffix}
    seed=42

    mkdir -p ${output_dir}

    echo "Getting hidden states for model: ${model_name}"
    echo "Input file: ${input_file}"
    echo "Output directory: ${output_dir}"

    python src/get_hidden_states.py \
        --model_name ${model_name} \
        --input_file ${input_file} \
        --output_dir ${output_dir} \
        --seed ${seed} \
        --debug

    echo "Finished processing model: ${model_name}"
done



echo Done
