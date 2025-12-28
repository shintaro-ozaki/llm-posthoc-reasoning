#!/bin/bash
set -eu
source .venv/bin/activate

model_names=(
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    # deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
    # nvidia/Llama-3.1-Nemotron-Nano-8B-v1
    openai/gpt-oss-20b
    # microsoft/Phi-4-reasoning
)

seed=42
total_size=21500
chunk_size=2000

for model_name in "${model_names[@]}"; do
    model_suffix=${model_name##*/}

    for ((start=0; start<total_size; start+=chunk_size)); do
        end=$((start + chunk_size))
        if [ $end -gt $total_size ]; then
            end=$total_size
        fi

        job_suffix="${model_suffix}_${start}_${end}"

        sbatch <<__EOF__
#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --job-name=0289_get_neuron_${job_suffix}
#SBATCH -o logs/slurm-${job_suffix}-%j.log

set -eu
source .venv/bin/activate

model_name="${model_name}"
model_suffix="${model_suffix}"

input_file="data/cache_model_output/all.\${model_suffix}.seed${seed}.jsonl"
output_dir="data/hidden_states/\${model_suffix}"

mkdir -p "\${output_dir}"

echo "========================================"
echo "Model: \${model_name}"
echo "Range: ${start} – ${end}"
echo "Input: \${input_file}"
echo "Output: \${output_dir}"
echo "========================================"

python src/get_hidden_states.py \
    --model_name "\${model_name}" \
    --input_file "\${input_file}" \
    --output_dir "\${output_dir}" \
    --start ${start} \
    --end ${end} \
    --seed ${seed}

echo "Finished model: \${model_name} [${start}, ${end})"
__EOF__

    done
done

echo "All sbatch jobs submitted."
