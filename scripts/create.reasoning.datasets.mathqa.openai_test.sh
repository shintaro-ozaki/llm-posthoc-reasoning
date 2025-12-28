#!/bin/bash
set -eu

# ========= 環境 =========
source .venv/bin/activate

# ========= model settings =========
model_name=openai/gpt-oss-20b
model_suffix=${model_name##*/}

# ========= dataset =========
qa=mathqa
input_file=data/model_input/${qa}.json

# ========= debug settings =========
DEBUG_SAMPLE_NUM=50     # debugで使う最大件数
START_IDX=0             # debugでは通常 0
END_IDX=50              # debugでは小さく

# ========= output =========
output_file=data/reasoning_datasets_before_split/${qa}.${model_suffix}.debug.${START_IDX}.${END_IDX}.json
mkdir -p "$(dirname "${output_file}")"

# ========= logging =========
echo "===== DEBUG RUN ====="
echo "QA: ${qa}"
echo "Model: ${model_name}"
echo "Input file: ${input_file}"
echo "Output file: ${output_file}"
echo "Start idx: ${START_IDX}"
echo "End idx: ${END_IDX}"
echo "Debug sample num: ${DEBUG_SAMPLE_NUM}"
echo "====================="

# ========= run =========
python src/create.reasoning.datasets.py \
    --input_file "${input_file}" \
    --model_name "${model_name}" \
    --output_file "${output_file}" \
    --seed 42 \
    --debug \
    --debug_sample_num "${DEBUG_SAMPLE_NUM}" \
    --start_idx "${START_IDX}" \
    --end_idx "${END_IDX}"

echo "✅ Done debug run"
