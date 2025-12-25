#!/bin/bash

# model=gpt-oss-20b
model=QwQ-32B
# qa=hellaswag
qa=mathqa

base_dir=/cl/home3/shintaro/llm-reasoning-posthoc/data/reasoning_datasets_before_split

jq -s 'flatten' \
  $(ls ${base_dir}/${qa}.${model}.*.json | sort -V) \
  > ${base_dir}/${qa}.${model}.json

echo "Merged files for model ${model} and QA ${qa}"
