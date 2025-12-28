#!/bin/bash
set -eu

model=QwQ-32B
qa=mathqa
base_dir=data/reasoning_datasets_before_split

jq -s 'flatten' \
  $(ls ${base_dir}/${qa}.${model}.*.json | sort -V) \
  > ${base_dir}/${qa}.${model}.json

echo "Merged files for model ${model} and QA ${qa}"

model=QwQ-32B
qa=hellaswag
jq -s 'flatten' \
  $(ls ${base_dir}/${qa}.${model}.*.json | sort -V) \
  > ${base_dir}/${qa}.${model}.json

echo "Merged files for model ${model} and QA ${qa}"


model=gpt-oss-20b
qa=mathqa
jq -s 'flatten' \
  $(ls ${base_dir}/${qa}.${model}.*.json | sort -V) \
  > ${base_dir}/${qa}.${model}.json

echo "Merged files for model ${model} and QA ${qa}"


model=gpt-oss-20b
qa=hellaswag
jq -s 'flatten' \
  $(ls ${base_dir}/${qa}.${model}.*.json | sort -V) \
  > ${base_dir}/${qa}.${model}.json

echo "Merged files for model ${model} and QA ${qa}"
