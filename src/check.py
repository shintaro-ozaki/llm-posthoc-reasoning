from pathlib import Path

from utils import load_json

# model_name="Llama-3.3-70B-Instruct"
model_name = "Qwen2.5-72B-Instruct"
# model_name="Phi-4-reasoning"


qa = "mathqa"
# qa = "helladswag"

correct_or_incorrect = "correct"

input_file = Path(
    f"data/reasoning_datasets_after_split/{qa}.{model_name}.{correct_or_incorrect}.json"
)
data = load_json(input_file)

for i, item in enumerate(data):
    question = item["question"]
    candidates = item["candidates"]
    gold = item["gold"]
    generated_reasoning = item["reasoning"]

    print(f"Question {i + 1}: {question}")
    print("Choices:")
    for idx, candidate in enumerate(candidates):
        print(f"  ({chr(65 + idx)}) {candidate}")
    print(f"Answer: {gold}")

    print("Generated Reasoning Steps:")
    for step_info in generated_reasoning:
        step_num = step_info["step"]
        reasoning_text = step_info["reasoning"]
        print(f"  Step {step_num}: {reasoning_text}")
    print("---")


# qa="mathqa"
# qa="hellaswag"

# print(f'Loading data for model: {model_name}, dataset: {qa}')

# input_file = Path(f'data/reasoning_datasets_before_split/{qa}.{model_name}.json')
# input_data = load_json(input_file)

# def build_qa_block(item):
#     question = item["question"].strip()
#     candidates = item["candidates"]

#     alphabet = ["A", "B", "C", "D", "E"][:len(candidates)]
#     candidates_str = " ".join(
#         f"({alphabet[i]}) {candidates[i]}" for i in range(len(candidates))
#     )

#     return (
#         f"Question: {question}\n"
#         f"Candidates: {candidates_str}\n"
#         f"Answer:"
#     )


# for i, item in enumerate(input_data):
#     question = item['question']
#     choices = item['candidates']
#     answer = item['gold']
#     pred = item['pred']
#     generated_reasoning = item['reasoning']
#     qa_block = build_qa_block(item)

#     print(f'===================ここから========================')
#     print('Generated Reasoning Steps:')
#     generated_reasoning = generated_reasoning.split('Step 1:')[-1].strip()
#     generated_reasoning = 'Step 1: ' + generated_reasoning
#     print(generated_reasoning)
#     print('===================ここまで========================')

# accuracy = sum(1 for item in input_data if item['gold'].strip().upper() == item['pred'].strip().upper()) / len(input_data)
# print(f'Accuracy: {accuracy:.4f} ({sum(1 for item in input_data if item["gold"].strip().upper() == item["pred"].strip().upper())} / {len(input_data)})')
