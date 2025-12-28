from pathlib import Path

from utils import load_jsonl, save_jsonl

numbers = [9, 11, 13, 25, 32, 38, 45, 53, 80, 86, 121, 128]


input_file = Path("data/reasoning_datasets/mathqa.jsonl")
output_file = Path("data/reasoning_datasets/mathqa_sample10.jsonl")

data = load_jsonl(input_file)

output_data = []
for i, item in enumerate(data):
    # numbersの行数のみを取得する
    if i + 1 not in numbers:
        continue
    question = item["question"]
    choices = item["choices"]
    answer = item["answer"]
    generated_reasoning = item["reasoning_prompt"]

    print(f"Question {i + 1}: {question}")
    print("Choices:")
    for idx, choice in enumerate(choices):
        print(f"  ({chr(65 + idx)}) {choice}")
    print(f"Answer: {answer}")

    print("Generated Reasoning Steps:")
    print(generated_reasoning)
    print("---")
    output_data.append(
        {
            "question": question,
            "choices": choices,
            "answer": answer,
            "reasoning_prompt": generated_reasoning,
        }
    )

save_jsonl(output_data, output_file)
