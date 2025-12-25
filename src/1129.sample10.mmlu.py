from pathlib import Path
from utils import load_jsonl, save_jsonl

categories = ["high_school_mathematics", "abstract_algebra"]

high_school_mathematics_numbers = [1, 5, 12, 15, 20, 26, 29, 31, 38, 42]
algebra_numbers = [3, 7, 8, 10, 11, 12, 15, 20, 27, 32]


for category in categories:
    input_file = Path(f'data/reasoning_datasets/{category}.jsonl')
    output_file = Path(f'data/reasoning_datasets/{category}_sample10.jsonl')

    data = load_jsonl(input_file)

    output_data = []
    for i, item in enumerate(data):
        # numbersの行数のみを取得する
        if i+1 not in (high_school_mathematics_numbers if category == "high_school_mathematics" else algebra_numbers):
            continue
        question = item['question']
        choices = item['choices']
        answer = item['answer']
        generated_reasoning = item['reasoning_prompt']

        print(f'Question {i+1}: {question}')
        print('Choices:')
        for idx, choice in enumerate(choices):
            print(f'  ({chr(65 + idx)}) {choice}')
        print(f'Answer: {answer}')

        print('Generated Reasoning Steps:')
        print(generated_reasoning)
        print('---')
        output_data.append({
            'question': question,
            'choices': choices,
            'answer': answer,
            'reasoning_prompt': generated_reasoning
        })

    save_jsonl(output_data, output_file)
    print(f'===================================================')
