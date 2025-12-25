from pathlib import Path
from utils import load_jsonl

input_file = Path(f'data/model_input/mathqa.jsonl')

data = load_jsonl(input_file)

output_data = []
for i, item in enumerate(data):
    question = item['question']
    choices = item['choices']
    answer = item['answer']
    reasoning_steps = item['reasoning']

    print(f'Question {i+1}: {question}')
    print('Choices:')
    for idx, choice in enumerate(choices):
        print(f'  ({chr(65 + idx)}) {choice}')
    print(f'Answer: {answer}')
    print('Reasoning steps:')
    for step in reasoning_steps:
        print(f"  Step {step['step']}: {step['reasoning']}")
        print(f"    Candidates: {step['candidates']}")
    printr(f'Pred')
    print()
