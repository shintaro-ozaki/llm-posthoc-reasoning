import re
from pathlib import Path
from utils import load_json, save_json
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--correct_output_file', type=str, required=True)
    parser.add_argument('--incorrect_output_file', type=str, required=True)
    return parser.parse_args()

def parse_reasoning_steps(text: str):
    pattern = r"Step\s+(\d+):\s*(.*?)(?=(?:Step\s+\d+:|$))"
    matches = re.findall(pattern, text, flags=re.DOTALL)

    results = []
    for step_num, block in matches:
        reasoning_match = re.search(r"Reasoning:\s*(.*?)(?:- Candidates:|$)", block, flags=re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        # \n- が入っていたらreasoningはその手前までにする
        reasoning = re.split(r"\n-+", reasoning)[0].strip()
        reasoning = re.sub(r"\*\*(.*?)\*\*", r"\1", reasoning)

        candidates_match = re.search(r"Candidates:\s*(.*)", block)
        if candidates_match:
            candidates_raw = candidates_match.group(1).strip()
            candidates = re.split(r"[,\s]+", candidates_raw)
            candidates = [c for c in candidates if c]
        else:
            candidates = []

        results.append({
            "step": int(step_num),
            "reasoning": reasoning,
            "candidates": candidates,
        })

    return results


def main():
    args = parse_args()
    input_path = Path(args.input_file)
    correct_output_path = Path(args.correct_output_file)
    incorrect_output_path = Path(args.incorrect_output_file)

    data = load_json(input_path)
    correct_output = []
    incorrect_output = []
    for i, item in enumerate(data):
        try:
            print(f'====================================================================')
            print(f"[INFO] Processing item {i+1}/{len(data)}")
            reasoning_prompt = item.get("reasoning", "")
            generated_reasoning = reasoning_prompt.split('Step 1:')[-1].strip()
            generated_reasoning = 'Step 1: ' + generated_reasoning
            generated_reasoning = re.sub(r"\*\*(.*?)\*\*", r"\1", generated_reasoning)
            parsed_steps = parse_reasoning_steps(generated_reasoning)

            print(f'Question: {item.get("question", "")}')
            print(f'Candidates: {item.get("candidates", "")}')

            for step in parsed_steps:
                step_no = step["step"]
                print(f"\n---- Step {step_no} ----")
                print(f"Reasoning: {step['reasoning']}")
                print(f"Candidates: {step['candidates']}")

            print(f'Predicted Answer: {item.get("pred", "")}, and Correct Answer: {item.get("gold", "")}')
            if item['pred'] == item['gold']:
                new_item = {
                    "question": item["question"],
                    "candidates": item["candidates"],
                    "gold": item["gold"],
                    "pred": item['pred'],
                    "reasoning": parsed_steps
                }
                correct_output.append(new_item)
            else:
                new_item = {
                    "question": item["question"],
                    "candidates": item["candidates"],
                    "gold": item["gold"],
                    "pred": item['pred'],
                    "reasoning": parsed_steps
                }
                incorrect_output.append(new_item)
        except Exception as e:
            continue

    save_json(correct_output, correct_output_path)
    save_json(incorrect_output, incorrect_output_path)
    print(f'====================================================================')
    print(f"[INFO] Saved parsed JSONL to: {correct_output_path}")
    print(f"[INFO] Input data length: {len(data)}")
    print(f"[INFO] Correct output data length: {len(correct_output)}")
    print(f"[INFO] Incorrect output data length: {len(incorrect_output)}")

if __name__ == "__main__":
    main()
