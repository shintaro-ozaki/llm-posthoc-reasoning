import argparse
from pathlib import Path

from utils import load_jsonl, save_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_file
    output_path = args.output_file

    data = load_jsonl(input_path)
    formatted_data = []

    for i, item in enumerate(data):
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]

        for step in item["reasoning"]:
            row = {
                "question": question,
                "choicesA": choices[0],
                "choicesB": choices[1],
                "choicesC": choices[2],
                "choicesD": choices[3],
                "choicesE": choices[4] if len(choices) == 5 else "",
                "answer": answer,
                "reasoning_step": step["reasoning"],
                "annotate_here": "",
            }
            if len(choices) != 5:
                # choicesEを削除
                row.pop("choicesE")
            formatted_data.append(row)
            if step["candidates"] == [answer]:
                break
    save_csv(output_path, formatted_data, fieldnames=formatted_data[0].keys())
    print(f"Formatted data saved to {output_path}")
