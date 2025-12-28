import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_device, load_json, save_json, seed_everything

seed_everything(42)
device = get_device()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    return parser.parse_args()


def initialize_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.bfloat16,
        device_map="auto",
    ).to(device)
    model.eval()
    return tokenizer, model


def compute_logprob(model, tokenizer, prompt, choice):
    full = prompt + choice
    inputs = tokenizer(full, return_tensors="pt").to(device)

    prompt_len = len(tokenizer(prompt)["input_ids"])
    input_ids = inputs["input_ids"][0]
    choice_ids = input_ids[prompt_len:]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0][prompt_len - 1 : -1]

    logprobs = F.log_softmax(logits, dim=-1)

    total = 0
    for i, tok in enumerate(choice_ids):
        total += logprobs[i, tok].item()

    return total


def compute_confidence(model, tokenizer, prompt, candidates):
    log_probs = []
    for candidate in candidates:
        logprob = compute_logprob(model, tokenizer, prompt, candidate)
        log_probs.append(logprob)

    max_log_prob = max(log_probs)
    lse = max_log_prob + math.log(
        sum(math.exp(log_prob - max_log_prob) for log_prob in log_probs)
    )

    confidence = {
        candidate: float(math.exp(log_probs[i] - lse))
        for i, candidate in enumerate(candidates)
    }
    return confidence


def build_choices_block(choices):
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    lines = [f"{label}. {choice}" for label, choice in zip(labels, choices)]
    return "\n".join(lines)


def main():
    args = parse_args()
    tokenizer, model = initialize_model(args.model_name)
    input_data = load_json(args.input_file)

    output_data = []

    cop_prompt = "So, the answer is "

    for item in input_data:
        question = item["question"]
        choices = item["candidates"]
        reasoning_steps = item["reasoning"]

        candidates = [chr(ord("A") + i) for i in range(len(choices))]
        new_steps = []

        for step in reasoning_steps:
            prior_steps = " ".join([s["reasoning"] for s in new_steps])

            choices_block = build_choices_block(choices)

            prompt = f"""
    {question}

    Choices:
    {choices_block}

    And now let's think step by step.
    {prior_steps} {cop_prompt}
    """.strip()

            step_confidence = compute_confidence(model, tokenizer, prompt, candidates)

            new_steps.append({**step, "confidence": step_confidence})

            # debug
            print(f"[DEBUG] Prompt:{prompt}")
            print(f"[DEBUG] Choices: {choices}")
            print(f"[DEBUG] Answer in this step : {step['candidates']}")
            print(f"[DEBUG] Step confidence: {step_confidence}")
            print(f"[DEBUG] Correct answer : {item['gold']}")
            print("--------------------------------------------------")

        item["reasoning"] = new_steps
        output_data.append(item)

    save_json(output_data, args.output_file)
    print(f"[INFO] Saved → {args.output_file}")


if __name__ == "__main__":
    main()
