import argparse
import random
import logging
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.cuda import OutOfMemoryError
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from prompts import convert_reasoning_prompt
from utils import load_json, save_json, seed_everything

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=Path, default=Path("data/MathQA/test.json")
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--debug_sample_num", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    return parser.parse_args()


def initialize_model(model_name, use_4bit):
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model.eval()
    return model, tokenizer


def build_qa_block(item):
    question = item["question"].strip()
    candidates = item["candidates"]

    alphabet = ["A", "B", "C", "D", "E"][: len(candidates)]
    candidates_str = " ".join(
        f"({alphabet[i]}) {candidates[i]}" for i in range(len(candidates))
    )

    return f"Question: {question}\nCandidates: {candidates_str}\nAnswer:"


@torch.no_grad()
def generate_reasoning(
    model,
    tokenizer,
    prompt_text: str,
    seed: int,
):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        max_new_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
        presence_penalty=0.3,
        do_sample=True,
        eos_token_id=tokenizer.convert_tokens_to_ids("</think>"),
    )

    torch.manual_seed(seed)

    outputs = model.generate(
        **inputs,
        generation_config=generation_config,
    )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return decoded


@torch.no_grad()
def force_decode_choice(
    model,
    tokenizer,
    prompt_text: str,
    choices: list[str],
):
    device = model.device

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = model(**inputs)

    next_logits = outputs.logits[0, -1]

    choice_token_ids = {}
    for c in choices:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Choice '{c}' is not a single token: {ids}")
        choice_token_ids[c] = ids[0]

    cand_ids = torch.tensor(
        [choice_token_ids[c] for c in choices],
        device=device,
    )

    cand_logits = next_logits[cand_ids]
    probs = torch.softmax(cand_logits, dim=0)

    choice_probs = {c: probs[i].item() for i, c in enumerate(choices)}
    pred_choice = max(choice_probs, key=choice_probs.get)

    return pred_choice, choice_probs


def main():
    args = parse_args()
    seed_everything(args.seed)

    model, tokenizer = initialize_model(
        args.model_name,
        args.use_4bit,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    input_data = load_json(args.input_file)
    if args.start_idx != 0 or args.end_idx != -1:
        input_data = input_data[args.start_idx : args.end_idx]

    if args.debug:
        random.seed(args.seed)
        input_data = random.sample(
            input_data,
            min(len(input_data), args.debug_sample_num),
        )

    output_data = []

    logger.info("Running Hugging Face inference")

    for item in tqdm(input_data):
        try:
            qa_block = build_qa_block(item)
            prompt = convert_reasoning_prompt.format(qa_block=qa_block)
            full_text = generate_reasoning(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                seed=args.seed,
            )

            reasoning_text = (
                full_text.replace(prompt, "").replace("</think>", "").strip()
            )
            logger.info(f"Generated reasoning:\n{reasoning_text}")

            alphabet = ["A", "B", "C", "D", "E"][: len(item["candidates"])]

            force_prompt = prompt + reasoning_text + "\nSo, the answer is "

            pred, choice_probs = force_decode_choice(
                model=model,
                tokenizer=tokenizer,
                prompt_text=force_prompt,
                choices=alphabet,
            )

            output_data.append(
                {
                    "question": item["question"],
                    "candidates": item["candidates"],
                    "gold": item["gold"].strip().upper(),
                    "reasoning": reasoning_text,
                    "pred": pred,
                    "choice_probs": choice_probs,
                }
            )
        except OutOfMemoryError:
            logger.error("OutOfMemoryError: Skipping this example.")
            continue
        except Exception as e:
            logger.error(f"Error: {e}. Skipping this example.")
            continue

    save_json(output_data, args.output_file)
    logger.info(f"[OK] Saved {len(output_data)} samples to {args.output_file}")


if __name__ == "__main__":
    main()
