import argparse
import logging
from pathlib import Path
import gc

import numpy as np
import torch
from tqdm import tqdm

from utils import (
    seed_everything,
    get_device,
    load_jsonl,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = get_device()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def initialize_model(model_name, use_4bit=False, device=None):
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def take_token(hidden_state: torch.Tensor, token_idx: int):
    """
    hidden_state: (B, T, H) or (T, H) or (H,)
    """
    if hidden_state is None:
        return None
    if hidden_state.dim() == 3:
        return hidden_state[:, token_idx, :]
    elif hidden_state.dim() == 2:
        return hidden_state[token_idx, :]
    else:
        return hidden_state


def find_token_index(input_ids: torch.Tensor, tokenizer, token_str: str):
    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
    ids = input_ids[0].tolist()

    logger.info(f"Token str: {token_str}, token ids: {token_ids}")
    for i in range(len(ids) - len(token_ids) + 1):
        if ids[i : i + len(token_ids)] == token_ids:
            return i
    return None


def save_activations_npy(
    out_dir: Path,
    idx: int,
    activations: dict,
    token_idx: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # hidden states
    hidden_states = np.stack(
        [
            take_token(h, token_idx).detach().cpu().float().numpy()
            for h in activations["hidden_states"]
        ],
        axis=0,
    )
    np.save(out_dir / f"{idx}_hidden_states.npy", hidden_states)

    # attention
    attn = [
        take_token(x, token_idx).cpu().numpy()
        for x in activations["attn_output"]
        if x is not None
    ]
    if len(attn) > 0:
        np.save(out_dir / f"{idx}_attn_output.npy", np.stack(attn, axis=0))

    # mlp
    mlp = [
        take_token(x, token_idx).cpu().numpy()
        for x in activations["mlp_output"]
        if x is not None
    ]
    if len(mlp) > 0:
        np.save(out_dir / f"{idx}_mlp_output.npy", np.stack(mlp, axis=0))


def build_collector(model, model_name: str, dtype):
    if "DeepSeek-R1-0528-Qwen3-8B" in model_name:
        from activation_collector import Qwen3ActivationCollector

        return Qwen3ActivationCollector(model, dtype=dtype)

    elif "DeepSeek-R1-Distill-Qwen-7B" in model_name:
        from activation_collector import Qwen2ActivationCollector

        return Qwen2ActivationCollector(model, dtype=dtype)

    elif (
        "DeepSeek-R1-Distill-Llama-8B" in model_name
        or "Llama-3.1-Nemotron-Nano-8B-v1" in model_name
    ):
        from activation_collector import LlamaActivationCollector

        return LlamaActivationCollector(model, dtype=dtype)

    elif "microsoft/Phi-4-reasoning" in model_name:
        from activation_collector import Phi3ActivationCollector

        return Phi3ActivationCollector(model, dtype=dtype)

    elif "openai/gpt-oss-20b" in model_name:
        from activation_collector import GptOssActivationCollector

        return GptOssActivationCollector(model, dtype=dtype)

    else:
        raise ValueError(f"No ActivationCollector defined for model: {model_name}")


def build_prompt(item: dict) -> str:
    question = item.get("question", "")
    candidates = item.get("candidates", [])
    reasoning = item.get("reasoning", "")
    pred = item.get("pred", "")

    cand_text = ""
    if isinstance(candidates, list) and len(candidates) > 0:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines = []
        for i, c in enumerate(candidates):
            lab = letters[i] if i < len(letters) else str(i)
            lines.append(f"{lab}. {c}")
        cand_text = "\nOPTIONS:\n" + "\n".join(lines)

    prompt = (
        f"{question}\n"
        f"{cand_text}\n\n"
        f"<think> \n{reasoning}\n</think> \n\n"
        f"So, the answer is {pred}. Now I will rate my confidence on a scale of 1-10.\n"
        f"Please generate only the score. Proposed confidence: "
    )
    return prompt


@torch.no_grad()
def run_force_decoding(model, tokenizer, prompt: str, collector):
    model.eval()

    max_len = min(tokenizer.model_max_length, model.config.max_position_embeddings)

    logger.info(f"Max length: {max_len}")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(model.device)

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    return {
        "input_ids": inputs["input_ids"],
        "hidden_states": outputs.hidden_states,
        "attn_output": collector.attn_output,
        "mlp_output": collector.mlp_output,
        "projection_output": collector.proj_output,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = get_device()
    dtype = torch.bfloat16

    model, tokenizer = initialize_model(
        args.model_name, use_4bit=args.use_4bit, device=device
    )

    data = load_jsonl(args.input_file)
    if args.end == -1 or args.end > len(data):
        args.end = len(data)

    data = data[args.start : args.end]

    if args.debug:
        logger.info("Debug mode: using first 1000 samples")
        data = data[:1000]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in tqdm(list(enumerate(data, start=args.start)), total=len(data)):
        activations = None
        try:
            base_dir = args.output_dir
            if (
                (base_dir / "last_token" / f"{idx}_hidden_states.npy").exists()
                and (base_dir / "think_tag" / f"{idx}_hidden_states.npy").exists()
                and (base_dir / "end_think_tag" / f"{idx}_hidden_states.npy").exists()
            ):
                logger.info(f"Skip idx {idx} (already exists)")
                continue

            prompt = build_prompt(item)
            collector = build_collector(model, args.model_name, dtype)
            collector.register()
            try:
                # heavy CPU allocation (model forward, numpy stack, etc.)
                activations = run_force_decoding(model, tokenizer, prompt, collector)
            except MemoryError:
                logger.info(f"CPU OOM at idx {idx}")
                item["error"] = "CPU_OOM"
                if activations is not None:
                    del activations
                gc.collect()
                continue
            collector.remove()

            input_ids = activations["input_ids"]
            think_idx = find_token_index(input_ids, tokenizer, "<think>")
            end_think_idx = find_token_index(input_ids, tokenizer, "</think>")
            last_idx = -1
            logger.info(f"Think idx: {think_idx}, input_ids shape: {input_ids.shape}")
            logger.info(
                f"End think idx: {end_think_idx}, input_ids shape: {input_ids.shape}"
            )

            if think_idx is not None:
                save_activations_npy(
                    base_dir / "think_tag",
                    idx,
                    activations,
                    think_idx,
                )

            if end_think_idx is not None:
                save_activations_npy(
                    base_dir / "end_think_tag",
                    idx,
                    activations,
                    end_think_idx,
                )

            save_activations_npy(
                base_dir / "last_token",
                idx,
                activations,
                last_idx,
            )
            item["activation_dir"] = str(base_dir.resolve())
            del activations["hidden_states"]
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            logger.exception(f"OOM at idx {idx}")
            item["error"] = "OOM"
            torch.cuda.empty_cache()
            continue

        except Exception as e:
            logger.exception(f"Error at idx {idx}")
            item["error"] = str(e)
            continue


if __name__ == "__main__":
    main()
