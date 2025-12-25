import os
import random
from typing import Optional
import json
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig

def save_csv(file, data, fieldnames):
    with open(file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def load_csv(file):
    with open(file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def save_json(data, save_path):
  # import json
  with open(save_path, 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(load_path):
  # import json
  with open(load_path, 'r') as f:
    data = json.load(f)
  return data

def save_jsonl(data, save_path):
  # import json
  with open(save_path, 'w') as f:
    for line in data:
      f.write(json.dumps(line, ensure_ascii=False) + '\n')

def load_jsonl(load_path):
  # import json
  with open(load_path, 'r') as f:
    data = [json.loads(line) for line in f]
  return data


def debug_check_model_arch(model) -> None:
  """Check the model architecture and configuration.

  Args:
    model: Model to check.
  """
  print(model.config)
  print()
  print(model)
  print()

  for name, param in model.named_parameters():
    print(name, param.size())
  print()


def initialize_model(model_name, quantize_type, device, hf_token=None):
  # import torch
  # from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
  """Initialize model and tokenizer in HuggingFace.

    Args:
        model_name (str): Model name in HuggingFace.
        quantize_type (str): Quantization type. Choose from "none", "4bit", "8bit", and "half".
        device (torch.device): Device for the model.
        hf_token (Optional[str]): Token for HuggingFace. Defaults to None.

    Returns:
        tuple: Model and tokenizer.
    """
  tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)

  if tokenizer.pad_token is None:
    if tokenizer.eos_token is None:
      tokenizer.add_special_tokens({"pad_token": "<pad>"})
    else:
      tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

  # Flash Attention を完全に無効化
  torch.backends.cuda.enable_flash_sdp(False)

  model_kwargs = {
      "low_cpu_mem_usage": True,
      "trust_remote_code": True,
      "token": hf_token,
      "device_map": "auto",
      "_attn_implementation": 'eager',
  }

  # 量子化タイプによる設定
  if quantize_type == "4bit":
    model_kwargs.update({
        "quantization_config":
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        "torch_dtype":
            torch.float16,  # 明示的に fp16 を指定
    })
  elif quantize_type == "8bit":
    model_kwargs.update({
        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        "use_cache": True,
    })
  elif quantize_type == "half":
    model_kwargs.update({"torch_dtype": torch.float16})  # fp16 に設定
  else:  # デフォルト (non-quantized)
    model_kwargs.update({"torch_dtype": torch.float16})  # 明示的に fp32 に

  # モデルをロード
  model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

  # flash attention 2 の無効化 (configレベル)
  if hasattr(model.config, "use_flash_attention_2"):
    model.config.use_flash_attention_2 = False

  if quantize_type in ["none", "half"]:
    model.to(device)

  model.eval()
  return model, tokenizer


def seed_everything(seed) -> None:
  # import torch
  # import random
  # import os
  # import numpy as np
  """Seed everything
  Args:
    seed (int): Seed value

  """
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True


def get_device() -> str:
  # import torch
  """Choose the device

  This function choose the optimal device for you.

  Returns:
    str: Device name, i.e., "mps", "cuda", or "cpu"
  """
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    major_version = int(torch.__version__.split(".")[0])
    if major_version >= 2:
      return torch.device("mps")
  return torch.device("cpu")
