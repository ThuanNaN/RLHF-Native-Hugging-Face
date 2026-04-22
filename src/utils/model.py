from typing import Dict, Optional

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig


def load_tokenizer(tokenizer_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def maybe_quant_config(use_4bit: bool = False):
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(cfg: Dict, task_type: str = "CAUSAL_LM") -> Optional[LoraConfig]:
    if not cfg.get("use_lora", False):
        return None
    return LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules"),
        bias="none",
        task_type=task_type,
    )


def load_causal_lm(
    model_name_or_path: str,
    use_4bit: bool = False,
    trust_remote_code: bool = False,
):
    quantization_config = maybe_quant_config(use_4bit)
    model_kwargs = {"trust_remote_code": trust_remote_code}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)


def load_reward_model(
    model_name_or_path: str,
    use_4bit: bool = False,
    trust_remote_code: bool = False,
):
    quantization_config = maybe_quant_config(use_4bit)
    model_kwargs = {"trust_remote_code": trust_remote_code, "num_labels": 1}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **model_kwargs)
