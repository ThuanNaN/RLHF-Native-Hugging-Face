import argparse

import torch
import yaml
from datasets import Dataset
from transformers import pipeline, set_seed
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model

from src.utils.dataset import ensure_sft_columns, load_json_dataset
from src.utils.model import load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL policy with PPO")
    parser.add_argument("--config", default="configs/ppo.yaml")
    return parser.parse_args()


def build_query_dataset(dataset: Dataset):
    ensure_sft_columns(dataset)

    def _map(example):
        return {"query": f"{example['instruction']}\n"}

    return dataset.map(_map, remove_columns=dataset.column_names)


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    tokenizer = load_tokenizer(cfg["tokenizer_name_or_path"])
    dataset = build_query_dataset(load_json_dataset(cfg["dataset_path"]))

    ppo_config = PPOConfig(
        model_name=cfg["policy_model_name_or_path"],
        learning_rate=cfg.get("learning_rate", 1e-6),
        batch_size=cfg.get("batch_size", 8),
        mini_batch_size=cfg.get("mini_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        ppo_epochs=cfg.get("ppo_epochs", 4),
        log_with=cfg.get("report_to", ["none"])[0],
        seed=cfg.get("seed", 42),
    )

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg["policy_model_name_or_path"])
    ref_model = create_reference_model(policy_model)

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    reward_pipe = pipeline(
        "text-classification",
        model=cfg["reward_model_name_or_path"],
        tokenizer=tokenizer,
        function_to_apply="none",
        truncation=True,
        device=0 if torch.cuda.is_available() else -1,
    )

    max_steps = cfg.get("max_ppo_steps", 20)
    generation_kwargs = {
        "max_new_tokens": cfg.get("max_new_tokens", 128),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= max_steps:
            break

        query_tensors = [tokenizer(q, return_tensors="pt", truncation=True, max_length=cfg.get("max_input_length", 512))["input_ids"].squeeze(0).to(policy_model.pretrained_model.device) for q in batch["query"]]
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        rewards_raw = reward_pipe(responses)
        rewards = [torch.tensor(float(item[0]["score"])) for item in rewards_raw]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    ppo_trainer.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
