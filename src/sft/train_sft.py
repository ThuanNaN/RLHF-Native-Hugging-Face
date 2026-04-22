import argparse
from pathlib import Path

import yaml
from transformers import TrainingArguments, set_seed
from trl import SFTTrainer

from src.utils.dataset import format_sft_dataset, load_json_dataset, train_eval_split
from src.utils.model import build_lora_config, load_causal_lm, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train SFT model with TRL")
    parser.add_argument("--config", default="configs/sft.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    tokenizer = load_tokenizer(cfg["tokenizer_name_or_path"])
    dataset = load_json_dataset(cfg["dataset_path"])
    dataset = format_sft_dataset(dataset)
    split = train_eval_split(dataset, cfg.get("eval_size", 0.0), cfg.get("seed", 42))

    model = load_causal_lm(cfg["model_name_or_path"], use_4bit=cfg.get("use_4bit", False))
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        do_train=cfg.get("do_train", True),
        do_eval=cfg.get("do_eval", False),
        num_train_epochs=cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg.get("learning_rate", 2e-5),
        weight_decay=cfg.get("weight_decay", 0.0),
        warmup_ratio=cfg.get("warmup_ratio", 0.0),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        evaluation_strategy=cfg.get("evaluation_strategy", "no"),
        report_to=cfg.get("report_to", ["none"]),
        bf16=cfg.get("bf16", False),
        fp16=cfg.get("fp16", False),
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split.get("test") if cfg.get("do_eval", False) else None,
        processing_class=tokenizer,
        peft_config=build_lora_config(cfg),
        dataset_text_field="text",
        max_seq_length=cfg.get("max_seq_length", 1024),
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    if "mlflow" in cfg.get("report_to", []):
        import mlflow

        with mlflow.start_run(run_name="sft"):
            mlflow.log_params({k: str(v) for k, v in cfg.items()})
            mlflow.log_artifacts(cfg["output_dir"], artifact_path="model")

    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
