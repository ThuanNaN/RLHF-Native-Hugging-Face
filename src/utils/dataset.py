from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{response}"


def load_json_dataset(path: str, split: str = "train") -> Dataset:
    return load_dataset("json", data_files={split: path})[split]


def ensure_sft_columns(dataset: Dataset) -> Dataset:
    required = {"instruction", "response"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"SFT dataset missing columns: {sorted(missing)}")
    return dataset


def ensure_preference_columns(dataset: Dataset) -> Dataset:
    required = {"prompt", "chosen", "rejected"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Preference dataset missing columns: {sorted(missing)}")
    return dataset


def format_sft_dataset(dataset: Dataset, template: str = DEFAULT_PROMPT_TEMPLATE) -> Dataset:
    ensure_sft_columns(dataset)

    def _format(example):
        return {"text": template.format(**example)}

    return dataset.map(_format)


def format_reward_dataset(dataset: Dataset) -> Dataset:
    ensure_preference_columns(dataset)

    def _format(example):
        return {
            "chosen": f"{example['prompt']}\n{example['chosen']}",
            "rejected": f"{example['prompt']}\n{example['rejected']}",
        }

    return dataset.map(_format)


def train_eval_split(dataset: Dataset, eval_size: float = 0.0, seed: int = 42) -> DatasetDict:
    if eval_size <= 0:
        return DatasetDict({"train": dataset})
    return dataset.train_test_split(test_size=eval_size, seed=seed)
