from transformers import DataCollatorWithPadding


def reward_collator(tokenizer):
    return DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
