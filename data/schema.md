# Dataset Schema

## SFT format (`instruction` / `response`)
JSONL rows:

```json
{"instruction": "Explain RLHF in one sentence", "response": "RLHF aligns models to human preferences using feedback-driven optimization."}
```

## Preference format (`prompt`, `chosen`, `rejected`)
JSONL rows:

```json
{"prompt": "Write a haiku about code", "chosen": "Clean loops in moonlight...", "rejected": "I do not know."}
```

Both schemas are converted to Hugging Face `datasets.Dataset` objects in `src/utils/dataset.py`.
