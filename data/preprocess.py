import argparse
import json
from pathlib import Path
from typing import Dict, Iterable


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_sft(rows: Iterable[Dict]):
    for row in rows:
        instruction = row.get("instruction") or row.get("prompt")
        response = row.get("response") or row.get("chosen")
        if instruction and response:
            yield {"instruction": instruction, "response": response}


def to_preference(rows: Iterable[Dict]):
    for row in rows:
        prompt = row.get("prompt") or row.get("instruction")
        chosen = row.get("chosen") or row.get("response")
        rejected = row.get("rejected")
        if prompt and chosen and rejected:
            yield {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw JSONL data to SFT/preference schemas")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--task", choices=["sft", "preference"], required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = read_jsonl(input_path)
    converted = to_sft(rows) if args.task == "sft" else to_preference(rows)
    write_jsonl(output_path, converted)


if __name__ == "__main__":
    main()
