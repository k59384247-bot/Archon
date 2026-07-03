"""Tokenize the Section 1.6 base-model comparison subset and cache to disk.

Adapted from spec Section 1.4 for the base-model comparison (Section 1.6):
  - BASE_MODEL is a CLI arg instead of a hardcoded constant, so this can be
    run once per candidate model.
  - Reads sam/data/processed/comparison_subset_500.jsonl instead of the full
    training_pairs.jsonl.
  - Splits the raw (untokenized) records first, then tokenizes both splits.
    The held-out split is also written back out as raw JSONL
    (held_out.jsonl) so evaluate_comparison.py can build prompts from it
    without re-deriving the split.
  - Everything is written under sam/training/comparison_runs/<model_slug>/
    so the three candidate runs never collide.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

MAX_LEN = 2048
COMPARISON_SUBSET = "sam/data/processed/comparison_subset_500.jsonl"
RUNS_ROOT = Path("sam/training/comparison_runs")


def model_slug(base_model: str) -> str:
    return base_model.replace("/", "__")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="e.g. Qwen/Qwen3-4B")
    parser.add_argument("--input", default=COMPARISON_SUBSET)
    parser.add_argument("--test-size", type=float, default=0.1)
    args = parser.parse_args()

    run_dir = RUNS_ROOT / model_slug(args.base_model)
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=args.input, split="train")
    split = ds.train_test_split(test_size=args.test_size, seed=42)

    # Held-out raw messages, for evaluate_comparison.py to generate against later.
    held_out_path = run_dir / "held_out.jsonl"
    with held_out_path.open("w") as f:
        for row in split["test"]:
            f.write(json.dumps({"messages": row["messages"]}) + "\n")

    def tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        tokens = tokenizer(text, truncation=True, max_length=MAX_LEN, padding=False)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = split.map(tokenize, remove_columns=split["train"].column_names)
    tokenized_out = run_dir / "tokenized_dataset"
    tokenized.save_to_disk(str(tokenized_out))

    print(f"base_model={args.base_model}")
    print(tokenized)
    print(f"tokenized dataset -> {tokenized_out}")
    print(f"held-out raw jsonl -> {held_out_path}")


if __name__ == "__main__":
    main()
