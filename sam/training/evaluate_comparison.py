"""Generate on the held-out split and score with the provisional validator.

This is the "score each against the Phase 2 validator's pass rate on a
held-out set" step from spec Section 1.6. The real Phase 2 validator doesn't
exist yet, so this uses sam/evaluation/quick_score.py — a documented,
provisional stand-in (structural sanity only, not full constraint checking).

Run after tokenize_and_cache.py and train_sam.py have produced
sam/training/comparison_runs/<model_slug>/{held_out.jsonl,checkpoints/latest_adapter}.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sam.evaluation.quick_score import score_file  # noqa: E402

RUNS_ROOT = Path("sam/training/comparison_runs")
MAX_NEW_TOKENS = 1024


def model_slug(base_model: str) -> str:
    return base_model.replace("/", "__")


def generate_one(model, tokenizer, messages: list[dict]) -> str:
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="e.g. Qwen/Qwen3-4B")
    args = parser.parse_args()

    run_dir = RUNS_ROOT / model_slug(args.base_model)
    held_out_path = run_dir / "held_out.jsonl"
    adapter_path = run_dir / "checkpoints" / "latest_adapter"
    generated_path = run_dir / "generated.jsonl"
    results_path = run_dir / "eval_results.json"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    records = []
    with held_out_path.open() as f:
        for line in f:
            records.append(json.loads(line))

    with generated_path.open("w") as f:
        for i, record in enumerate(records, start=1):
            messages = record["messages"]
            generated_text = generate_one(model, tokenizer, messages)
            out_messages = [m for m in messages if m["role"] != "assistant"]
            out_messages.append({"role": "assistant", "content": generated_text})
            f.write(json.dumps({"messages": out_messages}) + "\n")
            print(f"[{args.base_model}] generated {i}/{len(records)}")

    scored = score_file(generated_path)
    results_path.write_text(json.dumps({
        "base_model": args.base_model,
        "pass_rate": scored["pass_rate"],
        "total": scored["total"],
        "passed": scored["passed"],
    }, indent=2))

    print(f"base_model={args.base_model} pass_rate={scored['pass_rate']:.3f} "
          f"({scored['passed']}/{scored['total']})")
    print(f"results -> {results_path}")


if __name__ == "__main__":
    main()
