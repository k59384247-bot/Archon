"""Resumable QLoRA fine-tune for one base-model comparison candidate.

Adapted from spec Sections 1.4-1.5 for the Section 1.6 comparison:
  - BASE_MODEL is a CLI arg instead of a hardcoded constant.
  - Reads the tokenized dataset from and writes checkpoints under
    sam/training/comparison_runs/<model_slug>/, matching what
    tokenize_and_cache.py just produced for this candidate.
  - Same resumable-checkpoint behaviour as the full-scale harness (time
    budget stop, adapter-only saves, is_trainable=True on resume) — a
    comparison run is small, but Kaggle can still kill the session mid-run.
"""

import argparse
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainerCallback, TrainerControl, TrainerState, TrainingArguments,
)
from trl import SFTTrainer

RUNS_ROOT = Path("sam/training/comparison_runs")
SESSION_TIME_BUDGET_SECONDS = 7 * 3600 - 15 * 60  # stop 15 min before Kaggle's 7-hour limit


def model_slug(base_model: str) -> str:
    return base_model.replace("/", "__")


class TimeBudgetCallback(TrainerCallback):
    """Forces a clean checkpoint + stop before the environment kills the process."""

    def __init__(self, budget_seconds: int):
        self.budget_seconds = budget_seconds
        self.start_time = time.time()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed = time.time() - self.start_time
        if elapsed >= self.budget_seconds:
            control.should_save = True
            control.should_training_stop = True
        return control


def load_base_model_4bit(base_model: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # Pin the whole model to a single GPU rather than "auto" -- letting accelerate
    # split layers across both T4s triggers a device-mismatch bug in trl's chunked
    # cross-entropy loss. batch_size=1 + gradient checkpointing keep it under 16GB
    # on one card anyway, so there's no need for the split.
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map={"": 0},
    )
    return model


def get_lora_config() -> LoraConfig:
    # Exactly matches Bible Section 3.3: r=16, alpha=16, all projection layers.
    return LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def find_latest_checkpoint(checkpoint_dir: Path) -> str | None:
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="e.g. Qwen/Qwen3-4B")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    run_dir = RUNS_ROOT / model_slug(args.base_model)
    checkpoint_dir = run_dir / "checkpoints"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(str(run_dir / "tokenized_dataset"))

    resume_path = find_latest_checkpoint(checkpoint_dir)

    base_model = load_base_model_4bit(args.base_model)

    if resume_path:
        # Resume: load adapter as trainable, not frozen — the #1 PEFT+Trainer resume gotcha.
        print(f"Resuming from {resume_path}")
        model = PeftModel.from_pretrained(base_model, resume_path, is_trainable=True)
    else:
        print("Starting fresh run")
        model = get_peft_model(base_model, get_lora_config())

    # Gradient checkpointing trades compute for activation memory -- needed to fit
    # an 8B model's training activations on a single 16GB T4. enable_input_require_grads()
    # is required alongside it for a quantized base model + LoRA, otherwise backprop
    # fails with "element 0 of tensors does not require grad".
    model.enable_input_require_grads()

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,          # bound disk usage — old checkpoints auto-pruned
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        bf16=True,
        report_to=[],                 # wire to LLM tracing tool once Phase 5 sets it up
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        callbacks=[TimeBudgetCallback(SESSION_TIME_BUDGET_SECONDS)],
    )

    trainer.train(resume_from_checkpoint=resume_path)

    # Final adapter-only save for this run (small — tens of MB, fast to commit).
    trainer.save_model(str(checkpoint_dir / "latest_adapter"))
    print(f"base_model={args.base_model} adapter -> {checkpoint_dir / 'latest_adapter'}")


if __name__ == "__main__":
    main()
