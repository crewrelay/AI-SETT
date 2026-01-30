#!/usr/bin/env python3
"""LoRA fine-tuning script for Nemotron-3-Nano-30B-A3B-FP8.

Targets the hybrid Mamba+Transformer architecture with LoRA adapters
on attention projection layers. Uses TRL's SFTTrainer with our
AI-SETT training data in OpenAI chat format.

Usage:
    python3 train_lora.py                          # Train with defaults
    python3 train_lora.py --epochs 5               # Custom epochs
    python3 train_lora.py --rank 32                # Higher LoRA rank
    python3 train_lora.py --dataset balanced        # Use balanced set
    python3 train_lora.py --dry-run                # Test loading only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR.parent.parent / "models" / "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
OUTPUT_DIR = SCRIPT_DIR.parent.parent / "models" / "nemotron-30b-aisett-lora"

# Training data options
DATASETS = {
    "full": SCRIPT_DIR / "all_training_data.jsonl",
    "balanced": SCRIPT_DIR / "balanced_training_data.jsonl",
    "tagged": SCRIPT_DIR / "tagged_training_data.jsonl",
}

# LoRA target modules for hybrid Mamba+Transformer architecture
# Attention layers: q/k/v/o projections
# Mamba layers: in/out projections
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "in_proj", "out_proj",                     # Mamba SSM
]


def load_training_data(dataset_path):
    """Load JSONL training data into HuggingFace Dataset."""
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages", [])
            if len(messages) >= 2:
                examples.append({"messages": messages})

    print(f"Loaded {len(examples)} training examples from {dataset_path.name}")
    return Dataset.from_list(examples)


def format_chat(example, tokenizer):
    """Format messages using the model's chat template."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,  # Disable thinking for training
    )


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Nemotron-30B-A3B-FP8")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR),
                        help="Path to model directory")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for LoRA adapters")
    parser.add_argument("--dataset", type=str, default="full",
                        choices=list(DATASETS.keys()),
                        help="Which dataset to use (default: full)")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Custom dataset path (overrides --dataset)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="Max sequence length (default: 2048)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model and data but don't train")
    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output_dir

    # Resolve dataset
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = DATASETS[args.dataset]

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    if not Path(model_dir).exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        print("Download with: huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
        sys.exit(1)

    print("=" * 60)
    print("NEMOclaude LoRA Fine-Tuning")
    print("=" * 60)
    print(f"Model:    {model_dir}")
    print(f"Dataset:  {dataset_path}")
    print(f"Output:   {output_dir}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch_size} x {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective")
    print(f"LR:       {args.lr}")
    print(f"LoRA:     rank={args.rank}, alpha={args.alpha}")
    print(f"Max seq:  {args.max_seq_len}")
    print(f"Device:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")
    print()

    # --- Load tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load dataset ---
    print("Loading training data...")
    dataset = load_training_data(dataset_path)

    # Format examples using chat template
    print("Formatting with chat template...")
    formatted = []
    skipped = 0
    for ex in dataset:
        try:
            text = format_chat(ex, tokenizer)
            formatted.append({"text": text})
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Warning: skipped example: {e}")

    if skipped:
        print(f"  Skipped {skipped}/{len(dataset)} examples due to formatting errors")

    train_dataset = Dataset.from_list(formatted)
    print(f"Training set: {len(train_dataset)} formatted examples")

    # Show a sample
    if len(train_dataset) > 0:
        sample = train_dataset[0]["text"]
        print(f"\n--- Sample (first 500 chars) ---")
        print(sample[:500])
        print("---\n")

    # --- Load model ---
    print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # BF16 avoids CPU half-precision kernel issues
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # --- Print model structure for LoRA target identification ---
    print("\nModel modules (looking for LoRA targets):")
    target_found = set()
    for name, module in model.named_modules():
        for target in LORA_TARGET_MODULES:
            if name.endswith(target):
                target_found.add(target)
                break
    print(f"  Found target modules: {sorted(target_found)}")
    missing = set(LORA_TARGET_MODULES) - target_found
    if missing:
        print(f"  WARNING: Missing targets: {sorted(missing)}")
        # Filter to only found targets
        active_targets = sorted(target_found)
    else:
        active_targets = LORA_TARGET_MODULES

    # --- LoRA config ---
    print(f"\nConfiguring LoRA (rank={args.rank}, alpha={args.alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=active_targets,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.dry_run:
        print("\n--- DRY RUN COMPLETE ---")
        print("Model and data loaded successfully. Exiting without training.")
        return

    # --- Training arguments ---
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="none",
        dataloader_pin_memory=False,  # Better for unified memory (Jetson)
        optim="adamw_torch",
    )

    # --- Trainer ---
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
    )

    # --- Train ---
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    train_result = trainer.train()

    # --- Save ---
    print("\nSaving LoRA adapters...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    metrics = train_result.metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"LoRA adapters saved to: {output_dir}")
    print(f"Training loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Training runtime: {metrics.get('train_runtime', 'N/A'):.1f}s")
    print(f"\nTo merge and test:")
    print(f"  python3 merge_lora.py --base {model_dir} --lora {output_dir}")


if __name__ == "__main__":
    main()
