#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

from inspect_qwen_dataset import JsonlConversations, DataCollatorForCausal


class ConsoleLossCallback(TrainerCallback):
    def __init__(self, log_file: str = "") -> None:
        super().__init__()
        self.log_file = log_file
        self._fh = None
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            self._fh = open(self.log_file, "a", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        loss = logs.get("loss", logs.get("train_loss"))
        lr = logs.get("learning_rate")
        msg = f"step={step}"
        if loss is not None:
            msg += f" | loss={loss:.6f}"
        if lr is not None:
            msg += f" | lr={lr:.6e}"
        print(msg, flush=True)
        if self._fh is not None:
            self._fh.write(msg + "\n")
            self._fh.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen (no quantization, reuse dataset utilities)")

    # Data & model
    parser.add_argument("--train_file", type=str, required=True, help="Path to JSON/JSONL or JSON array dataset")
    parser.add_argument("--model_name_or_path", type=str, default="/root/autodl-tmp/qwen3-0_6b", help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./qwen_lora_output")

    # Sequence & tokenizer
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--local_files_only", action="store_true", help="Load tokenizer/model only from local cache")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    # Precision & memory
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Dataloader & seed
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--log_file", type=str, default="", help="Optional file to append plain logs")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    return parser.parse_args()


def build_lora_model(base_model, args: argparse.Namespace):
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    lora_model = get_peft_model(base_model, lora_cfg)
    lora_model.print_trainable_parameters()
    return lora_model


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading base model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    # Optional precision cast (no quantization)
    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)

    # Wrap with LoRA adapters
    model = build_lora_model(model, args)

    # Dataset
    print(f"Loading dataset: {args.train_file}")
    train_dataset = JsonlConversations(args.train_file, tokenizer, args.max_seq_length)

    data_collator = DataCollatorForCausal(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adamw_torch",
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[ConsoleLossCallback(args.log_file)] if args.log_file or True else None,
    )

    trainer.train()

    # Save adapter
    trainer.save_state()
    trainer.save_model(args.output_dir)

    print("Training complete. Adapter saved to:", args.output_dir)


if __name__ == "__main__":
    main() 