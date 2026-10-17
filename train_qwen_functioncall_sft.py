#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
import os
import sys
import math
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers import set_seed
from transformers import TrainerCallback

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT for Qwen function-calling with assistant-only loss masking")
    parser.add_argument("--train_file", type=str, required=True, help="Path to JSONL or JSON array file. Each sample has {messages|conversation, optional tools}")
    parser.add_argument("--model_name_or_path", type=str, default="qwen3-4b-instruct", help="Base model (e.g. qwen3-4b-instruct)")
    parser.add_argument("--output_dir", type=str, default="./qwen_tooluse_sft")

    parser.add_argument("--max_seq_length", type=int, default=4096)
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
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_first_step", action="store_true", help="Log on the first step")

    # QLoRA/PEFT
    parser.add_argument("--qlora", action="store_true", help="Enable QLoRA (4-bit) with PEFT")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Dataloader
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Logging file
    parser.add_argument("--log_file", type=str, default="", help="Optional path to append plain-text training logs (loss, lr, step)")

    args = parser.parse_args()
    return args


def _supports_tools_kw(tokenizer: AutoTokenizer) -> bool:
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
        return any(p.name == "tools" for p in sig.parameters.values())
    except Exception:
        return False


class JsonlConversations(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples: List[Dict[str, Any]] = []

        def _normalize_obj(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not isinstance(obj, dict):
                return None
            msgs = None
            if isinstance(obj.get("messages"), list):
                msgs = obj["messages"]
            elif isinstance(obj.get("conversation"), list):
                msgs = obj["conversation"]
            if msgs is None:
                return None
            new_obj: Dict[str, Any] = {"messages": msgs}
            if isinstance(obj.get("tools"), list):
                new_obj["tools"] = obj["tools"]
            return new_obj

        # Try JSON array
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        stripped = content.lstrip()
        parsed_any = False

        if stripped.startswith("["):
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for raw in data:
                        norm = _normalize_obj(raw)
                        if norm is not None:
                            self.examples.append(norm)
                    parsed_any = len(self.examples) > 0
            except Exception:
                parsed_any = False

        if not parsed_any:
            # Fallback to JSONL
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except Exception:
                        continue
                    norm = _normalize_obj(raw)
                    if norm is not None:
                        self.examples.append(norm)

        if len(self.examples) == 0:
            raise ValueError("No valid samples found. Expect JSON array or JSONL with objects containing a 'messages' or 'conversation' list.")

        self._use_tools_kw = _supports_tools_kw(self.tokenizer)

    def __len__(self) -> int:
        return len(self.examples)

    def _apply_template(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], tokenize: bool, return_tensors: Optional[str] = None):
        kwargs = dict(tokenize=tokenize, add_generation_prompt=False)
        if return_tensors is not None:
            kwargs["return_tensors"] = return_tensors
        if tools and self._use_tools_kw:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        messages: List[Dict[str, Any]] = ex["messages"]
        tools: Optional[List[Dict[str, Any]]] = ex.get("tools")

        # Full tokenization once
        try:
            full_ids = self._apply_template(messages, tools, tokenize=True, return_tensors=None)
        except Exception:
            full_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        total_len = len(full_ids)

        # Build assistant-only mask via prefix-delta method (O(n^2) but robust)
        assistant_mask = torch.zeros(total_len, dtype=torch.bool)
        prev_len = 0
        for i, msg in enumerate(messages):
            try:
                prefix_ids = self._apply_template(messages[: i + 1], tools, tokenize=True, return_tensors=None)
            except Exception:
                prefix_ids = self.tokenizer.apply_chat_template(messages[: i + 1], tokenize=True, add_generation_prompt=False)
            cur_len = len(prefix_ids)
            if msg.get("role") == "assistant":
                start, end = prev_len, cur_len
                if end > start:
                    assistant_mask[start:end] = True
            prev_len = cur_len

        # Truncate to max length (keep tail to better supervise long convos)
        if total_len > self.max_seq_length:
            overflow = total_len - self.max_seq_length
            full_ids = full_ids[overflow:]
            assistant_mask = assistant_mask[overflow:]

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[~assistant_mask] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class DataCollatorForCausal:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        if self.pad_to_multiple_of is not None:
            def _pad_to_multiple(t: torch.Tensor, value: int) -> torch.Tensor:
                length = t.size(1)
                pad_len = (self.pad_to_multiple_of - length % self.pad_to_multiple_of) % self.pad_to_multiple_of
                if pad_len == 0:
                    return t
                pad_shape = (t.size(0), pad_len)
                pad_tensor = torch.full(pad_shape, value, dtype=t.dtype, device=t.device)
                return torch.cat([t, pad_tensor], dim=1)

            batch_input_ids = _pad_to_multiple(batch_input_ids, self.tokenizer.pad_token_id)
            batch_attention_mask = _pad_to_multiple(batch_attention_mask, 0)
            batch_labels = _pad_to_multiple(batch_labels, -100)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }


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


def maybe_prepare_qlora_model(model, args):
    if not args.qlora:
        return model
    if not _HAS_PEFT:
        raise RuntimeError("PEFT is not installed. Install peft>=0.10.0 to use QLoRA.")
    if not _HAS_BNB:
        raise RuntimeError("bitsandbytes is not installed. Install bitsandbytes to use 4-bit quantization.")
    model = prepare_model_for_kbit_training(model)
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    load_kwargs = dict(trust_remote_code=True)
    if args.qlora:
        if not _HAS_BNB:
            raise RuntimeError("bitsandbytes is required for QLoRA (install with pip).")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs["device_map"] = "auto"

    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **load_kwargs,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = None
    if torch_dtype is not None and not args.qlora:
        model = model.to(torch_dtype=torch_dtype)

    model = maybe_prepare_qlora_model(model, args)

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
        logging_first_step=args.logging_first_step,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
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

    # Save adapter or full model
    trainer.save_state()
    trainer.save_model(args.output_dir)

    print("Training complete. Model saved to:", args.output_dir)


if __name__ == "__main__":
    main() 