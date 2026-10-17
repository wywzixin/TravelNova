#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
from typing import Dict, List, Any, Tuple

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRL SFT LoRA training for Qwen3 FC ability")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-4B-Instruct",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/root/autodl-tmp/merged_for_training_prompted.json",
        help="JSON or JSONL dataset path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/qwen3-4b-fc-sft",
        help="Where to save the adapter/finetuned model.",
    )

    # training hyper-params
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=0)

    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--packing", action="store_true")

    # dataset options
    parser.add_argument("--min_assistant_chars", type=int, default=1, help="Filter out samples whose assistant content shorter than this.")
    parser.add_argument("--show_samples", type=int, default=0, help="Preview N normalized samples and exit if --exit_after_preview is set.")
    parser.add_argument("--exit_after_preview", action="store_true", help="Exit after preview without training.")

    # quantization & lora
    parser.add_argument("--use_4bit", dest="use_4bit", action="store_true")
    parser.add_argument("--no_4bit", dest="use_4bit", action="store_false")
    parser.set_defaults(use_4bit=False)
    parser.add_argument("--bnb_compute_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--response_template",
        type=str,
        default="<|im_start|>assistant",
        help="Token/string used to identify assistant response start for label masking.",
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Whether to merge LoRA into base model weights when saving.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )

    return parser.parse_args()


def str_to_dtype(name: str):
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    return getattr(torch, name)


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    # Qwen chat prefers right padding
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    return tokenizer


def load_model(model_name_or_path: str, use_4bit: bool, bnb_compute_dtype: str) -> AutoModelForCausalLM:
    quantization_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    if use_4bit:
        compute_dtype = str_to_dtype(bnb_compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )
    return model


def detect_dataset_format(sample: Dict[str, Any]) -> str:
    if isinstance(sample, dict):
        if "messages" in sample:
            return "messages"
        if "text" in sample:
            return "text"
        if "prompt" in sample and "completion" in sample:
            return "prompt_completion"
        if "input" in sample and ("output" in sample or "label" in sample):
            return "input_output"
    return "unknown"


def _maybe_json_unescape(value: Any, max_rounds: int = 2) -> Any:
    # Try to decode nested JSON strings up to max_rounds
    if not isinstance(value, str):
        return value
    out = value
    for _ in range(max_rounds):
        try:
            parsed = json.loads(out)
        except Exception:
            break
        if isinstance(parsed, (dict, list)):
            return parsed
        if isinstance(parsed, str) and parsed != out:
            out = parsed
            continue
        break
    return out


def _stringify_content(content: Any) -> str:
    # Ensure content is a string for chat template
    content = _maybe_json_unescape(content)
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _normalize_role(role: Any) -> str:
    r = str(role).strip().lower() if role is not None else "user"
    if r in {"system", "user", "assistant", "tool"}:
        return r
    # common variants/typos
    if r in {"toll", "tools", "function", "functions", "tool_output", "observation", "toolresult", "tool_result"}:
        return "tool"
    if r in {"assistant_tool", "assistant_function", "function_call"}:
        return "assistant"
    return "user"


def build_messages_from_sample(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    # 1) Direct messages
    if isinstance(sample, dict) and isinstance(sample.get("messages"), list):
        msgs = []
        for m in sample["messages"]:
            role = _normalize_role(m.get("role", "user"))
            content = _stringify_content(m.get("content", ""))
            # Optional: append tool name into content if present
            name = m.get("name") or m.get("tool_name") or m.get("function_name")
            if role == "tool" and name:
                if isinstance(content, str) and content.strip():
                    content = f"{name}: {content}"
                else:
                    content = str(name)
            msgs.append({"role": role, "content": content})
        return msgs

    # 2) Common supervised keys
    fields = {k.lower(): k for k in sample.keys()} if isinstance(sample, dict) else {}
    def g(name: str, default: Any = None):
        key = fields.get(name.lower())
        return sample.get(key, default) if key is not None else default

    instruction = g("instruction") or g("task") or g("query") or g("question")
    user_input = g("input") or g("prompt") or g("user") or g("context")
    assistant_out = g("completion") or g("output") or g("response") or g("answer") or g("assistant") or g("label")

    # Handle content/result special case
    content_field = g("content")
    result_field = g("result")
    tool_ctx = g("tool") or g("tool_message") or g("tool_output")

    # Try to unescape nested json on these fields
    if isinstance(content_field, str):
        content_field = _maybe_json_unescape(content_field)
    if isinstance(result_field, str):
        result_field = _maybe_json_unescape(result_field)
    if isinstance(assistant_out, str):
        assistant_out = _maybe_json_unescape(assistant_out)
    if isinstance(tool_ctx, str):
        tool_ctx = _maybe_json_unescape(tool_ctx)

    # If content is an object like {"result": ...} unwrap
    if isinstance(content_field, dict) and "result" in content_field and assistant_out is None:
        assistant_out = content_field["result"]

    # Prefer explicit assistant_out; otherwise fallback to result/content
    assistant_text = assistant_out if assistant_out is not None else (result_field if result_field is not None else content_field)
    user_text = instruction if instruction is not None else user_input

    messages: List[Dict[str, str]] = []

    # Optional system prompt
    sys_prompt = g("system") or g("system_prompt")
    if sys_prompt is not None and str(sys_prompt).strip():
        messages.append({"role": "system", "content": _stringify_content(sys_prompt)})

    # Optional tool context precedes assistant
    if tool_ctx is not None and str(tool_ctx).strip():
        messages.append({"role": "tool", "content": _stringify_content(tool_ctx)})

    if user_text is not None and assistant_text is not None:
        messages.append({"role": "user", "content": _stringify_content(user_text)})
        messages.append({"role": "assistant", "content": _stringify_content(assistant_text)})
        return messages

    if user_text is not None:
        messages.append({"role": "user", "content": _stringify_content(user_text)})
        # No assistant -> unusable for SFT, but keep a placeholder to avoid crashes
        messages.append({"role": "assistant", "content": ""})
        return messages

    if assistant_text is not None:
        # Synthesize a generic user instruction when only assistant text exists
        messages.append({"role": "user", "content": "请根据上下文完成任务。"})
        messages.append({"role": "assistant", "content": _stringify_content(assistant_text)})
        return messages

    # Fallback: stringify whole sample
    body = json.dumps(sample, ensure_ascii=False)
    return [
        {"role": "user", "content": "将以下内容标准化输出："},
        {"role": "assistant", "content": body},
    ]


def build_text_from_sample(sample: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    # Always convert to messages first, then apply chat template
    messages = build_messages_from_sample(sample)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def has_non_empty_assistant(sample: Dict[str, Any], min_chars: int) -> bool:
    try:
        msgs = build_messages_from_sample(sample)
    except Exception:
        return False
    for m in msgs:
        if m.get("role") == "assistant":
            if isinstance(m.get("content"), str) and len(m["content"].strip()) >= min_chars:
                return True
    return False


def load_and_prepare_dataset(dataset_path: str, tokenizer: AutoTokenizer, min_assistant_chars: int) -> Tuple[Dataset, str]:
    # datasets supports both json array and jsonl
    extension = os.path.splitext(dataset_path)[1].lower()
    data_files = {"train": dataset_path}
    if extension in [".jsonl", ".jl"]:
        ds = load_dataset("json", data_files=data_files, split="train")
    else:
        # .json: could be array or object with list under a key; datasets handles array; otherwise may fail
        ds = load_dataset("json", data_files=data_files, split="train")

    # filter invalid/empty assistant samples first
    ds = ds.filter(lambda ex: has_non_empty_assistant(ex, min_chars=min_assistant_chars), desc="Filter empty assistant")

    # map to a unified 'text' field using chat template when available
    def map_to_text(example: Dict[str, Any]) -> Dict[str, Any]:
        return {"text": build_text_from_sample(example, tokenizer)}

    ds = ds.map(map_to_text, remove_columns=[c for c in ds.column_names if c != "text"], desc="Formatting dataset to text")
    return ds, "text"


def load_raw_dataset(dataset_path: str) -> Dataset:
    extension = os.path.splitext(dataset_path)[1].lower()
    data_files = {"train": dataset_path}
    if extension in [".jsonl", ".jl"]:
        return load_dataset("json", data_files=data_files, split="train")
    return load_dataset("json", data_files=data_files, split="train")


def get_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    # Typical target modules for LLaMA/Qwen-style architectures
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def main() -> None:
    setup_logging()
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = load_tokenizer(args.model_name_or_path)

    # Preview mode: print normalized samples before any heavy model loading
    if args.show_samples and args.show_samples > 0:
        logging.info(f"Previewing {args.show_samples} normalized samples from {args.dataset_path}...")
        raw_ds = load_raw_dataset(args.dataset_path)
        count = 0
        for ex in raw_ds:
            try:
                msgs = build_messages_from_sample(ex)
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                # Print compact preview
                snippet = text[:400].replace("\n", " ")
                print(f"[Sample {count}] messages_roles={[m.get('role') for m in msgs]} text_snippet={snippet}...")
                count += 1
                if count >= args.show_samples:
                    break
            except Exception as e:
                print(f"[Sample error] {e}")
        if args.exit_after_preview:
            return

    logging.info(f"Loading model (4bit={args.use_4bit}): {args.model_name_or_path}")
    model = load_model(args.model_name_or_path, args.use_4bit, args.bnb_compute_dtype)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # prepare dataset
    logging.info(f"Loading dataset from: {args.dataset_path}")
    train_dataset, dataset_text_field = load_and_prepare_dataset(args.dataset_path, tokenizer, args.min_assistant_chars)

    # collator to only compute loss on assistant response; default response_template fits Qwen ChatML
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=args.response_template,
        tokenizer=tokenizer,
    )

    # LoRA config
    lora_config = get_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    # training args
    bfloat16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        evaluation_strategy="no" if args.eval_steps == 0 else "steps",
        eval_steps=None if args.eval_steps == 0 else args.eval_steps,
        bf16=bfloat16,
        fp16=not bfloat16,
        dataloader_num_workers=2,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=["none"],
        seed=args.seed,
    )

    # build trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field=dataset_text_field,
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        peft_config=lora_config,
        args=training_args,
        data_collator=data_collator,
    )

    if args.resume_from_checkpoint:
        logging.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

    logging.info("Start training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # save adapter or merged model
    logging.info("Saving model and tokenizer...")
    if args.merge_lora:
        try:
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
            logging.info("Merged LoRA weights into base model and saved.")
        except Exception as e:
            logging.warning(f"Merging failed, saving adapter-only. Error: {e}")
            trainer.save_model()
            tokenizer.save_pretrained(args.output_dir)
    else:
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

    # brief effective batch size info
    eff_bs = args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * args.gradient_accumulation_steps
    logging.info(
        f"Done. Effective batch size: {eff_bs}. Model saved to: {args.output_dir}"
    )


if __name__ == "__main__":
    main() 