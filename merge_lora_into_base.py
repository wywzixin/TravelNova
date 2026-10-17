#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model and save merged model")
    parser.add_argument("--base_model", type=str, default="/root/autodl-tmp/qwen3-0_6b", help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, default="/root/autodl-tmp/qwen3-0_6b_lora_v5_last_assistant")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/qwen3-0_6b_lora_v5_merged", help="Where to save merged model")
    parser.add_argument("--local_files_only", action="store_true", help="Load only from local cache")
    parser.add_argument("--bf16", action="store_true", help="Load model in bfloat16 for lower memory during merge")
    parser.add_argument("--fp16", action="store_true", help="Load model in float16 for lower memory during merge")
    parser.add_argument("--device_map_auto", action="store_true", help="Use device_map='auto' to shard across devices during merge")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    print(f"Loading tokenizer from base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        device_map="auto" if args.device_map_auto else None,
        torch_dtype=torch_dtype,
    )

    print(f"Loading LoRA adapter: {args.adapter_path}")
    lora_model = PeftModel.from_pretrained(
        model,
        args.adapter_path,
        is_trainable=False,
    )

    print("Merging LoRA weights into base model (this may take a while)...")
    merged_model = lora_model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print("Done. Merged model saved at:", args.output_dir)


if __name__ == "__main__":
    main() 