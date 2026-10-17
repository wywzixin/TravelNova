#!/usr/bin/env bash
set -euo pipefail

# 可按需修改以下三个路径
TRAIN_FILE="/root/autodl-tmp/Agent+SFT/merged_train_final_multiturn_v2.json"
MODEL_PATH="/root/autodl-tmp/Agent+SFT/qwen3-0_6b"
OUTPUT_DIR="/root/autodl-tmp/Agent+SFT/qwen3-0_6b_lora_v2_last_assistant"

mkdir -p "${OUTPUT_DIR}"

python3 /root/autodl-tmp/Agent+SFT/train_qwen_last_assistant_lora.py \
  --train_file "${TRAIN_FILE}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_seq_length 20000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --warmup_ratio 0.03 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --lr_scheduler_type cosine \
  --bf16 \
  --gradient_checkpointing \
  --local_files_only 