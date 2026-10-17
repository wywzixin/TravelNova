#!/usr/bin/env python3
# coding: utf-8

import argparse
import json
import os
import inspect as _inspect
from typing import List, Tuple, Any, Dict, Optional

import torch
from transformers import AutoTokenizer


# ===============
# 模板与数据处理（独立实现，无外部依赖）
# ===============

def _supports_tools_kw(tokenizer: AutoTokenizer) -> bool:
	"""
	检测分词器的 `apply_chat_template` 是否支持 `tools` 关键字参数。
	某些模型模板支持将工具描述（tool schema）传入，从而生成带工具调用格式的 prompt。
	"""
	try:
		sig = _inspect.signature(tokenizer.apply_chat_template)
		return any(p.name == "tools" for p in sig.parameters.values())
	except Exception:
		return False


class JsonlConversations:
	"""
	数据集读取与样本构造（适配 Qwen 聊天模板）：
	- 输入：JSON 数组或 JSONL，每行/项至少包含一个消息列表字段：`messages` 或 `conversation`
	- 可选：`tools` 字段（工具描述数组），若分词器模板支持将会参与模板化
	- 输出：`__getitem__` 返回 dict：
	  - input_ids: 模板化后的 token 序列
	  - attention_mask: 与 input_ids 等长的全 1（本脚本不做 padding，打印更直观）
	  - labels: 与 input_ids 等长，被 mask 掉（-100）的部分不会计算 loss；仅 assistant 段保留为原 token id 参与 loss
	- 采用“前缀差分”方式定位每条 `assistant` 消息在模板化序列中的 token 区间，从而生成仅 assistant 段的 mask。
	"""

	def __init__(self, path: str, tokenizer: AutoTokenizer, max_seq_length: int, only_last_assistant: bool = False, default_tools: Optional[List[Dict[str, Any]]] = None):
		"""
		读取 JSON/JSONL 数据，并标准化为统一结构：
		- 允许 JSON 数组文件，或 JSONL（每行一个 JSON 对象）
		- 每个样本保留：messages（或 conversation 重命名）、可选 tools
		"""
		self.tokenizer = tokenizer
		self.max_seq_length = max_seq_length
		self.examples: List[Dict[str, Any]] = []
		self._default_tools: Optional[List[Dict[str, Any]]] = (default_tools if isinstance(default_tools, list) and len(default_tools) > 0 else None)
		self.only_last_assistant = only_last_assistant
		
		def _normalize_obj(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
			"""
			将任意输入对象标准化为：
			{ "messages": [...], "tools": [...](可选) }
			"""
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
			elif self._default_tools is not None:
				# 为未提供 tools 的样本注入默认工具列表（用于统一工具监督）
				new_obj["tools"] = self._default_tools
			return new_obj
		
		# 读取文件（先尝试 JSON 数组，再回退 JSONL）
		with open(path, "r", encoding="utf-8") as f:
			content = f.read()
		stripped = content.lstrip()
		parsed_any = False
		
		# 优先尝试 JSON 数组
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
		
		# 回退 JSONL
		if not parsed_any:
			print(f"Failed to parse JSON array, trying JSONL: {path}")
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
		"""
		调用分词器的 chat 模板接口 `apply_chat_template`：
		- 根据分词器能力，选择是否传入 tools
		- 关闭 add_generation_prompt，保持严格的监督信号构造
		- 返回：若 tokenize=True，则返回 token id 序列（list[int]）
		"""
		kwargs = dict(tokenize=tokenize, add_generation_prompt=False)
		if return_tensors is not None:
			kwargs["return_tensors"] = return_tensors
		if tools and self._use_tools_kw:
			kwargs["tools"] = tools
		return self.tokenizer.apply_chat_template(messages, **kwargs)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		"""
		单样本构造：
		1) 对完整 messages 做一次模板化，得到总序列 `full_ids`
		2) 通过“前缀差分”：
		   - 依次取 messages[:i+1] 模板化，得到 `cur_len`
		   - 若第 i 条消息是 assistant，则 (prev_len, cur_len) 为该 assistant 段在 full_ids 中的区间
		   - 将该区间置为 True，表示参与 loss
		3) 超长则从头部截断（保留尾部），同步截断 mask
		4) labels 复制自 input_ids，将非 loss 区域置为 -100
		"""
		ex = self.examples[idx]
		messages: List[Dict[str, Any]] = ex["messages"]
		tools: Optional[List[Dict[str, Any]]] = ex.get("tools")
		
		# 一次性完整模板化得到 token 序列
		try:
			full_ids = self._apply_template(messages, tools, tokenize=True, return_tensors=None)
		except Exception:
			# 少数分词器旧版本不支持 kwargs 时的降级调用
			full_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
		total_len = len(full_ids)
		
		# 通过前缀差分方式构建 assistant 段的区间
		assistant_spans = []
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
					assistant_spans.append((start, end))
			prev_len = cur_len
		
		# 根据 only_last_assistant 决定 mask 哪些区间
		assistant_mask = torch.zeros(total_len, dtype=torch.bool)
		if self.only_last_assistant:
			if len(assistant_spans) > 0:
				start, end = assistant_spans[-1]
				assistant_mask[start:end] = True
		else:
			for start, end in assistant_spans:
				assistant_mask[start:end] = True
		
		# 截断到最大长度（保留尾部更有利于监督长对话的最新轮次）
		if total_len > self.max_seq_length:
			overflow = total_len - self.max_seq_length
			full_ids = full_ids[overflow:]
			assistant_mask = assistant_mask[overflow:]
		
		# 生成张量；注意此处不做 padding，保持每条样本各自长度便于肉眼检查
		input_ids = torch.tensor(full_ids, dtype=torch.long)
		attention_mask = torch.ones_like(input_ids)  # 无 padding，均为 1
		labels = input_ids.clone()
		labels[~assistant_mask] = -100  # -100 表示忽略（不会参与 loss 计算）
		
		return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class DataCollatorForCausal:
	"""
	可选的 batch collator（本脚本打印不使用，仅在需要导出/批量处理时可用）：
	- 将变长样本按 batch 维度对齐（pad）
	- input_ids 用 pad_token_id 填充
	- attention_mask 用 0 填充
	- labels 用 -100 填充（保持忽略语义）
	- 可选对齐到 `pad_to_multiple_of`，在部分硬件上利于吞吐
	"""

	def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: Optional[int] = None) -> None:
		self.tokenizer = tokenizer
		self.pad_to_multiple_of = pad_to_multiple_of

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


# ===============
# 可视化与导出
# ===============

def parse_args() -> argparse.Namespace:
	"""
	命令行参数：
	--data_file: 数据文件路径（JSON 数组或 JSONL）
	--model_name_or_path: 分词器来源（如本地路径或模型名）
	--max_seq_length: 最大序列长度，超长从头部截断
	--num_samples: 打印样本数量
	--start: 起始样本索引
	--show_tokens: 打印 token 字符串（默认打印 id）
	--max_print_tokens: 每条样本打印的头部 token 数限制
	--color: 使用 ANSI 颜色高亮参与 loss 的 token
	--export_dir: 可选导出目录（导出为 JSON 的张量）
	--local_files_only: 仅本地加载分词器（离线）
	"""
	parser = argparse.ArgumentParser(description="Inspect Qwen SFT dataset: visualize assistant-only mask and converted tensors (standalone)")
	parser.add_argument("--data_file", type=str, default="/root/autodl-tmp/merged_for_training_prompted.json",help="Path to JSON/JSONL file")
	parser.add_argument("--model_name_or_path", type=str, default="qwen3-4b-instruct", help="Tokenizer source")
	parser.add_argument("--max_seq_length", type=int, default=20000)
	parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to print")
	parser.add_argument("--start", type=int, default=0, help="Start index in dataset")
	parser.add_argument("--show_tokens", action="store_true", help="Show token strings rather than ids")
	parser.add_argument("--max_print_tokens", type=int, default=20000, help="Max tokens to print per sample (head)")
	parser.add_argument("--color", action="store_true", help="Use ANSI colors to highlight loss tokens")
	parser.add_argument("--export_dir", type=str, default="", help="Optional dir to export converted tensors per sample as JSON")
	parser.add_argument("--local_files_only", action="store_true", help="Load tokenizer only from local files (offline)")
	parser.add_argument("--show_ignored_segments", action="store_true", help="Show decoded spans that do NOT contribute to loss")
	parser.add_argument("--show_full_decoded", action="store_true", help="Show full decoded text per sample")
	parser.add_argument("--only_last_assistant", action="store_true", help="Only mask the last assistant segment for loss")
	# Global tools injection for samples missing tools
	parser.add_argument("--tools_file", type=str, default="/root/autodl-tmp/all_tools.json", help="Path to JSON file with a list of tools to inject when sample.tools is missing")
	parser.add_argument("--tools_json", type=str, default="", help="Inline JSON string (list) of tools to inject when sample.tools is missing")
	return parser.parse_args()


def _ansi(color: str) -> str:
	"""
	简单 ANSI 颜色辅助（可选，用于终端高亮）。
	"""
	if color == "red":
		return "\033[31m"
	if color == "green":
		return "\033[32m"
	if color == "yellow":
		return "\033[33m"
	if color == "blue":
		return "\033[34m"
	if color == "magenta":
		return "\033[35m"
	if color == "cyan":
		return "\033[36m"
	if color == "bold":
		return "\033[1m"
	if color == "reset":
		return "\033[0m"
	return ""


def _highlight(tokens: List[str], loss_mask: torch.Tensor, use_color: bool) -> str:
	"""
	将 token 序列按 loss_mask 高亮：
	- use_color=True：使用绿色高亮
	- use_color=False：使用方括号包裹
	"""
	assert len(tokens) == loss_mask.numel()
	parts: List[str] = []
	for i, tok in enumerate(tokens):
		if loss_mask[i].item():
			if use_color:
				parts.append(f"{_ansi('green')}{tok}{_ansi('reset')}")
			else:
				parts.append(f"[{tok}]")
		else:
			parts.append(tok)
	return " ".join(parts)


def _contiguous_true_spans(mask: torch.Tensor) -> List[Tuple[int, int]]:
	"""
	将布尔 mask 拆分为若干个连续为 True 的区间 [start, end)。
	用于回显连续参与 loss 的文本片段。
	"""
	spans: List[Tuple[int, int]] = []
	start = None
	for i, v in enumerate(mask.tolist()):
		if v and start is None:
			start = i
		elif not v and start is not None:
			spans.append((start, i))
			start = None
	if start is not None:
		spans.append((start, mask.numel()))
	return spans


def _export_sample_json(path: str, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> None:
	"""
	将单条样本的张量导出为 JSON，便于离线排查。
	"""
	data = {
		"input_ids": input_ids.tolist(),
		"attention_mask": attention_mask.tolist(),
		"labels": labels.tolist(),
	}
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False)


def main() -> None:
	"""
	入口：
	- 加载分词器（可离线）
	- 加载并构造数据集
	- 逐条打印样本，展示：
	  - 序列长度 / 参与 loss token 数
	  - token id（或字符串）与对应的 loss 掩码
	  - 连续参与 loss 的原文片段（包含特殊符号，便于定位模板边界）
	- 可选导出处理后的张量
	"""
	args = parse_args()

	print(f"Loading tokenizer: {args.model_name_or_path}")
	tokenizer = AutoTokenizer.from_pretrained(
		args.model_name_or_path,
		trust_remote_code=True,
		local_files_only=args.local_files_only,
	)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "right"

	# Optional: load a global tools list to inject when sample lacks tools
	default_tools = None
	if args.tools_file:
		try:
			with open(args.tools_file, "r", encoding="utf-8") as f:
				obj = json.load(f)
			if isinstance(obj, list):
				default_tools = obj
			else:
				raise ValueError("tools_file must contain a JSON array of tool objects")
		except Exception as e:
			raise RuntimeError(f"Failed to load tools_file: {args.tools_file}: {e}")
	elif args.tools_json:
		try:
			obj = json.loads(args.tools_json)
			if isinstance(obj, list):
				default_tools = obj
			else:
				raise ValueError("tools_json must be a JSON array of tool objects")
		except Exception as e:
			raise RuntimeError(f"Failed to parse tools_json: {e}")

	print(f"Loading dataset: {args.data_file}")
	dataset = JsonlConversations(args.data_file, tokenizer, args.max_seq_length, args.only_last_assistant, default_tools=default_tools)

	total = len(dataset)
	start = max(0, args.start)
	end = min(total, start + args.num_samples)

	if args.export_dir:
		os.makedirs(args.export_dir, exist_ok=True)

	for idx in range(start, end):
		sample = dataset[idx]
		input_ids = sample["input_ids"]
		attention_mask = sample["attention_mask"]
		labels = sample["labels"]

		# 参与 loss 的位置：labels != -100
		loss_mask = labels.ne(-100)

		# 限制打印长度（仅打印头部，避免超长输出）
		head_len = min(input_ids.numel(), args.max_print_tokens)
		head_input_ids = input_ids[:head_len]
		head_loss_mask = loss_mask[:head_len]

		# 基本信息
		print("\n" + "=" * 80)
		print(f"Sample #{idx} / {total}")
		print(f"seq_len={input_ids.numel()}  loss_tokens={int(loss_mask.sum().item())}  masked_out={int((~loss_mask).sum().item())}")

		# 以 token 字符串或 id 方式展示，并标注哪些会算 loss
		if args.show_tokens:
			tokens = tokenizer.convert_ids_to_tokens(head_input_ids.tolist())
			line = _highlight(tokens, head_loss_mask, args.color)
			print("Tokens (green or [brackets] = contribute to loss):")
			print(line)
		else:
			# 打印 id 与 loss 掩码行（1=参与 loss，.=忽略）
			ids_line = " ".join(str(i) for i in head_input_ids.tolist())
			mask_line = " ".join("1" if b else "." for b in head_loss_mask.tolist())
			print("input_ids:")
			print(ids_line)
			print("loss_mask (1=loss, .=ignore):")
			print(mask_line)

		# 展示参与 loss 的连续文本片段，便于验证模板与 mask 的对应关系
		spans = _contiguous_true_spans(loss_mask)
		if spans:
			print("Loss segments (decoded with special tokens):")
			for s, e in spans:
				segment_ids = input_ids[s:e]
				text = tokenizer.decode(segment_ids, skip_special_tokens=False)
				display_text = text if len(text) <= 512 else (text[:509] + "...")
				print(f"  [{s}:{e}] -> {display_text!r}")
		else:
			print("No loss segments found (unexpected for assistant-only masking)")

		# 可选：展示未参与 loss 的片段（更可读）
		if args.show_ignored_segments:
			ignored_spans = _contiguous_true_spans(~loss_mask)
			if ignored_spans:
				print("Ignored segments (decoded with special tokens):")
				for s, e in ignored_spans:
					segment_ids = input_ids[s:e]
					text = tokenizer.decode(segment_ids, skip_special_tokens=False)
					display_text = text if len(text) <= 512 else (text[:509] + "...")
					print(f"  [{s}:{e}] -> {display_text!r}")
			else:
				print("No ignored segments (all tokens contribute to loss)")

		# 可选：打印整段解码文本
		if args.show_full_decoded:
			print("Full decoded (skip_special_tokens=True):")
			print(tokenizer.decode(input_ids.tolist(), skip_special_tokens=True))
			print("Full decoded (skip_special_tokens=False):")
			print(tokenizer.decode(input_ids.tolist(), skip_special_tokens=False))

		# 可选导出张量
		if args.export_dir:
			out_path = os.path.join(args.export_dir, f"sample_{idx:06d}.json")
			_export_sample_json(out_path, input_ids, attention_mask, labels)
			print(f"Exported tensors to: {out_path}")


if __name__ == "__main__":
	main()
 
 
# python -u /root/autodl-tmp/Agent+SFT/inspect_qwen_dataset.py \
#   --data_file /root/autodl-tmp/Agent+SFT/merged_train_final.json \
#   --model_name_or_path /root/autodl-tmp/Agent+SFT/qwen3-0_6b \
#   --tools_file /root/autodl-tmp/Agent+SFT/all_tools.json \
#   --only_last_assistant \
#   --num_samples 8 --start 0 --show_full_decoded