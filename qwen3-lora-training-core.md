---
title: LoRA 是怎么接进旅行助手的：Qwen3-0.6B 微调脚本拆解
date: 2026-09-27
tags: [大模型, LoRA, Qwen, SFT, Agent]
---

# LoRA 是怎么接进旅行助手的：Qwen3-0.6B 微调脚本拆解

> 微调实战 · [第一篇：数据构造](travel-assistant-sft-data-pipeline.md) → 第二篇：LoRA 训练原理 → [第三篇：微调与 Agent 联调](travel-assistant-llm-finetuning.md)
>
> [博客阅读](https://wywzixin.github.io/agent/qwen3-lora-training/) · [返回项目首页](README.md)

上一篇文章讲了旅行助手的微调数据如何生成。这篇继续往下看训练代码，专门拆解项目中的 `train_qwen_lora.py`：模型怎样加载，LoRA 适配器怎样插入，哪些 token 会参与 loss，`Trainer` 怎样完成训练，以及训练完成后如何使用 LoRA。

这份脚本实现的是**普通 LoRA**。它会完整加载 Qwen3-0.6B 基座模型，在指定的注意力层和 MLP 层旁边增加可训练的低秩矩阵；基座模型参数保持冻结。脚本没有使用 4-bit 量化，因此它不是 QLoRA。

整条链路可以画成这样：

```text
merged_train_final.json
          |
          v
JsonlConversations
  读取 conversation/messages
  套用 Qwen chat template
  只保留 assistant token 的 labels
          |
          v
DataCollatorForCausal
  batch 内 padding
          |
          v
Qwen3-0.6B 基座模型 + LoRA adapters
          |
          v
Transformers Trainer
          |
          v
adapter_model.safetensors
```

## 一、LoRA 到底改变了什么

完整微调会更新模型的全部权重。LoRA 则冻结原来的权重 `W`，在目标线性层上增加两个较小的矩阵 `A` 和 `B`：

```text
W' = W + (alpha / r) * B * A
```

训练期间只更新 `A` 和 `B`。推理时可以把这两个矩阵临时叠加到基座模型上，也可以把它们合并回基座模型，生成一个独立的完整模型目录。

这对旅行助手很合适：目标是让小模型更稳定地识别旅行意图、组织工具参数、接收工具结果并继续回答，而不是重新学习通用语言能力。LoRA 的优点是训练参数少、适配器文件小、可以为同一个基座保存多个行为版本。代价是适配器必须和对应的基座模型配套，不能单独使用。

## 二、脚本负责哪些工作

`train_qwen_lora.py` 主要串起六件事：

1. 解析命令行参数并固定随机种子。
2. 加载 tokenizer 和 Qwen 基座模型。
3. 根据 `LoraConfig` 把适配器插入模型。
4. 读取 JSON/JSONL 对话并制作 assistant-only labels。
5. 用 `Trainer` 执行梯度累积、日志和 checkpoint 保存。
6. 在训练结束后保存 LoRA adapter 和训练状态。

它不负责生成旅行数据，也不负责调用高德、天气或 RAG 服务。训练阶段只需要训练文件和本地基座模型；工具服务是在转换数据或运行 Agent 时使用的。

## 三、先看命令行参数

参数在 `parse_args()` 中定义。最重要的几组参数如下：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--train_file` | 必填 | JSON 数组或 JSONL 训练文件 |
| `--model_name_or_path` | `/root/autodl-tmp/qwen3-0_6b` | 基座模型路径或 Hugging Face 名称 |
| `--output_dir` | `./qwen_lora_output` | LoRA 输出目录 |
| `--max_seq_length` | `4096` | 单条样本最大 token 数 |
| `--learning_rate` | `2e-4` | 学习率 |
| `--num_train_epochs` | `3` | 训练轮数 |
| `--per_device_train_batch_size` | `1` | 每张卡每步的样本数 |
| `--gradient_accumulation_steps` | `8` | 梯度累积步数 |
| `--fp16` / `--bf16` | 关闭 | 选择半精度训练 |
| `--gradient_checkpointing` | 关闭 | 用计算换显存 |
| `--lora_r` | `32` | LoRA 的秩，决定适配器容量 |
| `--lora_alpha` | `64` | LoRA 更新的缩放系数 |
| `--lora_dropout` | `0.05` | LoRA 分支的 dropout |
| `--target_modules` | 七个 Qwen 层名 | 指定插入适配器的模块 |

脚本里的默认模型路径是 Linux 路径。Windows 下必须显式传入项目中的模型目录，例如 `.models\qwen3-0.6b-clean`；否则脚本会去寻找不存在的 `/root/autodl-tmp/...`。`--local_files_only` 可以强制只从本地读取，避免训练时意外下载文件。

## 四、加载 tokenizer 和基座模型

入口函数首先固定随机种子并创建输出目录：

```python
args = parse_args()
set_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)
```

然后分别加载 tokenizer 和语言模型：

```python
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
    local_files_only=args.local_files_only,
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
    local_files_only=args.local_files_only,
)
```

如果 tokenizer 没有 `pad_token`，代码把它设成 `eos_token`，并使用右侧 padding：

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

`trust_remote_code=True` 允许 Transformers 使用模型仓库提供的自定义代码。已经把模型完整下载到 `models\qwen3-0.6b-clean` 后，训练可以加 `--local_files_only`，这样更容易排查路径问题。

## 五、训练前的显存选项

脚本提供两种半精度选项，但它们只是 dtype 转换，不是量化：

```python
torch_dtype = None
if args.bf16:
    torch_dtype = torch.bfloat16
elif args.fp16:
    torch_dtype = torch.float16
if torch_dtype is not None:
    model = model.to(dtype=torch_dtype)
```

在 RTX 4060 上通常先使用 `--fp16`。`--bf16` 可以作为另一种选择，二者不要同时传入；如果同时传入，训练参数里会优先启用 BF16。

梯度检查点的逻辑是：

```python
if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
```

训练时关闭 `use_cache` 是因为缓存主要服务于生成，和 gradient checkpointing 一起使用会产生冲突。训练结束、进入推理阶段时，再把 `use_cache` 设回 `True`。

## 六、LoRA 插入模型的核心代码

真正把 LoRA 接到 Qwen 上的是 `build_lora_model()`：

```python
def build_lora_model(base_model, args):
    target_modules = [
        m.strip()
        for m in args.target_modules.split(",")
        if m.strip()
    ]

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
```

项目的默认目标模块是：

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

前四个是注意力投影层，后三个是 Qwen MLP 中的投影层。让 LoRA 同时覆盖这两组层，通常比只覆盖 `q_proj` 和 `v_proj` 更有表达能力，但适配器也会更大、训练显存略高。

默认配置的含义是：

- `r=32`：低秩矩阵的维度。值越大，能学习的变化越多，参数量也越大。
- `lora_alpha=64`：缩放系数。当前 `alpha/r=2`，表示 LoRA 更新有较明显的影响。
- `lora_dropout=0.05`：训练时随机丢弃一小部分 LoRA 分支，降低过拟合风险。
- `bias="none"`：不额外训练原有 bias。
- `task_type="CAUSAL_LM"`：告诉 PEFT 这是因果语言模型。

`print_trainable_parameters()` 会在启动时显示可训练参数数量。看到大部分参数被冻结、只有 LoRA 参数可训练，才说明适配器挂载成功。如果报“找不到 target module”，通常是模型层名和参数不一致，需要先检查 `model.named_modules()`，再调整 `--target_modules`。

## 七、数据怎样进入 loss

训练脚本复用了 `inspect_qwen_dataset.py` 中的两个工具类：

```python
train_dataset = JsonlConversations(
    args.train_file,
    tokenizer,
    args.max_seq_length,
)
data_collator = DataCollatorForCausal(tokenizer=tokenizer)
```

`JsonlConversations` 可以读取 JSON 数组或 JSONL。每条记录至少需要有 `messages` 或 `conversation` 列表，例如：

```json
{
  "conversation": [
    {"role": "system", "content": "你是旅行助手"},
    {"role": "user", "content": "我想去烟台玩三天"},
    {"role": "assistant", "content": "我先查询烟台攻略。"}
  ]
}
```

数据集类会调用 Qwen tokenizer 的 `apply_chat_template()`，把多条消息转换为一条 token 序列。随后它用“前缀差分”定位每个 assistant 消息在序列中的 token 区间：

```text
system + user                         -> 不计算 loss
assistant 的工具调用                  -> 计算 loss
tool 返回结果                         -> 不计算 loss
assistant 的最终答复                  -> 计算 loss
```

非 assistant token 的 label 会被设为 `-100`。PyTorch 的交叉熵会忽略这些位置，因此模型学习的是“助手应该怎样调用和回答”，而不是背诵用户问题或工具返回文本。

这里要区分两个脚本：`train_qwen_lora.py` 使用 `only_last_assistant=False` 的默认行为，**一条对话中所有 assistant 段都会参与 loss**；`train_qwen_last_assistant_lora.py` 才会显式传入 `only_last_assistant=True`，只训练最后一段 assistant 回复。两者数据格式相近，但监督重点不同。

`DataCollatorForCausal` 在 batch 内把变长样本补齐：`input_ids` 用 pad token，`attention_mask` 用 0，`labels` 用 `-100`。这样 padding 位置也不会影响 loss。

当前这个 `train_qwen_lora.py` 没有 `--tools_file` 参数，也不会自动向缺少 `tools` 字段的样本注入全局工具 schema。项目现有训练文件中的工具调用轨迹和 system 内容需要先准备好；不要把另一个训练脚本的 `--tools_file` 参数直接复制到这里。

## 八、Trainer 怎样控制训练

脚本把命令行参数映射到 Transformers 的 `TrainingArguments`：

```python
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    warmup_ratio=args.warmup_ratio,
    logging_steps=args.logging_steps,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    lr_scheduler_type=args.lr_scheduler_type,
    optim="adamw_torch",
    bf16=args.bf16,
    fp16=args.fp16 and not args.bf16,
    report_to=[],
    remove_unused_columns=False,
    seed=args.seed,
    save_safetensors=True,
)
```

单卡 batch 为 1、梯度累积为 8 时，实际每 8 个小步更新一次参数。它不等于一次把 8 条样本全部放进显存，而是把 8 次反向传播的梯度累积起来，再执行一次 optimizer step，所以能在 4060 这类显存有限的卡上训练较长样本。

脚本注册了 `ConsoleLossCallback`。每次日志触发时，它会打印：

```text
step=10 | loss=... | lr=...
```

传入 `--log_file .\outputs\qwen3-0.6b-lora-core\train.log` 后，还会追加写入日志文件。代码中的 callback 条件包含常量 `or True`，所以即使不传 `--log_file`，控制台日志也会启用；不传文件时只打印，不落盘。
传入 `--log_file .\outputs\qwen3-0.6b-lora-core\train.log` 后，还会追加写入日志文件。代码中的 callback 条件包含常量 `or True`，所以即使不传 `--log_file`，控制台日志也会启用；不传文件时只打印，不落盘。

## 九、Windows 上怎样启动 LoRA 训练

在项目根目录执行。下面的命令使用现有的 808 条 `merged_train_final.json`，并把输出写入一个新目录，避免覆盖之前的适配器：

```powershell
cd E:\AI\Agent+SFT_SHENWEI
conda activate qwen3-sft

python .\train_qwen_lora.py `
  --train_file .\merged_train_final.json `
  --model_name_or_path .\models\qwen3-0.6b-clean `
  --output_dir .\outputs\qwen3-0.6b-lora-core `
  --max_seq_length 4096 `
  --num_train_epochs 1 `
  --learning_rate 1e-4 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --logging_steps 10 `
  --save_steps 1000 `
  --fp16 `
  --local_files_only `
  --log_file .\outputs\qwen3-0.6b-lora-core\train.log
```

这是一个适合先跑通流程的起点，不代表唯一正确的超参数。若显存不足，可以依次降低 `--max_seq_length`、打开 `--gradient_checkpointing`，或把 `--gradient_accumulation_steps` 调小。若 Windows 数据加载进程出现异常，可临时加上 `--dataloader_num_workers 0`。

不要同时传 `--fp16` 和 `--bf16`。想使用 BF16 时，把命令中的 `--fp16` 换成 `--bf16`。如果模型已经在本地，保留 `--local_files_only`；如果希望从 Hugging Face 下载模型，则去掉它并确保网络和缓存权限正常。

## 十、训练完成后会得到什么

训练结束时，代码执行：

```python
trainer.train()
trainer.save_state()
trainer.save_model(args.output_dir)
```

输出目录通常包含：

```text
outputs\qwen3-0.6b-lora-core\
  adapter_config.json
  adapter_model.safetensors
  tokenizer.json / tokenizer_config.json / vocab.json / merges.txt
  training_args.bin
  trainer_state.json
  checkpoint-*/
```

其中最核心的是 `adapter_model.safetensors` 和 `adapter_config.json`。它们只保存 LoRA 增量参数和配置，不包含 Qwen3-0.6B 的完整基座权重，因此文件会比完整模型小很多。`checkpoint-*` 目录用于中断恢复；`save_total_limit=3` 会限制保留的 checkpoint 数量。

已有的 `outputs\qwen3-0.6b-existing-data-lora` 目录就是这种 adapter 输出的例子，其中 `adapter_model.safetensors` 约 80 MB。这个目录之前由项目中的另一份训练脚本和参数生成，不能把它的 loss 直接当成当前脚本默认配置的结果；复现实验时应记录脚本名、数据文件和全部命令行参数。

## 十一、方式一：推理时动态加载 LoRA

动态加载需要同时提供**同一个基座模型**和 adapter 目录：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_path = r".\models\qwen3-0.6b-clean"
adapter_path = r".\outputs\qwen3-0.6b-lora-core"

tokenizer = AutoTokenizer.from_pretrained(
    base_path,
    local_files_only=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.float16,
    local_files_only=True,
)
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    is_trainable=False,
)
model.eval()

messages = [
    {"role": "user", "content": "我想去烟台玩三天，应该先做什么？"}
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    output = model.generate(
        inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = output[:, inputs.shape[-1]:]
print(tokenizer.decode(new_tokens[0], skip_special_tokens=True))
```

这种方式适合对比多个 adapter：同一个 Qwen 基座可以依次加载不同旅行助手版本。adapter 目录不能直接传给 `AutoModelForCausalLM.from_pretrained()`，因为它不是完整模型；要使用 `PeftModel.from_pretrained(base_model, adapter_path)`。

## 十二、方式二：把 LoRA 合并成完整模型

项目提供了 `merge_lora_into_base.py`。它的核心是：

```python
lora_model = PeftModel.from_pretrained(
    model,
    args.adapter_path,
    is_trainable=False,
)
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained(args.output_dir, safe_serialization=True)
tokenizer.save_pretrained(args.output_dir)
```

Windows 下可以这样合并：

```powershell
python .\merge_lora_into_base.py `
  --base_model .\models\qwen3-0.6b-clean `
  --adapter_path .\outputs\qwen3-0.6b-lora-core `
  --output_dir .\models\qwen3-0.6b-lora-core-merged `
  --local_files_only `
  --fp16
```

合并后的目录包含完整权重和 tokenizer，可以像普通 Qwen 模型一样加载。合并会占用额外磁盘和内存，完成后仍要保留基座与 adapter，除非已经确认不再需要回滚或比较。

## 十三、接入当前旅行助手

项目中的 `local_qwen.py` 通过 `AutoModelForCausalLM.from_pretrained()` 读取一个完整模型目录。为了让现有旅行助手直接使用微调结果，先合并 LoRA，再指向合并目录：

```powershell
$env:TRAVEL_ASSISTANT_BACKEND="local_qwen"
$env:LOCAL_QWEN_MODEL_PATH="E:\AI\Agent+SFT_SHENWEI\models\qwen3-0.6b-lora-core-merged"

python .\travel_assistant_funcall_fixed.py
```

如果仍然把 `LOCAL_QWEN_MODEL_PATH` 指向 `outputs\qwen3-0.6b-lora-core`，现有 backend 会把 adapter 当成完整语言模型读取，通常会因为缺少基座权重而失败。要直接加载 adapter，需要改造 backend，在加载基座后显式调用 `PeftModel.from_pretrained()`。

## 十四、几个参数怎样调

LoRA 参数不是越大越好，应该结合数据量和验证集观察：

| 参数 | 调大后的效果 | 代价 |
| --- | --- | --- |
| `r` | 适配器表达能力更强 | 参数量、显存和过拟合风险增加 |
| `lora_alpha` | LoRA 更新影响更明显 | 学习过快或输出风格漂移的风险增加 |
| `lora_dropout` | 正则化更强 | 数据很少时可能降低拟合能力 |
| `max_seq_length` | 能保留更长的工具轨迹 | 显存占用和训练时间上升 |
| `learning_rate` | 参数变化更快 | 太大时 loss 抖动、模型遗忘基座能力 |
| `num_train_epochs` | 更充分地重复数据 | 模板数据容易过拟合 |

对当前 808 条左右的训练样本，可以先用 1 个 epoch、`1e-4`、`r=32` 跑通，再用没有出现在训练集里的问题做评估。不要只看训练 loss；要检查模型是否能提取烟台、日期和人数，是否在缺少目的地时追问，是否正确输出工具调用，工具结果回来后是否停止重复追问。

## 十五、常见问题和排查顺序

### 1. 找不到 LoRA 目标模块

先打印 `model.named_modules()`，确认模型确实有 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。不同架构的模块名可能不同，不能盲目沿用 Qwen 配置。

### 2. 显存不够

先降低 `max_seq_length`，再打开 `--gradient_checkpointing` 和 `--fp16`，最后再考虑减少目标模块或梯度累积。普通 LoRA 并不等于低显存训练；基座模型仍然要完整加载，只有可训练参数变少。

### 3. adapter 加载失败

确认三件事：adapter 路径里有 `adapter_config.json` 和 `adapter_model.safetensors`；加载时使用的是同一个基座模型；没有把 adapter 目录误传给 `AutoModelForCausalLM.from_pretrained()`。

### 4. 模型仍然反复问“你要去哪儿”

这不一定是 LoRA 参数问题。先检查训练样本里的多轮消息、assistant-only mask、工具调用结果和 Agent 的上下文拼接。如果训练数据没有让模型看到“已经确认目的地后继续执行”的完整轨迹，单纯增加 epoch 可能只会放大错误模式。

## 结语

`train_qwen_lora.py` 的核心并不复杂：读取对话，构造 assistant labels，把 LoRA 适配器插到 Qwen 的关键投影层，然后交给 `Trainer` 更新少量参数。真正决定效果的，是三件事能否对齐：训练样本中的消息顺序、mask 保留的监督范围，以及运行时使用的基座模型和工具编排。

LoRA 适合把旅行助手的行为快速做成可比较的版本。训练完先保留 adapter，确认效果后再合并；这样既能让当前 Agent 使用完整模型，也能随时回到原始 Qwen，比较微调前后的工具调用差异。
