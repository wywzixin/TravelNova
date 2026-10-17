---
title: 旅行助手中的大模型微调：从 Qwen3-0.6B 到可调用工具的本地 Agent
date: 2026-09-27
tags: [大模型, LoRA, Agent, RAG, Qwen]
---

# 旅行助手中的大模型微调：从 Qwen3-0.6B 到可调用工具的本地 Agent

> 微调实战 · [第一篇：数据构造](travel-assistant-sft-data-pipeline.md) → [第二篇：LoRA 训练原理](qwen3-lora-training-core.md) → 第三篇：微调与 Agent 联调
>
> [博客阅读](https://wywzixin.github.io/agent/travel-assistant-finetuning/) · [返回项目首页](README.md)

这次实践得到一个很明确的结论：**微调负责让模型学会做事，RAG 负责提供知识，工具负责连接真实世界。** 把三件事混成一件事，模型会变得难以调试；把它们拆开，旅行助手的每一步都能单独验证。

最终的调用链是：

```text
用户问题
   |
   v
本地 Qwen3-0.6B
   |
   +-- search_travel_guide --> Milvus + RAG
   +-- get_weather_info     --> 和风天气
   +-- query_route          --> 高德地图
   |
   v
整合工具结果，生成旅行计划
```

这篇文章记录从环境准备、LoRA 训练、模型合并到本地 Agent 联调的完整过程。

## 一、为什么选择 Qwen3-0.6B

目标不是训练一个新的通用大模型，而是让一个足够小、可以在 RTX 4060 上本地运行的模型学会旅行助手的工作流：识别目的地和日期，按顺序调用工具，读取工具结果，再给出最终答复。

Qwen3-0.6B 的优点是体积小、下载和加载都快，适合做本地实验。代价是它的上下文和推理能力比在线大模型弱，长文本输入会让生成变慢，工具调用也更容易受到提示词长度和格式的影响。

Windows 上使用 Python 3.11，是为了让 PyTorch、Transformers、PEFT、Milvus Lite 等依赖都有成熟的二进制包。Python 3.14 并不是模型本身不能用，而是相关依赖的兼容性和预编译轮子还不够稳定。

## 二、准备环境

进入项目目录。`conda create` 本身可以在任意目录执行，但后续本地模型和数据路径最好都从项目根目录运行。

```powershell
cd E:\AI\Agent+SFT_SHENWEI

conda create -n qwen3-sft python=3.11 -y
conda activate qwen3-sft

python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

python -m pip install "transformers==4.51.0" "datasets==3.6.0" "accelerate==1.6.0" "peft==0.15.2" safetensors sentencepiece huggingface_hub colorlog
```

RAG 服务还需要 Milvus Lite。Windows 下仅安装 `pymilvus[milvus_lite]` 有时不会自动安装 Lite 组件，因此可以显式安装：

```powershell
python -m pip install "pymilvus==3.0.2" "milvus-lite==3.2.1" flask openai jieba requests
```

## 三、模型和训练数据

本次使用的基座模型在项目中的位置是：

```text
models\qwen3-0.6b-clean
```

训练数据没有重新生成，直接使用了项目已有的 `merged_train_final.json`。这个文件有 808 条样本，另外还有多轮对话版本：

```text
merged_train_final_multiturn.json
merged_train_final_multiturn_v2.json
```

一条合格的工具调用样本至少要包含四段关系：用户问题、助手的工具调用、工具返回结果、助手的最终回复。例如用户说“国庆节去烟台玩三天”，助手应该先生成结构化的 `search_travel_guide` 调用，而不是直接编造攻略。

这里有一个容易混淆的地方：训练数据不是旅游百科。它主要教模型识别意图、提取参数、选择工具和遵守调用顺序。具体景点、天气和路线仍然交给 RAG、天气接口和地图接口。

## 四、用 LoRA 训练行为

LoRA 不修改基座模型的全部参数，而是在目标层上增加一组小参数。这样训练显存和输出文件都小很多，适合先验证数据和工作流是否正确。

这次在 RTX 4060 上成功跑通的配置是：1 个 epoch、学习率 `1e-4`、单卡 batch size 为 1、梯度累积 8 步、FP16。命令如下：

```powershell
python .\train_qwen_last_assistant_lora.py `
  --train_file .\merged_train_final.json `
  --model_name_or_path .\models\qwen3-0.6b-clean `
  --output_dir .\outputs\qwen3-0.6b-existing-data-lora `
  --num_train_epochs 1 `
  --learning_rate 1e-4 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --fp16 `
  --local_files_only `
  --tools_file .\all_tools.json
```

如果训练数据已经为每条样本提供了工具定义，最后的 `--tools_file` 也可以删掉。

这次训练共 101 个优化步，训练时间约 25 分钟，最终记录的 `train_loss` 约为 `0.818`。LoRA 适配器约 80 MB，输出目录是：

```text
outputs\qwen3-0.6b-existing-data-lora
```

Loss 下降只能说明模型在训练集上的拟合变好，不能证明工具调用一定正确。真正的验收必须使用没有出现在训练日志里的新问题。

## 五、合并 LoRA，得到可部署模型

推理时可以加载基座模型和 adapter，也可以把 adapter 合并成一个完整模型。为了让旅行助手只需要读取一个目录，本次选择合并：

```powershell
python .\merge_lora_into_base.py `
  --base_model .\models\qwen3-0.6b-clean `
  --adapter_path .\outputs\qwen3-0.6b-existing-data-lora `
  --output_dir .\models\qwen3-0.6b-existing-data-merged `
  --local_files_only `
  --fp16
```

合并后的目录包含完整的模型权重和 tokenizer：

```text
models\qwen3-0.6b-existing-data-merged
```

## 六、启动 RAG 和外部工具

RAG 服务从 `rag-system` 目录启动，因为 Milvus 数据库使用相对路径 `./milvus_store.db`：

```powershell
cd E:\AI\Agent+SFT_SHENWEI\rag-system
conda activate qwen3-sft
python .\rag_api.py
```

另开一个 PowerShell 检查服务：

```powershell
curl.exe http://127.0.0.1:8010/health
```

正常结果应包含：

```json
{"status":"healthy","collection_size":361}
```

RAG 的向量检索需要 Embedding API Key。高德和和风天气也应该通过环境变量配置，文章中不放真实密钥：

```powershell
$env:DASHSCOPE_API_KEY="<your-dashscope-key>"
$env:AMAP_API_KEY="<your-amap-key>"
$env:QWEATHER_API_KEY="<your-qweather-key>"
$env:QWEATHER_API_HOST="https://your-account.re.qweatherapi.com"
```

实际项目联调时，曾经分别验证过：高德地理编码可以把“天坛”转换为经纬度，和风天气可以返回北京未来 3 天数据，RAG 可以根据城市名“烟台”返回 `city_code=370600` 的攻略。

## 七、启动本地旅行助手

主程序通过环境变量切换到本地 Qwen，并指向合并后的模型：

```powershell
cd E:\AI\Agent+SFT_SHENWEI
conda activate qwen3-sft

$env:TRAVEL_ASSISTANT_BACKEND="local_qwen"
$env:LOCAL_QWEN_MODEL_PATH="E:\AI\Agent+SFT_SHENWEI\models\qwen3-0.6b-existing-data-merged"

python .\travel_assistant_funcall_fixed.py
```

测试时可以输入：

```text
国庆节我想去烟台玩三天
```

然后补充日期。成功时，终端会先显示 `search_travel_guide` 的调用和烟台攻略，再输出三天行程。这个结果证明的不是“模型记住了烟台”，而是模型提取出了目的地，并把事实查询交给了 RAG。

## 八、为什么有时会慢，甚至像卡住

本地 Qwen 慢，通常不是 RAG 挂了，而是最终生成阶段的上下文太长。烟台攻略本身包含交通、住宿、行程、美食和预算，工具结果全部塞回 0.6B 模型后，输入变长，生成速度自然下降。

可以先把最大输出限制到 256 个 token，观察交互速度：

```powershell
$env:LOCAL_QWEN_MAX_NEW_TOKENS="256"
python .\travel_assistant_funcall_fixed.py
```

更稳的优化是让 RAG 返回更短的摘要，或者只返回最相关的一条攻略。生产环境还应该让程序明确编排“先查攻略，再查天气”，不要把关键顺序完全交给一个 0.6B 模型自行决定。

## 九、这次实践真正解决了什么

微调解决的是行为问题：模型知道什么时候调用工具、工具参数放在哪里、工具结果回来后该继续做什么。它不能替代天气 API，也不能保证模型永远记住长对话中的所有槽位。

RAG 解决的是知识问题：烟台的景点、交通和预算存放在 Milvus 中，需要时检索出来。知识会更新时，优先更新 RAG 数据，而不是重新训练模型。

工具解决的是现实世界的问题：天气和路线依赖实时接口，模型只负责决定调用哪个接口以及如何解释结果。这样做的好处是每一层都能单独测试：先测模型是否产出正确工具调用，再测接口是否返回 200，最后测模型能否整合结果。

这也是小模型做 Agent 时最重要的经验：不要要求它独自记住一切。让它负责有限、清晰的决策，把记忆、检索和执行交给更适合的组件，整个系统反而更容易理解和改进。

