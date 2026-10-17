# TravelNova

基于 Qwen3、Function Calling 和 RAG 的旅行助手，包含工具调用编排、微调数据构造、LoRA / 全参数训练、本地推理和 Langfuse 观测。

助手可查询城市攻略、天气、路线和酒店信息。模型支持本地 Qwen3 与在线 DeepSeek，攻略检索使用 Milvus 和 DashScope Embedding。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `travel_assistant_funcall_fixed.py` | 旅行助手入口与工具调用编排 |
| `local_qwen.py` | 本地 Qwen3 推理适配 |
| `tools/` | 天气、路线、酒店工具 |
| `rag-system/` | 攻略原文、城市映射、向量导入与检索 API |
| `data/` | 城市攻略生成脚本 |
| `generate_dataset.py`、`convert_dataset_final_fixed.py` | 对话数据生成与训练格式转换 |
| `train_*.py`、`run_train_*.sh` | 微调脚本 |
| `merged_*.json`、`travel_assistant_*dataset*.json` | 训练和测试数据 |
| `observability.py` | Langfuse 追踪与敏感字段脱敏 |
| `design/` | 静态界面设计稿 |

## 安装与配置

```powershell
git clone https://github.com/wywzixin/TravelNova.git
cd TravelNova
conda create -n qwen3-sft python=3.10
conda activate qwen3-sft
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写使用到的服务配置：

- `DASHSCOPE_API_KEY`：攻略向量生成和检索。
- `AMAP_API_KEY`：高德地理编码和路线查询。
- `QWEATHER_API_KEY`、`QWEATHER_API_HOST`：和风天气，Host 使用自己账号的 API Host。
- `DEEPSEEK_API_KEY`：选择在线 DeepSeek 后端时使用。
- `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`：可选的请求追踪。

`.env`、模型权重、数据库和运行日志均由 `.gitignore` 排除。仓库包含源码和数据，模型权重需要自行准备或按训练文档生成。

## 启动

先导入攻略并启动 RAG 服务：

```powershell
cd rag-system
python import_travel_guides.py
python rag_api.py
```

首次导入会调用 Embedding API，生成本地向量数据库。后续启动可直接运行 `rag_api.py`，服务地址为 `http://127.0.0.1:8010`。

另开终端，在项目根目录启动本地模型模式：

```powershell
conda activate qwen3-sft
$env:TRAVEL_ASSISTANT_BACKEND = "local_qwen"
$env:LOCAL_QWEN_MODEL_PATH = ".\models\qwen3-0.6b-existing-data-merged"
python travel_assistant_funcall_fixed.py
```

将 `LOCAL_QWEN_MODEL_PATH` 指向准备好的模型目录。也可使用 `.env` 中配置的 DeepSeek 密钥启动在线模式：

```powershell
$env:TRAVEL_ASSISTANT_BACKEND = "deepseek"
python travel_assistant_funcall_fixed.py
```

Windows 一键启动脚本为 `start_travel_assistant.bat`，使用 `qwen3-sft` Conda 环境及 `models/qwen3-0.6b-existing-data-merged` 模型目录。

## 数据与训练文档

- [微调数据构造流程](travel-assistant-sft-data-pipeline.md)
- [Qwen3 微调与本地 Agent 联调](travel-assistant-llm-finetuning.md)
- [LoRA 训练核心说明](qwen3-lora-training-core.md)
- [RAG 服务说明](rag-system/README.md)
- [Langfuse 观测说明](LANGFUSE.md)
