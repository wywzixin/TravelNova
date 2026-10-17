# Langfuse 观测

项目现在会自动从项目根目录的 `.env` 读取 Langfuse 配置。`.env` 已被 `.gitignore` 忽略，提交代码时只保留 `.env.example`。

启动前确认：

```powershell
python -m pip install -r .\requirements.txt
```

在 `TravelAssistantFuncCall.process_user_input()` 外层会创建一个 `travel-assistant` agent observation；其中包含：

- `agent.decide-tools`：模型决定是否调用工具；
- `agent.continue-tools`：需要时的第二轮工具决策；
- `agent.final-answer`：最终回答生成；
- `tool.<工具名>`：RAG、天气、路线、酒店等工具执行；
- `llm.<模型名>`：酒店工具内部通过 `llm_main` 发起的模型调用。

每个请求会使用一个 session id 关联多轮对话，并在请求结束时 flush。追踪输入和输出会截断过长文本，并遮蔽 `api_key`、`token`、URL 查询参数中的 `key` 等敏感字段。

临时关闭追踪：

```powershell
$env:LANGFUSE_TRACING_ENABLED = "false"
```

重新打开终端后即可恢复 `.env` 中的配置。
