---
title: 旅行助手的微调数据是怎么来的：从场景生成到工具调用轨迹
date: 2026-09-27
tags: [大模型, SFT, 数据集, Agent, 工具调用]
---

# 旅行助手的微调数据是怎么来的：从场景生成到工具调用轨迹

> 微调实战 · 第一篇：数据构造 → [第二篇：LoRA 训练原理](qwen3-lora-training-core.md) → [第三篇：微调与 Agent 联调](travel-assistant-llm-finetuning.md)
>
> [博客阅读](https://wywzixin.github.io/agent/travel-assistant-sft-data/) · [返回项目首页](README.md)

大模型微调最容易被误解的地方，是把“训练数据”理解成一堆问题和答案。对 Agent 来说，真正要学的是一条可执行的轨迹：什么时候应该追问，什么时候应该调用工具，工具参数如何组织，工具返回后怎样继续工作，最后怎样把结果说给用户。

这次旅行助手的数据流程正好说明了这一点。数据不是一次性写出来的，而是经过五层加工：先生成场景，再分层切分；让助手实际跑一遍工具调用；把运行过程保存成 ChatML 对话；最后做去重、检查和训练集整理。

```text
场景模板
   |
   v
1010 条结构化原始样本
   |
   v
按工作流分层切分：808 train / 202 test
   |
   v
旅行助手实际运行并捕获工具调用
   |
   v
conversation 格式 + tool_call/tool 结果
   |
   v
合并、检查、可选的多轮前缀扩展
   |
   v
LoRA SFT
```

## 一、先定义模型要学的行为

数据生成前，先把旅行助手拆成几个工作流。当前项目使用五类：

| 工作流 | 行为 | 是否需要工具 |
| --- | --- | --- |
| 1 | 旅行规划 | 旅行攻略、天气 |
| 2 | 问路导航 | 高德地图 |
| 3 | 酒店查询 | 酒店推荐和评价 |
| 4 | 旅行相关闲聊 | 通常直接回答 |
| 5 | 非旅行问题拒答 | 不调用旅行工具 |

每个场景还标记“是否追问”。例如“我需要住的地方”缺少城市，正确行为是先问城市；“杭州西湖附近预算 500 元的酒店”信息已经足够，就应该直接调用酒店工具。

这个设计先规定边界，再填内容。没有工作流标签，随机问题很容易变成一堆相似的闲聊；没有追问标签，模型会把缺少参数和参数完整的请求混为一谈。

## 二、生成第一层：结构化场景

`generate_dataset.py` 不调用大模型，它用城市、日期、问题模板和追问答案组合出原始场景。一次完整生成包含：

```text
旅行规划（不追问） 400
旅行规划（需要追问） 50
问路（不追问）     100
问路（需要追问）    20
酒店（不追问）     200
酒店（需要追问）    40
旅行闲聊           100
拒答               100
合计              1010
```

每条原始记录大致长这样：

```json
{
  "用户名字": "某个用户",
  "用户所处城市": "101220601",
  "出发日期": "2025-09-16",
  "起点坐标": "116.794664,33.971707",
  "工作流": 1,
  "用户问题": "我要去嘉兴度假，求推荐",
  "是否追问": "否",
  "追问回答": null
}
```

城市 ID、起点坐标和日期并不是装饰。它们会被转换器写进 system 消息，训练模型在问路和天气场景中使用这些上下文。

运行脚本：

```powershell
cd E:\AI\Agent+SFT_SHENWEI
conda activate qwen3-sft
python .\generate_dataset.py
```

脚本会生成带时间戳的 `travel_assistant_dataset_*.json`。本项目已经保留了一份 `travel_assistant_dataset_20250914_112354.json`，共 1010 条。脚本没有固定随机种子，所以重新运行时，城市、姓名、日期和问题组合都会变化；同一模板也可能被重复抽到。当前原始数据里就有一批重复或近似重复的场景，不能把记录条数直接当成独立语义数量。

## 三、第二层：按工作流分层切分

`split_dataset.py` 不是简单地把文件前 80% 放进训练集。它先按“工作流 + 是否追问”分组，再在每组内用随机种子 42 打乱，按 80:20 切分，最后再打乱训练集和测试集。

最终结果是：

```text
travel_assistant_train_dataset.json  808 条
travel_assistant_test_dataset.json   202 条
```

训练集仍然保持了工作流比例：旅行规划 360 条，问路 96 条，酒店 192 条，闲聊 80 条，拒答 80 条。测试集对应为 90、24、48、20、20 条。

这种分层切分有两个作用。第一，测试集不会因为随机波动而缺少某个工作流；第二，可以分别观察模型在“该追问”和“该直接调用”两类场景上的表现。

当前 `split_dataset.py` 的输入文件名写死为旧的时间戳。如果重新生成了新的原始文件，需要先把脚本里的输入文件改成实际文件名，或者复制成它期待的文件名再执行。更稳妥的做法是把输入路径改成命令行参数。

分层切分也不等于去重。由于问题来自有限模板，同一个问题或近似问题可能同时出现在训练集和测试集里。正式评估前，还应该按规范化后的用户问题、工具名和参数做一次去重，避免测试分数被模板记忆抬高。

## 四、第三层：让助手真实运行一遍

原始记录还不是训练对话。真正有价值的步骤是 `convert_dataset_final_fixed.py`：它为每条记录创建 `TravelAssistantFuncCall`，装载 system prompt 和工具定义，再把用户问题交给助手处理。

转换器会记录真实运行过程中发生的消息：

```text
system      用户信息、工作流和工具规则
user        用户问题
assistant   工具调用及 JSON 参数
tool        工具返回结果
assistant   最终回复
```

典型的工具调用轨迹是：

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "id": "call_xxx",
      "type": "function",
      "function": {
        "name": "query_route",
        "arguments": "{\"start_location\": \"116.48,39.98\", \"end_location\": \"火车站\"}"
      }
    }
  ]
}
```

紧接着必须有对应的工具消息：

```json
{
  "role": "tool",
  "tool_call_id": "call_xxx",
  "content": "从当前位置到火车站的路线……"
}
```

最后才是助手根据路线结果生成的自然语言答复。`tool_call_id` 必须和调用消息中的 ID 对上，否则训练模板无法准确表达“哪个结果属于哪个调用”。

在 Windows 项目目录中可以这样转换：

```powershell
python .\convert_dataset_final_fixed.py `
  .\travel_assistant_train_dataset.json `
  .\converted_train_final
```

转换器使用 8 个线程，并每处理一批就保存 `converted_batch_*.json`，这样中途出错时不会丢掉全部结果。当前代码会实例化旅行助手；如果要用在线模型生成高质量轨迹，需要提前设置对应的后端和 API Key。如果使用本地 Qwen，则要确认 RAG、地图和天气服务都已经可用。

这里还有一个实际边界：转换脚本调用的是“当前版本”的助手代码。以后修改了 system prompt、工具参数或调用顺序，再次转换同一批原始场景，得到的数据也会变化。因此应该把代码版本、模型版本和 API 配置一起记录，不能只保存最终 JSON。

## 五、第四层：合并批次文件

转换器输出的是多个数组文件，需要合并成训练脚本可以读取的单个 JSON：

```powershell
python -c "from merge_json_files import merge_json_files; merge_json_files(r'.\converted_train_final', r'.\merged_train_final.json')"
python -c "from merge_json_files import merge_json_files; merge_json_files(r'.\converted_test_final', r'.\merged_test_final.json')"
```

不要直接把目录里所有文件都丢给合并脚本。这个项目里存在 `converted_batch_001.json` 和 `converted_batch_001(1).json` 这样的备份副本，而 `merge_json_files.py` 会匹配目录下所有 `*.json`。如果不清理，样本会被重复合并，训练集大小和类别比例都会失真。

合并后，训练文件的结构应该是：

```json
[
  {
    "conversation": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "tool_calls": [...]},
      {"role": "tool", "tool_call_id": "...", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
]
```

本次用于 LoRA 的 `merged_train_final.json` 有 808 条记录；测试文件有 200 条有效对话记录。

## 六、第五层：扩展多轮对话前缀

`conversation_splitter.py` 用于从一条完整对话生成多个训练前缀。对于包含工具消息的对话，它会在每个 assistant 消息结束处截取一个片段；最后一个 assistant 片段额外复制两份，总共保留三份。

这样做的意图是让模型分别看到：

```text
看到用户问题后如何调用工具
看到工具结果后如何继续
完成全部步骤后如何回答
```

但这也是一种简单的过采样。它不会创造新的事实，只会改变同一条轨迹在训练集中的权重。项目中因此产生了两个多轮版本：

```text
merged_train_final_multiturn.json      1830 条
merged_train_final_multiturn_v2.json   2207 条
```

版本差异来自输入数据和拆分时机。使用扩展数据前要重新统计重复率，避免最后一轮回复被过度放大。

Windows 下可以直接调用函数，绕过脚本里写死的 Linux 路径：

```powershell
python -c "from conversation_splitter import split_conversations; split_conversations(r'.\merged_train_final.json', r'.\merged_train_final_multiturn_v2.json')"
```

## 七、RAG 知识库是另一条生成链

旅行攻略文本不是通过上面的 SFT 对话转换器生成的，它来自另一条 RAG 数据链。`data/generate_travel_guides.py` 读取城市编码映射，为每个城市构造提示词，再通过 `data/llm.py` 调用兼容 OpenAI 接口的 `deepseek-v3`，生成包含城市简介、交通、住宿、逐日行程、美食、购物、贴士和预算的攻略文件。

这些文本会写入：

```text
rag-system\original_data\travel_guides\{城市编码}_{城市名}_travel_guide.txt
```

随后 `rag-system/import_travel_guides.py` 为文本生成 Embedding，写入 Milvus。用户提问时，RAG API 先从 Milvus 找到相关城市和内容，再把结果交给旅行助手。它和 SFT 的关系是：SFT 教模型如何调用 `search_travel_guide`，RAG 提供调用之后的事实内容。

这条边界很重要。攻略内容更新时，优先重新生成或导入 RAG 数据，不需要重新训练 Qwen；只有工具选择、参数格式或回答流程发生变化时，才需要重新设计 SFT 样本。

## 八、训练 loader 如何使用这些对话

合并后的数据还要经过训练 loader。`inspect_qwen_dataset.py` 和训练脚本会读取 `conversation`，调用 Qwen tokenizer 的 chat template，把消息转换成模型输入。

训练标签不是整段对话都计算 loss。loader 会定位 assistant 回复的 token，只让 assistant 部分参与损失计算，system、user 和 tool 内容的 label 设为 `-100`。这样模型学习的是“应该怎样回复和调用”，而不是机械背诵用户问题或工具结果。

当前转换结果的工具 schema 没有放在每条样本的顶层 `tools` 字段里，而是写在 system prompt 中。对于缺少工具定义的样本，训练脚本可以通过 `all_tools.json` 注入全局工具列表。超长样本会被截断，若 system 规则或工具说明被截掉，模型的调用能力会明显下降，因此需要在训练前统计 token 长度。

## 九、训练前的质量检查

数据能被 JSON 解析，不代表它适合训练。至少要检查以下内容：

1. 每个 `tool_calls` 是否都有对应的 `tool` 消息。
2. `tool_call_id` 是否一一匹配。
3. 工具参数是否是合法 JSON，城市、日期和坐标是否存在。
4. 对于缺少目的地或预算的请求，助手是否真的追问，而不是猜一个值。
5. 拒答样本是否只拒绝非旅行任务，没有把正常旅行问题误判为无关问题。
6. 训练集和测试集是否有完全重复的用户问题和回复。
7. 是否混入了 `converted_batch_*(1).json` 等重复备份。
8. 工具返回内容里是否残留 API Key、请求 URL、调试堆栈或个人信息。

尤其要检查密钥。工具调用失败时，某些 API 会把完整的 `key=...` 放进异常 URL；如果不清理，这些字符串会被直接送进训练集。训练数据中出现过的 RapidAPI、高德或其他服务密钥都应该立即失效并重新生成。博客、公开仓库和模型发布包里都不应出现真实密钥。

可以先做一个轻量统计：

```powershell
python -c "import json; d=json.load(open('merged_train_final.json', encoding='utf-8')); print('samples=', len(d)); print('conversation=', sum('conversation' in x for x in d))"
```

再随机抽取几条，人工确认“问题—工具—结果—最终答复”这一条链完整存在。Agent 数据的质量，往往在这一步就已经决定了大半。

## 十、数据生成中的几个取舍

模板随机生成的优点是覆盖面广、成本低，可以快速构造“需要追问”和“必须拒答”等边界样本。它的缺点是语言变化有限，问题之间容易长得像，不能代替真实用户语料。

让助手真实调用工具的优点是轨迹格式自然，工具参数和返回结构更接近上线环境。缺点是速度慢、需要 API、结果会随时间变化，而且外部服务失败时可能把错误文本也写入数据。

多轮前缀扩展可以补足短对话，但它不会增加新的意图。如果训练集已经很大，重复复制最后回复可能造成过拟合，让模型过度偏爱某种收尾句式。

因此，数据量不是唯一目标。更重要的是让每种行为都有清晰样本，让成功调用、空结果、接口失败、缺少参数和拒答边界都被覆盖。

## 十一、从数据到模型，真正被学进去的是什么

最终用于 LoRA 的不是“烟台旅游百科”，而是这样的关系：

```text
用户表达目的地和日期
        |
        v
模型提取参数并选择工具
        |
        v
工具返回外部事实
        |
        v
模型整合事实并回答
```

微调让模型更稳定地执行这条关系；RAG、天气和地图服务负责提供会变化的事实。数据生成流程的价值，就在于把这条关系从代码和人工想象，变成模型可以反复看到的完整轨迹。

当模型又开始反复追问“你要去哪儿”，首先应该检查对话数据里有没有清晰的多轮样本、工具结果是否正确接回、system prompt 是否把已确认槽位传进去。只有确认数据和编排都正确后，才值得继续调 LoRA 参数。

