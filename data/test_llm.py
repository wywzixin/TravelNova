from llm import call_llm, get_model_list
# 可以支持的模型
#   - gpt-5
#   - claude-opus
#   - gemini-pro
#   - gemini-flash
#   - qwen-plus
#   - qwen-32b
#   - qwen-14b
#   - deepseek-v3
#   - deepseek-r1
#   - glm-4.5

# 获取模型列表
print("可用模型:", get_model_list())

# 简单调用示例
model = "gpt-5"  # 可以换成任意模型名
response = call_llm(model, "你是什么模型")
print(f"\n{model} 的回答:")
print(response)

# 带系统提示的调用
response_with_system = call_llm(
    "claude-opus", 
    "写一个Python函数计算斐波那契数列", 
    system_prompt="你是一个Python编程专家，请提供简洁的代码"
)
print(f"\nclaude-opus 的回答:")
print(response_with_system)

# 思考模型调用示例
# thinking_response = call_llm(
#     "deepseek-r1", 
#     "解释一下量子计算的基本原理", 
#     show_thinking=True
# )
# print(f"\ndeepseek-r1 的完整回答（包含思考过程）:")
# print(thinking_response)