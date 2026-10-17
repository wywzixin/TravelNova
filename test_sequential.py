#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试顺序工具调用版本的数据集转换脚本
"""

import json
import os
from convert_dataset_sequential import DatasetConverter

def test_hotel_sequential():
    """测试酒店工作流的顺序工具调用"""
    
    # 创建酒店查询测试数据
    test_data = [
        {
            "用户名字": "张伟",
            "用户所处城市": "101010100",
            "出发日期": "2025-09-20",
            "起点坐标": "116.481028,39.989643",
            "工作流": 3,
            "用户问题": "推荐一些北京500-800元的酒店",
            "是否追问": "否",
            "追问回答": None
        },
        {
            "用户名字": "李明",
            "用户所处城市": "101020100",
            "出发日期": "2025-09-25",
            "起点坐标": "121.473701,31.230416",
            "工作流": 3,
            "用户问题": "如家酒店怎么样？",
            "是否追问": "否",
            "追问回答": None
        },
        {
            "用户名字": "王芳",
            "用户所处城市": "101280101",
            "出发日期": "2025-10-01",
            "起点坐标": "113.280637,23.125178",
            "工作流": 3,
            "用户问题": "我需要订酒店",
            "是否追问": "是",
            "追问回答": "广州天河区，预算400-600元"
        }
    ]
    
    # 保存测试数据
    test_file = "test_sequential_dataset.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print("创建酒店工作流测试数据集完成")
    
    # 创建转换器
    converter = DatasetConverter(test_file, "test_sequential_converted")
    
    # 测试单个数据转换
    print("\n测试酒店工作流的顺序工具调用...")
    for i, item in enumerate(test_data):
        print(f"\n--- 测试数据 {i+1} ---")
        print(f"用户问题: {item['用户问题']}")
        print(f"是否追问: {item['是否追问']}")
        
        try:
            result = converter.convert_single_item(item)
            if result:
                print("✅ 转换成功")
                conversation = result["conversation"]
                print(f"对话轮数: {len(conversation)}")
                
                # 显示对话结构，特别关注工具调用顺序
                tool_call_count = 0
                for j, msg in enumerate(conversation):
                    role = msg["role"]
                    if role == "system":
                        print(f"  {j+1}. {role}: [系统提示词]")
                    elif role == "user":
                        print(f"  {j+1}. {role}: {msg['content'][:50]}...")
                    elif role == "assistant":
                        if "tool_calls" in msg:
                            tool_call_count += 1
                            tool_names = [tc["function"]["name"] for tc in msg["tool_calls"]]
                            print(f"  {j+1}. {role}: [工具调用{tool_call_count}] {tool_names}")
                        else:
                            print(f"  {j+1}. {role}: {msg['content'][:50]}...")
                    elif role == "tool":
                        print(f"  {j+1}. {role}: [工具结果]")
                
                # 检查是否符合顺序调用模式
                if item["工作流"] == 3 and "推荐" in item["用户问题"]:
                    assistant_msgs = [msg for msg in conversation if msg["role"] == "assistant" and "tool_calls" in msg]
                    if len(assistant_msgs) >= 2:
                        print("✅ 检测到顺序工具调用模式")
                        first_tools = [tc["function"]["name"] for tc in assistant_msgs[0]["tool_calls"]]
                        second_tools = [tc["function"]["name"] for tc in assistant_msgs[1]["tool_calls"]]
                        print(f"    第一轮工具: {first_tools}")
                        print(f"    第二轮工具: {second_tools}")
                    else:
                        print("⚠️  未检测到预期的顺序工具调用")
                        
            else:
                print("❌ 转换失败")
                
        except Exception as e:
            print(f"❌ 转换出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("\n测试完成")

if __name__ == "__main__":
    test_hotel_sequential()
