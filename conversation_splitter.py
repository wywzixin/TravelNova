#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话数据拆分脚本
功能：对merged_for_training_prompted.json进行操作，对于每一个元素，只要有一个assistant就拆分一次数据
拆分规则：
- 从system到第一个assistant的回复（包含）
- 从system到第二个assistant的回复（包含）
- 从system到最后一个assistant的回复（包含）
- 最后一轮的assistant复制2份（一共3份）
"""

import json
import copy
from typing import List, Dict, Any

def split_conversations(input_file: str, output_file: str):
    """
    拆分对话数据
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    print(f"开始处理文件: {input_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据包含 {len(data)} 个对话")
    
    result = []
    
    for idx, conversation_data in enumerate(data):
        if 'conversation' not in conversation_data:
            print(f"警告: 第 {idx} 个对话没有conversation字段，跳过")
            continue
            
        conversation = conversation_data['conversation']
        
        # 检查是否包含任何"role": "tool"的消息
        has_tool_messages = any(message.get('role') == 'tool' for message in conversation)
        if not has_tool_messages:
            print(f"第 {idx} 个对话没有tool消息，保持原样不拆分")
            # 没有tool消息的对话直接加入结果，不进行拆分
            result.append(conversation_data)
            continue
        
        # 找到所有assistant的位置
        assistant_indices = []
        for i, message in enumerate(conversation):
            if message.get('role') == 'assistant':
                assistant_indices.append(i)
        
        if not assistant_indices:
            print(f"警告: 第 {idx} 个对话没有assistant回复，跳过")
            continue
        
        print(f"处理第 {idx} 个对话，找到 {len(assistant_indices)} 个assistant回复")
        
        # 为每个assistant回复创建一个拆分的对话
        for assistant_idx, end_pos in enumerate(assistant_indices):
            # 创建从开始到当前assistant回复的对话片段
            split_conversation = {
                'conversation': conversation[:end_pos + 1]  # 包含当前assistant回复
            }
            result.append(split_conversation)
            
            # 如果是最后一个assistant回复，额外复制2份（总共3份）
            if assistant_idx == len(assistant_indices) - 1:
                # 第二份
                result.append(copy.deepcopy(split_conversation))
                # 第三份
                result.append(copy.deepcopy(split_conversation))
                print(f"  - 最后一轮assistant回复额外复制了2份")
    
    print(f"拆分完成，总共生成 {len(result)} 个对话片段")
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")
    
    # 打印统计信息
    print("\n统计信息:")
    print(f"原始对话数量: {len(data)}")
    print(f"拆分后对话数量: {len(result)}")
    print(f"扩展倍数: {len(result) / len(data):.2f}")

def main():
    """主函数"""
    input_file = "/root/autodl-tmp/Agent+SFT/merged_train_final.json"
    output_file = "/root/autodl-tmp/Agent+SFT/merged_train_final_multiturn_v2.json"
    
    try:
        split_conversations(input_file, output_file)
        print("\n✅ 拆分任务完成！")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 