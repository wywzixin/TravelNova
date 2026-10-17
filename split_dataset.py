#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据集划分脚本
将数据集按类型划分为80%训练集和20%测试集
"""

import json
import random
from collections import defaultdict

def split_dataset():
    # 读取原始数据集
    with open('travel_assistant_dataset_20250914_112354.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("开始划分数据集...")
    print(f"原始数据总量: {len(data)}条")
    
    # 按类型分组数据
    grouped_data = defaultdict(list)
    
    for item in data:
        workflow = item['工作流']
        ask = item['是否追问']
        
        if workflow == 1:  # 旅行规划
            if ask == '否':
                key = '旅行规划（不需要反问）'
            else:
                key = '旅行规划（需要反问）'
        elif workflow == 2:  # 问路
            if ask == '否':
                key = '问路（不需要反问）'
            else:
                key = '问路（需要反问）'
        elif workflow == 3:  # 查询酒店
            if ask == '否':
                key = '查询酒店（不需要反问）'
            else:
                key = '查询酒店（需要反问）'
        elif workflow == 4:  # 旅行相关
            key = '旅行相关'
        elif workflow == 5:  # 拒答
            key = '拒答'
        else:
            key = f'未知工作流{workflow}'
        
        grouped_data[key].append(item)
    
    # 显示原始数据统计
    print("\n原始数据统计:")
    print("=" * 50)
    for key, items in grouped_data.items():
        print(f"{key}: {len(items)}条")
    
    # 设置随机种子确保可重现
    random.seed(42)
    
    train_data = []
    test_data = []
    train_stats = {}
    test_stats = {}
    
    # 对每个类型按80:20划分
    for key, items in grouped_data.items():
        # 打乱数据
        random.shuffle(items)
        
        total = len(items)
        train_size = int(total * 0.8)
        test_size = total - train_size
        
        # 划分训练集和测试集
        train_items = items[:train_size]
        test_items = items[train_size:]
        
        train_data.extend(train_items)
        test_data.extend(test_items)
        
        train_stats[key] = len(train_items)
        test_stats[key] = len(test_items)
    
    # 打乱最终的训练集和测试集
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # 保存训练集
    train_filename = 'travel_assistant_train_dataset.json'
    with open(train_filename, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存测试集
    test_filename = 'travel_assistant_test_dataset.json'
    with open(test_filename, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 显示划分结果
    print("\n数据集划分完成!")
    print("=" * 50)
    print(f"训练集: {len(train_data)}条 -> {train_filename}")
    print(f"测试集: {len(test_data)}条 -> {test_filename}")
    
    print("\n训练集统计:")
    print("-" * 30)
    for key in sorted(train_stats.keys()):
        print(f"{key}: {train_stats[key]}条")
    
    print("\n测试集统计:")
    print("-" * 30)
    for key in sorted(test_stats.keys()):
        print(f"{key}: {test_stats[key]}条")
    
    print("\n划分比例验证:")
    print("-" * 30)
    for key in sorted(train_stats.keys()):
        total = train_stats[key] + test_stats[key]
        train_ratio = train_stats[key] / total * 100
        test_ratio = test_stats[key] / total * 100
        print(f"{key}: 训练集{train_ratio:.1f}% ({train_stats[key]}/{total}), 测试集{test_ratio:.1f}% ({test_stats[key]}/{total})")
    
    print(f"\n总计: 训练集{len(train_data)}条, 测试集{len(test_data)}条")
    print(f"总体比例: 训练集{len(train_data)/(len(train_data)+len(test_data))*100:.1f}%, 测试集{len(test_data)/(len(train_data)+len(test_data))*100:.1f}%")

if __name__ == "__main__":
    split_dataset()
