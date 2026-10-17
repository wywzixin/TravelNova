#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试数据生成
"""

import json
from generate_dataset import DatasetGenerator

def main():
    print("测试数据生成...")
    generator = DatasetGenerator()
    
    # 生成少量测试数据
    test_data = []
    test_data.extend(generator.generate_travel_planning_no_ask(2))
    test_data.extend(generator.generate_travel_planning_ask(1))
    test_data.extend(generator.generate_route_no_ask(1))
    test_data.extend(generator.generate_hotel_no_ask(1))
    
    print(f"生成了 {len(test_data)} 条测试数据")
    
    # 显示前几条数据
    for i, item in enumerate(test_data[:3]):
        print(f"\n示例 {i+1}:")
        print(json.dumps(item, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
