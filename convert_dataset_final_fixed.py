#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集格式转换脚本（最终修复版）
修复重复调用问题和追问逻辑
"""

import json
import os
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 使用修复版的旅行助手
from travel_assistant_funcall_fixed import TravelAssistantFuncCall


class DatasetConverter:
    def __init__(self, input_file: str, output_dir: str = "converted_datasets"):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 线程锁和进度跟踪
        self.lock = threading.Lock()
        self.processed_count = 0
        self.converted_data = []
        self.current_batch = 0
        self.total_count = 0
        self.progress_bar = None

    def load_dataset(self) -> List[Dict]:
        """加载原始数据集"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载 {len(data)} 条数据")
            self.total_count = len(data)
            return data
        except Exception as e:
            print(f"加载数据集失败: {e}")
            return []

    def convert_single_item(self, item: Dict) -> Dict:
        """转换单个数据项为目标格式"""
        try:
            # 创建旅行助手实例
            assistant = TravelAssistantFuncCall(
                user_name=item["用户名字"],
                user_city_id=item["用户所处城市"],
                travel_date_range=f"{item['出发日期']}~{item['出发日期']}",
                start_coordinates=item["起点坐标"]
            )
            
            # 构建包含用户信息的系统提示词
            user_info_text = f"""
## 用户信息
- 用户名: {item['用户名字']}
- 当前城市ID: {item['用户所处城市']}
- 出发日期: {item['出发日期']}
- 起点坐标: {item['起点坐标']}

请在处理用户请求时考虑这些信息，比如：
- 问路时如果没有明确起点，使用起点坐标{item['起点坐标']}
- 旅行规划时根据用户的出发日期范围提供建议
- 天气查询时根据旅行攻略中的天数来确定查询天数，使用时间段查询
- 路线查询时起点优先使用起点坐标
"""

            system_prompt = user_info_text + """
你是一个专业的旅行助手，严格按照以下工作流程处理用户请求：

## 工作流1: 旅行规划
**触发条件**: 用户想制定旅行计划、询问某地旅游攻略、景点推荐等
**处理流程**:
1. 首先检查是否有目的地信息
   - 如果没有目的地或目的地不明确，必须反问："请告诉我您想去哪个城市旅行？"
   - 如果用户问"附近有什么好玩的"，可以基于用户当前城市提供建议
   - 不要进行任何工具调用，直接反问
2. 如果有明确的目的地，检查是否有出行日期
   - 如果没有明确日期，可以基于当前日期推荐合适的出行时间
   - 如果完全没有时间信息，反问："请问您计划什么时候出行？"
3. 信息完整后，必须**按照顺序**调用以下两个工具：
   - `search_travel_guide`: 搜索目的地旅行攻略
   - `get_weather_info`: 查询出行日期的天气信息
4. 综合天气和攻略信息，制定详细的旅行计划，规划里面说明每一天的天气信息。不需要详细，只需要说明每天的温度，晴雨天等信息，无需根据天气改变景点顺序，如果search_travel_guide这个工具返回的结果为空，则不再调用天气工具，直接返回无相应旅行路线

## 工作流2: 问路/地图导航
**触发条件**: 用户询问路线、问路、导航、"怎么走"、"如何到达"等
**处理流程**:
1. 检查起点和终点信息，起点信息在起点坐标获得，无需询问，也无需询问城市，城市就是当前城市ID
   - 如果用户说"从X到Y"、"X到Y怎么走"，则X是起点，Y是终点，直接调用工具，只要说了大致的地点，比如说了火车站、医院、学校等信息，则无需追问，直接调用工具查询即可，无需追问具体在哪里
   - 如果用户说"怎么回家"，可以提醒用户提供具体地址
   - 如果完全没有重点的地点信息，才反问："请问您要去哪里？"能不追问尽量不追问，只要有信息则直接调用工具开始查询
2. 信息完整后，调用`query_route`工具获取路线
3. 为用户提供步行、公交、驾车等多种路线选择，如果query_route这个工具返回的结果为空，则直接返回查询不到对应路线

## 工作流3: 酒店查询
**触发条件**: 用户询问酒店推荐、酒店预订、住宿等
**处理流程**:
### 3A: 酒店推荐
1. 检查必要信息：目的地，只有在缺少城市的时候追问，其余时候都直接调用工具
   - 缺少目的地：反问"请问您要在哪个城市找酒店？"
   - 如果用户说"本地酒店"或"附近酒店"，使用当前城市

2. 信息收集完毕后，**必须按顺序执行**：
   - 第一步：调用`recommend_hotels`工具获取酒店推荐
   - 第二步：**立即**为推荐的酒店调用`get_hotel_reviews`工具获取评价
   - **重要**：不要在第一步后就返回结果，必须完成所有工具调用，如果第一步没有推荐结果，则直接返回"暂时没有找到符合您需求的酒店推荐"，无需进行第二步工具调用
3. 整合酒店信息和评价，生成完整的推荐结果（包含酒店详情+用户评价）

### 3B: 酒店评价查询
**触发条件**: 询问某酒店"怎么样"、"评价"、"好不好"等
1. 提取酒店名称
   - 如果没有明确酒店名称，反问："请问您想了解哪家酒店的评价？"，
2. 调用`get_hotel_reviews`获取评价信息

## 工作流4: 闲聊
**触发条件**: 旅行相关的一般性问题、打招呼等
**处理流程**:
1. 判断是否与旅行相关
   - 如果是旅行相关话题，直接回答，不调用工具
   - 如果完全无关（如数学、编程等），礼貌拒绝："抱歉，我是专门的旅行助手，只能回答旅行相关的问题。"

## 重要原则:
1. **信息不足时必须反问，不要猜测**
2. **一次只问一个关键信息，避免一次问太多**
3. **工具调用顺序很重要，旅行规划时必须先调用攻略，再根据返回情况决定是否调用天气**
4. **酒店推荐工作流特别重要**：
   - 先调用`recommend_hotels`获取酒店列表
   - 然后为推荐酒店调用`get_hotel_reviews`
   - 最后整合所有信息统一回复
   - 绝不在获得酒店推荐后立即回复，必须获取评价后才能给最终答案
5. **非旅行相关问题要礼貌拒绝**

## 反问示例:
- "请告诉我您想去哪个城市旅行？"
- "请问您计划什么时候出行？（请提供具体日期）"
- "请问您要去哪里？"
- "请问您要在哪个城市找酒店？"
- "请问您的预算范围是多少？"

严格按照以上流程处理用户请求，确保信息完整后再调用相应工具。"""
            
            # 初始化对话
            messages = [
                {
                    "role": "system",
                    "content": system_prompt.strip()
                },
                {
                    "role": "user", 
                    "content": item["用户问题"]
                }
            ]
            
            # 获取助手第一轮响应（修复版，避免重复调用）
            self._process_tools_fixed(assistant, item["用户问题"], messages)
            
            # 处理追问情况 - 修复逻辑
            if item["是否追问"] == "是" and item["追问回答"]:
                # 检查最后一个assistant消息是否是反问（没有工具调用）
                last_assistant_msg = None
                for msg in reversed(messages):
                    if msg["role"] == "assistant":
                        last_assistant_msg = msg
                        break
                
                # 只有工作流1（旅行规划）在反问时才继续处理追问
                if (item["工作流"] == 1 and 
                    last_assistant_msg and 
                    "tool_calls" not in last_assistant_msg):
                    
                    messages.append({
                        "role": "user",
                        "content": item["追问回答"]
                    })
                    
                    # 处理追问后的响应
                    self._process_tools_fixed(assistant, item["追问回答"], messages)
            
            return {"conversation": messages}
            
        except Exception as e:
            print(f"转换失败: {e}")
            traceback.print_exc()
            return None

    def _process_tools_fixed(self, assistant: TravelAssistantFuncCall, user_input: str, messages: List[Dict]):
        """处理工具调用（修复重复调用问题），直接修改messages列表"""
        try:
            # 保存原始方法
            original_call_function = assistant.call_function
            
            # 用于收集工具调用信息（避免重复调用）
            tool_call_results = {}  # 用字典存储，避免重复
            
            def capture_tool_calls(function_name, arguments):
                # 生成唯一的调用ID
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                
                # 检查是否已经调用过相同的工具和参数
                call_key = f"{function_name}_{json.dumps(arguments, sort_keys=True)}"
                
                if call_key not in tool_call_results:
                    # 执行工具调用
                    result = original_call_function(function_name, arguments)
                    
                    # 存储结果
                    tool_call_results[call_key] = {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False)
                        },
                        "result": result
                    }
                
                return tool_call_results[call_key]["result"]
            
            # 替换工具调用方法
            assistant.call_function = capture_tool_calls
            
            # 获取最终响应
            final_response = assistant.process_user_input(user_input)
            
            # 根据工具调用情况构建对话
            if tool_call_results:
                # 按调用顺序排序（根据工具调用的逻辑顺序）
                sorted_calls = list(tool_call_results.values())
                
                # 检查是否需要顺序调用（酒店工作流）
                hotel_calls = [call for call in sorted_calls 
                              if call["function"]["name"] in ["recommend_hotels", "get_hotel_reviews"]]
                
                if len(hotel_calls) > 1:
                    # 酒店工作流：分别处理每个工具调用
                    for call_data in sorted_calls:
                        # 添加工具调用
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": call_data["id"],
                                "type": call_data["type"],
                                "function": call_data["function"]
                            }]
                        })
                        
                        # 添加工具结果
                        messages.append({
                            "role": "tool",
                            "content": call_data["result"],
                            "tool_call_id": call_data["id"]
                        })
                else:
                    # 其他工作流：一次性处理所有工具调用
                    tool_calls = []
                    for call_data in sorted_calls:
                        tool_calls.append({
                            "id": call_data["id"],
                            "type": call_data["type"],
                            "function": call_data["function"]
                        })
                    
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls
                    })
                    
                    # 添加所有工具结果
                    for call_data in sorted_calls:
                        messages.append({
                            "role": "tool",
                            "content": call_data["result"],
                            "tool_call_id": call_data["id"]
                        })
                
                # 添加最终回复
                messages.append({
                    "role": "assistant",
                    "content": final_response
                })
            else:
                # 没有工具调用，直接回复
                messages.append({
                    "role": "assistant",
                    "content": final_response
                })
                
        except Exception as e:
            print(f"处理工具调用失败: {e}")
            # 如果出错，添加错误回复
            messages.append({
                "role": "assistant",
                "content": f"抱歉，处理您的请求时出现了错误：{e}"
            })

    def save_batch(self, data: List[Dict], batch_num: int):
        """保存批次数据"""
        try:
            filename = f"converted_batch_{batch_num:03d}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"保存批次 {batch_num}: {len(data)} 条数据 -> {filepath}")
            
        except Exception as e:
            print(f"保存批次 {batch_num} 失败: {e}")

    def update_progress(self):
        """更新进度"""
        with self.lock:
            self.processed_count += 1
            if self.progress_bar:
                self.progress_bar.update(1)

    def convert_worker(self, items: List[Dict], thread_id: int):
        """工作线程函数"""
        converted_items = []
        
        for i, item in enumerate(items):
            try:
                result = self.convert_single_item(item)
                if result:
                    converted_items.append(result)
                
                # 更新进度
                self.update_progress()
                
                # 更新全局计数
                with self.lock:
                    self.converted_data.extend([result] if result else [])
                    
                    # 每20条保存一次
                    if len(self.converted_data) >= 20:
                        self.current_batch += 1
                        self.save_batch(self.converted_data.copy(), self.current_batch)
                        self.converted_data.clear()
                    
                    # 每100条创建新的JSON文件
                    if self.processed_count % 100 == 0:
                        self.current_batch += 1
                
            except Exception as e:
                print(f"线程 {thread_id} 处理失败: {e}")
                self.update_progress()
                continue

    def convert_dataset(self, num_threads: int = 8):
        """主转换函数"""
        print("开始转换数据集...")
        
        # 加载原始数据
        raw_data = self.load_dataset()
        if not raw_data:
            print("没有数据需要转换")
            return
        
        # 初始化进度条
        self.progress_bar = tqdm(total=self.total_count, desc="转换进度")
        
        try:
            # 分配给各线程
            chunk_size = max(1, len(raw_data) // num_threads)
            threads_data = []
            
            for i in range(num_threads):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < num_threads - 1 else len(raw_data)
                if start_idx < len(raw_data):
                    threads_data.append(raw_data[start_idx:end_idx])
            
            # 启动线程池
            print(f"使用 {num_threads} 个线程并行转换...")
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                for thread_id, data_chunk in enumerate(threads_data):
                    if data_chunk:  # 确保有数据
                        future = executor.submit(self.convert_worker, data_chunk, thread_id)
                        futures.append(future)
                
                # 等待所有线程完成
                for future in futures:
                    future.result()
            
            # 保存剩余数据
            with self.lock:
                if self.converted_data:
                    self.current_batch += 1
                    self.save_batch(self.converted_data, self.current_batch)
        
        finally:
            if self.progress_bar:
                self.progress_bar.close()
        
        print(f"转换完成！总共处理了 {self.processed_count} 条数据")
        print(f"输出目录: {self.output_dir}")


def main():
    """主函数"""
    # 检查输入参数
    if len(sys.argv) < 2:
        print("用法: python convert_dataset_final_fixed.py <input_json_file> [output_dir]")
        print("示例: python convert_dataset_final_fixed.py travel_assistant_train_dataset.json converted_train_final")
        return
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "converted_datasets_final"
    
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return
    
    # 创建转换器并开始转换
    converter = DatasetConverter(input_file, output_dir)
    converter.convert_dataset(num_threads=8)


if __name__ == "__main__":
    main()
