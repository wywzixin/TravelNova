#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行生成中国城市旅游攻略脚本
使用deepseek-v3模型为每个城市生成1-5天的详细旅游攻略
"""

import json
import os
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import threading
from llm import call_llm

# 全局锁，用于线程安全的文件操作和日志输出
file_lock = threading.Lock()
console_lock = threading.Lock()

class TravelGuideGenerator:
    def __init__(self, max_workers: int = 5):
        """
        初始化旅游攻略生成器
        
        Args:
            max_workers: 最大并行工作线程数
        """
        self.max_workers = max_workers
        project_root = Path(__file__).resolve().parents[1]
        self.data_dir = Path(os.getenv("TRAVEL_DATA_DIR", str(project_root / "rag-system")))
        self.city_mapping_file = Path(
            os.getenv("CITY_MAPPING_FILE", str(self.data_dir / "city_code_mapping.json"))
        )
        self.guides_dir = Path(
            os.getenv("TRAVEL_GUIDES_DIR", str(self.data_dir / "original_data" / "travel_guides"))
        )
        self.progress_file = Path(
            os.getenv("TRAVEL_GUIDE_PROGRESS_FILE", str(project_root / "data" / "generation_progress.json"))
        )
        
        # 确保输出目录存在
        self.guides_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载城市数据
        self.cities = self._load_cities()
        
        # 加载或初始化进度记录
        self.progress = self._load_progress()
        
        with console_lock:
            print(f"✓ 初始化完成，共{len(self.cities)}个城市")
            print(f"✓ 已完成{len(self.progress.get('completed', []))}个城市的攻略生成")
    
    def _load_cities(self) -> Dict[str, Dict]:
        """从城市映射文件加载城市信息"""
        try:
            with open(self.city_mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("城市编码映射", {})
        except Exception as e:
            with console_lock:
                print(f"❌ 加载城市数据失败: {e}")
            return {}
    
    def _load_progress(self) -> Dict:
        """加载生成进度记录"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                with console_lock:
                    print(f"⚠️ 加载进度文件失败: {e}")
        return {"completed": [], "failed": [], "last_update": ""}
    
    def _save_progress(self, city_code: str, city_name: str, success: bool):
        """保存生成进度"""
        with file_lock:
            try:
                # 更新进度记录
                if success:
                    if city_code not in self.progress.get("completed", []):
                        self.progress.setdefault("completed", []).append(city_code)
                    # 从失败列表中移除（如果存在）
                    if city_code in self.progress.get("failed", []):
                        self.progress["failed"].remove(city_code)
                else:
                    if city_code not in self.progress.get("failed", []):
                        self.progress.setdefault("failed", []).append(city_code)
                
                self.progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
                
                # 保存到文件
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(self.progress, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                with console_lock:
                    print(f"❌ 保存进度失败 ({city_name}): {e}")
    
    def _generate_prompt(self, city_name: str, city_info: Dict) -> Tuple[str, str]:
        """生成旅游攻略的提示词"""
        
        # 随机选择攻略天数（1-5天）
        days = random.randint(1, 5)
        
        system_prompt = """你是一位专业的旅游攻略规划师，擅长为游客制定详细实用的旅行计划。
请根据用户提供的城市信息，生成一份专业、详细、实用的旅游攻略。

攻略要求：
1. 内容要准确真实，基于该城市的实际情况
2. 包含景点推荐、美食推荐、住宿建议、交通指南等
3. 按天数合理安排行程，避免过于紧凑
4. 提供实用的旅行小贴士
5. 语言生动有趣，富有吸引力"""

        user_prompt = f"""请为{city_info.get('province', '')}的{city_name}制定一份{days}天的详细旅游攻略。

城市信息：
- 城市名称：{city_name}
- 行政级别：{city_info.get('level', '未知')}
- 所属省份：{city_info.get('province', '未知')}
- 攻略天数：{days}天

请按以下格式输出攻略：

# {city_name}{days}日游攻略

## 🏙️ 城市简介
[城市的基本介绍、历史文化背景、最佳旅行时间等]

## 🚗 交通指南
[如何到达该城市，以及市内交通方式]

## 🏨 住宿推荐
[推荐3-5个不同价位的住宿区域或酒店]

## 📅 详细行程安排

### 第1天：[主题]
- **上午**：[具体安排]
- **下午**：[具体安排] 
- **晚上**：[具体安排]
- **推荐美食**：[当地特色菜品]

[如果是多天行程，继续第2天、第3天等...]

## 🍽️ 必吃美食
[详细介绍当地特色美食，包括推荐餐厅]

## 🎁 购物推荐
[特产、纪念品购买建议]

## 💡 实用小贴士
[气候、着装、注意事项等实用信息]

## 💰 预算参考
[大致的花费预算，包含交通、住宿、餐饮、门票等]

请确保内容丰富详实，具有很强的实用性。"""

        return system_prompt, user_prompt
    
    def _generate_single_guide(self, city_code: str, city_info: Dict) -> bool:
        """为单个城市生成旅游攻略"""
        city_name = city_info.get("name", "未知城市")
        
        # 检查是否已经生成过
        if city_code in self.progress.get("completed", []):
            with console_lock:
                print(f"⏭️ {city_name} 攻略已存在，跳过")
            return True
        
        # 生成文件名（使用城市编码避免重名问题）
        filename = f"{city_code}_{city_name.replace('市', '')}_travel_guide.txt"
        filepath = os.path.join(self.guides_dir, filename)
        
        # 如果文件已存在，跳过
        if os.path.exists(filepath):
            with console_lock:
                print(f"⏭️ {city_name} 攻略文件已存在，跳过")
            self._save_progress(city_code, city_name, True)
            return True
        
        try:
            with console_lock:
                print(f"🚀 开始生成 {city_name} 的旅游攻略...")
            
            # 生成提示词
            system_prompt, user_prompt = self._generate_prompt(city_name, city_info)
            
            # 调用deepseek-v3生成攻略
            start_time = time.time()
            guide_content = call_llm(
                model_name="deepseek-v3",
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                show_thinking=False
            )
            end_time = time.time()
            
            # 检查生成结果
            if not guide_content or len(guide_content.strip()) < 500:
                raise ValueError("生成的攻略内容过短或为空")
            
            # 保存攻略到文件
            with file_lock:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# {city_name}旅游攻略\n")
                    f.write(f"城市编码: {city_code}\n")
                    f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"生成用时: {end_time - start_time:.2f}秒\n")
                    f.write(f"生成模型: deepseek-v3\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(guide_content)
            
            # 更新进度
            self._save_progress(city_code, city_name, True)
            
            with console_lock:
                print(f"✅ {city_name} 攻略生成完成 ({end_time - start_time:.2f}秒)")
            
            # 添加随机延迟，避免API调用过于频繁
            time.sleep(random.uniform(1, 3))
            
            return True
            
        except Exception as e:
            with console_lock:
                print(f"❌ {city_name} 攻略生成失败: {e}")
            
            # 更新失败记录
            self._save_progress(city_code, city_name, False)
            return False
    
    def generate_all_guides(self):
        """并行生成所有城市的旅游攻略"""
        if not self.cities:
            print("❌ 没有找到城市数据")
            return
        
        # 过滤掉已完成的城市
        pending_cities = {
            code: info for code, info in self.cities.items() 
            if code not in self.progress.get("completed", [])
        }
        
        if not pending_cities:
            print("🎉 所有城市的攻略都已生成完成！")
            return
        
        print(f"📋 准备为{len(pending_cities)}个城市生成旅游攻略...")
        print(f"🔧 使用{self.max_workers}个并行线程")
        print(f"🤖 使用模型: deepseek-v3")
        print("-" * 50)
        
        # 使用线程池并行处理
        success_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_city = {
                executor.submit(self._generate_single_guide, city_code, city_info): (city_code, city_info)
                for city_code, city_info in pending_cities.items()
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_city):
                city_code, city_info = future_to_city[future]
                city_name = city_info.get("name", "未知城市")
                
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    with console_lock:
                        print(f"❌ {city_name} 处理异常: {e}")
                    failed_count += 1
                
                # 显示总体进度
                total_processed = success_count + failed_count
                total_cities = len(pending_cities)
                progress_percent = (total_processed / total_cities) * 100
                
                with console_lock:
                    print(f"📊 总进度: {total_processed}/{total_cities} ({progress_percent:.1f}%) | "
                          f"成功: {success_count} | 失败: {failed_count}")
        
        # 最终统计
        print("\n" + "=" * 50)
        print("🎯 攻略生成完成!")
        print(f"✅ 成功生成: {success_count}个城市")
        print(f"❌ 生成失败: {failed_count}个城市")
        print(f"📁 攻略文件保存在: {self.guides_dir}")
        
        if failed_count > 0:
            print(f"\n⚠️ 失败的城市可以重新运行脚本进行重试")

    def retry_failed_cities(self):
        """重新生成失败的城市攻略"""
        failed_cities = self.progress.get("failed", [])
        if not failed_cities:
            print("😊 没有需要重试的城市")
            return
        
        print(f"🔄 开始重试{len(failed_cities)}个失败的城市...")
        
        # 清空失败记录，准备重试
        self.progress["failed"] = []
        self._save_progress("", "", True)  # 保存清空的失败列表
        
        # 重新生成失败的城市
        for city_code in failed_cities:
            if city_code in self.cities:
                city_info = self.cities[city_code]
                self._generate_single_guide(city_code, city_info)

def main():
    """主函数"""
    print("🎯 中国城市旅游攻略生成器")
    print("=" * 50)
    
    # 创建生成器实例
    generator = TravelGuideGenerator(max_workers=3)  # 可以调整并发数
    
    # 询问用户操作
    while True:
        print("\n请选择操作:")
        print("1. 生成所有城市攻略")
        print("2. 重试失败的城市")
        print("3. 查看生成进度")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            generator.generate_all_guides()
        elif choice == "2":
            generator.retry_failed_cities()
        elif choice == "3":
            completed = len(generator.progress.get("completed", []))
            failed = len(generator.progress.get("failed", []))
            total = len(generator.cities)
            print(f"\n📊 生成进度统计:")
            print(f"✅ 已完成: {completed}/{total} ({completed/total*100:.1f}%)")
            print(f"❌ 失败: {failed}")
            print(f"📅 最后更新: {generator.progress.get('last_update', '未知')}")
        elif choice == "4":
            print("👋 退出程序")
            break
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main()
