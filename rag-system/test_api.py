#!/usr/bin/env python3
"""
RAG API测试客户端
用于测试旅游攻略查询服务
"""

import requests
import json
import time

# API服务地址
BASE_URL = "http://localhost:8010"

class RAGAPIClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self):
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ 服务健康状态良好")
                print(json.dumps(response.json(), indent=2, ensure_ascii=False))
                return True
            else:
                print(f"❌ 健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 连接服务失败: {e}")
            return False
    
    def search(self, query: str, limit: int = 3):
        """向量搜索测试"""
        print(f"\n🔍 搜索查询: '{query}'")
        try:
            payload = {
                "query": query,
                "limit": limit
            }
            response = self.session.post(
                f"{self.base_url}/search",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 搜索成功，找到 {result['results_count']} 条结果:")
                
                for i, item in enumerate(result['results'], 1):
                    print(f"\n{i}. {item['location']} (相似度: {item['score']:.4f})")
                    print(f"   内容预览: {item['content'][:100]}...")
                
                return result
            else:
                print(f"❌ 搜索失败: {response.status_code}")
                print(response.text)
                return None
                
        except Exception as e:
            print(f"❌ 搜索请求失败: {e}")
            return None
    
    def search_by_location(self, province: str = None, city: str = None, limit: int = 3):
        """按位置搜索测试"""
        print(f"\n📍 按位置搜索: 省份='{province}', 城市='{city}'")
        try:
            payload = {
                "province": province,
                "city": city,
                "limit": limit
            }
            response = self.session.post(
                f"{self.base_url}/search_by_location",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 搜索成功，找到 {result['results_count']} 条结果:")
                
                for i, item in enumerate(result['results'], 1):
                    print(f"\n{i}. {item['location']}")
                    print(f"   内容预览: {item['content'][:100]}...")
                
                return result
            else:
                print(f"❌ 按位置搜索失败: {response.status_code}")
                print(response.text)
                return None
                
        except Exception as e:
            print(f"❌ 按位置搜索请求失败: {e}")
            return None
    
    def get_stats(self):
        """获取统计信息"""
        print(f"\n📊 获取数据库统计信息")
        try:
            response = self.session.get(f"{self.base_url}/stats")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 统计信息:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                return result
            else:
                print(f"❌ 获取统计信息失败: {response.status_code}")
                print(response.text)
                return None
                
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return None

def main():
    """主测试函数"""
    print("🚀 开始测试RAG旅游攻略API")
    print("=" * 50)
    
    client = RAGAPIClient()
    
    # 1. 健康检查
    if not client.health_check():
        print("服务不可用，请先启动API服务")
        return
    
    print("\n" + "=" * 50)
    
    # 2. 统计信息
    client.get_stats()
    
    print("\n" + "=" * 50)
    
    # 3. 向量搜索测试
    test_queries = [
        "北京有什么好玩的景点",
        "西湖旅游攻略",
        "适合春天旅游的地方",
        "古城历史文化"
    ]
    
    for query in test_queries:
        client.search(query, limit=2)
        time.sleep(1)  # 避免请求过于频繁
    
    print("\n" + "=" * 50)
    
    # 4. 按位置搜索测试
    location_tests = [
        {"province": "浙江省", "city": None},
        {"province": None, "city": "北京"},
        {"province": "江苏省", "city": "南京"},
    ]
    
    for location in location_tests:
        client.search_by_location(
            province=location["province"],
            city=location["city"],
            limit=2
        )
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("🎉 API测试完成!")

if __name__ == "__main__":
    main()
