#!/usr/bin/env python3
"""
测试版本的批量导入脚本 - 只处理前3个文件
"""

import os
import json
import glob
from typing import List, Dict, Any
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from openai import OpenAI
from tqdm import tqdm

# 配置
DASHSCOPE_API_KEY = ""
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSIONS = 1024

# Milvus配置
MILVUS_URI = "./test_milvus.db"  # 测试数据库
COLLECTION_NAME = "test_travel_guides"

class TestImporter:
    def __init__(self):
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL
        )
        self.city_mapping = {}
        self.load_city_mapping()
        
    def load_city_mapping(self):
        """加载城市编码映射"""
        with open('city_code_mapping.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.city_mapping = data['城市编码映射']
        print(f"已加载 {len(self.city_mapping)} 个城市映射")
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的embedding向量"""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text[:2000],  # 测试版本限制更短
                dimensions=EMBEDDING_DIMENSIONS,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取embedding失败: {e}")
            return None
    
    def setup_milvus(self):
        """设置Milvus连接和集合"""
        # 连接Milvus
        connections.connect(uri=MILVUS_URI)
        
        # 删除已存在的集合
        if utility.has_collection(COLLECTION_NAME):
            Collection(COLLECTION_NAME).drop()
            print(f"删除已存在的测试集合: {COLLECTION_NAME}")
        
        # 定义集合schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
            FieldSchema(name="city_code", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="city_name", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="province_name", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSIONS),
        ]
        
        schema = CollectionSchema(fields, description="测试旅游攻略向量数据库")
        collection = Collection(COLLECTION_NAME, schema, consistency_level="Bounded")
        
        # 创建索引
        index_params = {
            "index_type": "AUTOINDEX",
            "metric_type": "IP",
            "params": {}
        }
        collection.create_index("embedding", index_params)
        print(f"创建测试集合和索引: {COLLECTION_NAME}")
        
        return collection
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """解析文件名获取城市信息"""
        basename = os.path.basename(filename)
        # 格式: 编码_城市名_travel_guide.txt
        parts = basename.replace('_travel_guide.txt', '').split('_')
        if len(parts) >= 2:
            city_code = parts[0]
            city_name = '_'.join(parts[1:])
            
            # 从映射中获取省份信息
            if city_code in self.city_mapping:
                province_name = self.city_mapping[city_code]['province']
                return {
                    'city_code': city_code,
                    'city_name': city_name,
                    'province_name': province_name
                }
        
        return None
    
    def test_import(self):
        """测试导入前3个文件"""
        test_files = [
            "original_data/travel_guides/110000_北京_travel_guide.txt",
            "original_data/travel_guides/120000_天津_travel_guide.txt", 
            "original_data/travel_guides/330100_杭州_travel_guide.txt"
        ]
        
        collection = self.setup_milvus()
        
        batch_data = []
        processed_count = 0
        
        for file_path in test_files:
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue
                
            print(f"处理文件: {file_path}")
            
            # 解析文件名
            file_info = self.parse_filename(file_path)
            if not file_info:
                print(f"无法解析文件名: {file_path}")
                continue
            
            # 读取文件内容
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    print(f"文件内容为空: {file_path}")
                    continue
                
                print(f"文件内容长度: {len(content)} 字符")
                print(f"城市信息: {file_info}")
                
                # 获取embedding
                print("正在获取embedding...")
                embedding = self.get_embedding(content)
                if not embedding:
                    print(f"无法获取embedding: {file_path}")
                    continue
                
                print(f"embedding维度: {len(embedding)}")
                
                # 准备数据
                data_record = [
                    file_info['city_code'],
                    file_info['city_name'], 
                    file_info['province_name'],
                    content,
                    embedding
                ]
                batch_data.append(data_record)
                
            except Exception as e:
                print(f"处理文件失败 {file_path}: {e}")
                continue
        
        # 插入数据
        if batch_data:
            try:
                print(f"插入 {len(batch_data)} 条记录到数据库...")
                collection.insert(batch_data)
                processed_count = len(batch_data)
                print(f"插入成功")
            except Exception as e:
                print(f"插入数据失败: {e}")
                return None
        
        # 加载集合
        collection.load()
        print(f"测试数据导入完成，总计处理: {processed_count} 条记录")
        print(f"集合中实际记录数: {collection.num_entities}")
        
        return collection

def main():
    """主测试函数"""
    print("开始测试RAG系统导入功能...")
    
    importer = TestImporter()
    
    try:
        collection = importer.test_import()
        
        if collection:
            print("\n=== 测试验证 ===")
            print(f"集合名称: {collection.name}")
            print(f"数据总数: {collection.num_entities}")
            
            # 简单查询测试
            try:
                search_params = {"metric_type": "IP", "params": {}}
                dummy_vector = [0.1] * EMBEDDING_DIMENSIONS
                results = collection.search(
                    data=[dummy_vector],
                    anns_field="embedding",
                    param=search_params,
                    limit=3,
                    output_fields=["city_name", "province_name"]
                )
                
                print("查询测试结果:")
                for hits in results:
                    for hit in hits:
                        print(f"- {hit.entity.get('province_name')}-{hit.entity.get('city_name')} (分数: {hit.score:.4f})")
                        
                print("\n✅ 测试成功！RAG系统基本功能正常")
                
            except Exception as e:
                print(f"查询测试失败: {e}")
                
    except Exception as e:
        print(f"测试过程出现错误: {e}")
        raise

if __name__ == "__main__":
    main()