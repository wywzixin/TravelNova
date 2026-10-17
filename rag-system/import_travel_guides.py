#!/usr/bin/env python3
"""
批量导入旅游攻略到Milvus向量数据库
使用阿里云Qwen embedding model
支持多线程并发处理embedding
"""

import os
import json
import glob
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).rstrip("/")
EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIMENSIONS = 1024

# Milvus配置
# PyMilvus requires the local URI to end in `.db`; Milvus Lite 3.x creates a
# directory at that path. Keep the old milvus.db file intact.
MILVUS_URI = os.getenv("RAG_MILVUS_URI", "./milvus_store.db")  # 本地存储
COLLECTION_NAME = "travel_guides"

# 多线程配置
MAX_WORKERS = 10  # 10个线程

class TravelGuideImporter:
    def __init__(self):
        # 为了线程安全，每个线程都会创建自己的client
        self.city_mapping = {}
        self.load_city_mapping()
        self._lock = threading.Lock()  # 用于线程安全的计数器
        
    def load_city_mapping(self):
        """加载城市编码映射"""
        with open('city_code_mapping.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.city_mapping = data['城市编码映射']
        print(f"已加载 {len(self.city_mapping)} 个城市映射")
    
    def _create_client(self) -> OpenAI:
        """创建OpenAI客户端（线程安全）"""
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY must be set")
        return OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL
        )
    
    def get_embedding(self, text: str, client: OpenAI = None) -> List[float]:
        """获取文本的embedding向量"""
        if client is None:
            client = self._create_client()
            
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text[:8192],  # 限制文本长度
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
            print(f"删除已存在的集合: {COLLECTION_NAME}")
        
        # 定义集合schema
        fields = [
            # Milvus requires an INT64 primary key when auto_id is enabled.
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="city_code", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="city_name", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="province_name", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSIONS),
        ]
        
        schema = CollectionSchema(fields, description="旅游攻略向量数据库")
        collection = Collection(COLLECTION_NAME, schema, consistency_level="Bounded")
        
        # 创建索引
        index_params = {
            "index_type": "AUTOINDEX",
            "metric_type": "IP",
            "params": {}
        }
        collection.create_index("embedding", index_params)
        print(f"创建集合和索引: {COLLECTION_NAME}")
        
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
    
    def process_single_file(self, file_path: str) -> Tuple[bool, Any]:
        """处理单个文件（用于多线程）"""
        # 每个线程创建自己的client
        client = self._create_client()
        
        try:
            # 解析文件名
            file_info = self.parse_filename(file_path)
            if not file_info:
                return False, f"无法解析文件名: {file_path}"
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                return False, f"文件内容为空: {file_path}"
            
            # 获取embedding
            embedding = self.get_embedding(content, client)
            if not embedding:
                return False, f"无法获取embedding: {file_path}"
            
            # 准备数据记录
            data_record = {
                'city_code': file_info['city_code'],
                'city_name': file_info['city_name'], 
                'province_name': file_info['province_name'],
                'content': content,
                'embedding': embedding
            }
            
            return True, data_record
            
        except Exception as e:
            return False, f"处理文件失败 {file_path}: {e}"
    
    def insert_batch_data(self, collection: Collection, batch_records: List[Dict]):
        """批量插入数据到Milvus（正确的格式）"""
        if not batch_records:
            return
            
        # 将记录按字段重新组织
        city_codes = []
        city_names = []
        province_names = []
        contents = []
        embeddings = []
        
        for record in batch_records:
            city_codes.append(record['city_code'])
            city_names.append(record['city_name'])
            province_names.append(record['province_name'])
            contents.append(record['content'])
            embeddings.append(record['embedding'])
        
        # Milvus期望的格式：按字段分组的列表
        data_to_insert = [
            city_codes,
            city_names,
            province_names,
            contents,
            embeddings
        ]
        
        collection.insert(data_to_insert)
    
    def process_travel_guides(self, data_dir: str):
        """处理所有旅游攻略文件（多线程版本）"""
        travel_guide_pattern = os.path.join(data_dir, "original_data/travel_guides/*_travel_guide.txt")
        files = glob.glob(travel_guide_pattern)
        print(f"找到 {len(files)} 个旅游攻略文件")
        
        collection = self.setup_milvus()
        
        # 多线程处理数据
        batch_size = 50
        batch_records = []
        processed_count = 0
        failed_count = 0
        
        print(f"开始使用 {MAX_WORKERS} 个线程并发处理...")
        
        # 使用ThreadPoolExecutor进行多线程处理
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_file = {executor.submit(self.process_single_file, file_path): file_path 
                             for file_path in files}
            
            # 使用tqdm显示进度
            with tqdm(total=len(files), desc="处理旅游攻略") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        success, result = future.result()
                        
                        if success:
                            batch_records.append(result)
                            
                            # 批量插入数据
                            if len(batch_records) >= batch_size:
                                try:
                                    self.insert_batch_data(collection, batch_records)
                                    with self._lock:
                                        processed_count += len(batch_records)
                                    batch_records = []
                                    print(f"已处理 {processed_count} 条记录")
                                except Exception as e:
                                    print(f"插入数据失败: {e}")
                                    batch_records = []
                        else:
                            with self._lock:
                                failed_count += 1
                            print(f"处理失败: {result}")
                            
                    except Exception as e:
                        with self._lock:
                            failed_count += 1
                        print(f"处理文件时出现异常 {file_path}: {e}")
                    
                    pbar.update(1)
        
        # 插入剩余数据
        if batch_records:
            try:
                self.insert_batch_data(collection, batch_records)
                processed_count += len(batch_records)
                print(f"已处理 {processed_count} 条记录")
            except Exception as e:
                print(f"插入最后批次数据失败: {e}")
        
        # 加载集合
        collection.load()
        print(f"\n=== 处理完成 ===")
        print(f"成功处理: {processed_count} 条记录")
        print(f"失败: {failed_count} 个文件")
        print(f"集合中实际记录数: {collection.num_entities}")
        
        return collection

def main():
    """主函数"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    importer = TravelGuideImporter()
    
    try:
        collection = importer.process_travel_guides(current_dir)
        
        # 简单验证
        print("\n=== 数据验证 ===")
        print(f"集合名称: {collection.name}")
        print(f"数据总数: {collection.num_entities}")
        
        # 查询示例
        search_params = {"metric_type": "IP", "params": {}}
        results = collection.search(
            data=[[0.1] * EMBEDDING_DIMENSIONS],  # 随机向量用于测试
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["city_name", "province_name"]
        )
        
        print("查询测试结果:")
        for hits in results:
            for hit in hits:
                print(f"- {hit.entity.get('province_name')}-{hit.entity.get('city_name')} (分数: {hit.score:.4f})")
                
    except Exception as e:
        print(f"导入过程出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
