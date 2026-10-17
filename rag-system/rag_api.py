#!/usr/bin/env python3
"""
RAG系统微服务API
提供旅游攻略查询接口
支持混合检索（向量检索 + 关键词检索）
"""

import os
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from pymilvus import connections, Collection
from openai import OpenAI
import logging
from collections import defaultdict
import jieba
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).rstrip("/")
EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIMENSIONS = 1024

# Milvus配置
# PyMilvus validates local URIs by their `.db` suffix, while Milvus Lite 3.x
# creates a directory at that path.  Keep the checked-in milvus.db legacy file
# untouched and use a separate path for the new database.
MILVUS_URI = os.getenv("RAG_MILVUS_URI", "./milvus_store.db")
COLLECTION_NAME = "travel_guides"

# Flask应用
app = Flask(__name__)

class RAGService:
    def __init__(self):
        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY must be set")
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL
        )
        self.collection = None
        self.setup_milvus()
        self.VECTOR_SIMILARITY_THRESHOLD = 0.35
        self.KEYWORD_SCORE_THRESHOLD = 1
        self.RRF_SCORE_THRESHOLD = 0.02
        # 初始化中文分词
        jieba.initialize()
    
    def setup_milvus(self):
        """设置Milvus连接（带重试机制）"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # 先断开已有连接
                try:
                    connections.disconnect("default")
                except:
                    pass
                
                # 等待一下再连接
                if attempt > 0:
                    logger.info(f"重试连接Milvus (第{attempt+1}次)...")
                    time.sleep(retry_delay)
                
                # 建立连接
                connections.connect(uri=MILVUS_URI)
                self.collection = Collection(COLLECTION_NAME)
                self.collection.load()
                
                logger.info(f"成功连接Milvus集合: {COLLECTION_NAME}")
                logger.info(f"集合中的数据量: {self.collection.num_entities}")
                return
                
            except Exception as e:
                logger.warning(f"连接Milvus失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"连接Milvus失败，已重试{max_retries}次: {e}")
                    raise
                retry_delay *= 2  # 指数退避
    
    def ensure_connection(self):
        """确保Milvus连接正常"""
        try:
            if self.collection is None:
                self.setup_milvus()
            else:
                # 测试连接
                _ = self.collection.num_entities
        except Exception as e:
            logger.warning(f"检测到连接问题，重新连接: {e}")
            self.setup_milvus()
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的embedding向量"""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text[:8192],
                dimensions=EMBEDDING_DIMENSIONS,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"获取embedding失败: {e}")
            return None
    
    def extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 使用jieba分词
        words = jieba.cut_for_search(text)
        # 过滤停用词和短词
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '里', '啊', '哦', '哈', '吧', '呢', '吗',"酒店"}
        keywords = [word.strip() for word in words if len(word.strip()) > 1 and word.strip() not in stop_words]
        return keywords
    
    def extract_locations(self, query: str) -> Dict[str, List[str]]:
        """从查询中提取位置信息（省份和城市）"""
        try:
            self.ensure_connection()
            
            # 获取所有省份和城市数据
            all_locations = self.collection.query(
                expr="",
                output_fields=["province_name", "city_name"],
                limit=1000  # 获取足够多的数据用于匹配
            )
            
            # 构建省份和城市列表
            provinces = set()
            cities = set()
            for location in all_locations:
                if location.get("province_name"):
                    provinces.add(location.get("province_name"))
                if location.get("city_name"):
                    cities.add(location.get("city_name"))
            
            # 在查询中查找匹配的省份和城市
            found_provinces = []
            found_cities = []
            
            for province in provinces:
                if province in query or province.replace('省', '').replace('市', '').replace('自治区', '') in query:
                    found_provinces.append(province)
            
            for city in cities:
                if city in query or city.replace('市', '').replace('县', '') in query:
                    found_cities.append(city)
            
            return {
                "provinces": found_provinces,
                "cities": found_cities
            }
            
        except Exception as e:
            logger.error(f"位置提取失败: {e}")
            return {"provinces": [], "cities": []}
    
    def keyword_search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """关键词搜索"""
        try:
            self.ensure_connection()
            
            # 提取查询关键词
            keywords = self.extract_keywords(query)
            if not keywords:
                return []
            
            # 构建关键词查询表达式
            # 使用content字段进行关键词匹配
            keyword_exprs = []
            for keyword in keywords:
                keyword_exprs.append(f'content like "%{keyword}%"')
            
            # 使用OR连接关键词（至少匹配一个关键词）
            expr = " or ".join(keyword_exprs)
            
            # 执行查询
            results = self.collection.query(
                expr=expr,
                output_fields=["city_code", "city_name", "province_name", "content"],
                limit=limit
            )
            
            # 计算关键词匹配分数
            scored_results = []
            for result in results:
                content = result.get("content", "").lower()
                score = 0
                matched_keywords = []
                
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in content:
                        # 简单的TF计分：关键词出现次数
                        count = content.count(keyword_lower)
                        score += count
                        matched_keywords.append(keyword)
                
                scored_results.append({
                    "city_code": result.get("city_code"),
                    "city_name": result.get("city_name"),
                    "province_name": result.get("province_name"),
                    "content": result.get("content"),
                    "keyword_score": score,
                    "matched_keywords": matched_keywords,
                    "location": f"{result.get('province_name')}-{result.get('city_name')}"
                })
            
            # 按关键词分数排序
            scored_results.sort(key=lambda x: x["keyword_score"], reverse=True)
            return scored_results
            
        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return []
    
    def reciprocal_rank_fusion(self, vector_results: List[Dict], keyword_results: List[Dict], 
                             vector_weight: float = 1.0, keyword_weight: float = 1.0, 
                             k: int = 60) -> List[Dict[str, Any]]:
        """RRF (Reciprocal Rank Fusion) 融合算法 - 支持权重调节"""
        # 创建文档ID到结果的映射
        doc_scores = defaultdict(float)
        doc_info = {}
        
        # 处理向量检索结果（应用向量权重）
        for rank, result in enumerate(vector_results):
            doc_id = f"{result['city_code']}_{result['city_name']}"
            rrf_score = vector_weight * (1.0 / (k + rank + 1))
            doc_scores[doc_id] += rrf_score
            doc_info[doc_id] = result
            doc_info[doc_id]['vector_rank'] = rank + 1
            doc_info[doc_id]['vector_score'] = result.get('score', 0)
            doc_info[doc_id]['vector_rrf_score'] = rrf_score
        
        # 处理关键词检索结果（应用关键词权重）
        for rank, result in enumerate(keyword_results):
            doc_id = f"{result['city_code']}_{result['city_name']}"
            rrf_score = keyword_weight * (1.0 / (k + rank + 1))
            doc_scores[doc_id] += rrf_score
            if doc_id not in doc_info:
                doc_info[doc_id] = result
            doc_info[doc_id]['keyword_rank'] = rank + 1
            doc_info[doc_id]['keyword_score'] = result.get('keyword_score', 0)
            doc_info[doc_id]['keyword_rrf_score'] = rrf_score
            doc_info[doc_id]['matched_keywords'] = result.get('matched_keywords', [])
        
        # 按RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 构建最终结果
        fused_results = []
        for doc_id, total_rrf_score in sorted_docs:
            result = doc_info[doc_id].copy()
            result['rrf_score'] = total_rrf_score
            result['search_type'] = 'hybrid'
            # 添加权重信息用于调试
            result['fusion_info'] = {
                'vector_weight': vector_weight,
                'keyword_weight': keyword_weight,
                'vector_rrf_contribution': result.get('vector_rrf_score', 0),
                'keyword_rrf_contribution': result.get('keyword_rrf_score', 0)
            }
            fused_results.append(result)
        
        return fused_results
    
    def location_priority_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """位置优先搜索"""
        try:
            # 1. 提取位置信息
            locations = self.extract_locations(query)
            found_provinces = locations["provinces"]
            found_cities = locations["cities"]
            
            # 2. 如果找到位置信息，优先按位置搜索
            if found_provinces or found_cities:
                location_results = []
                
                # 按省份搜索
                for province in found_provinces:
                    results = self.search_by_location(province=province, limit=limit)
                    for result in results:
                        result["match_type"] = "province_match"
                        result["matched_location"] = province
                        result["location_score"] = 1.0
                    location_results.extend(results)
                
                # 按城市搜索
                for city in found_cities:
                    results = self.search_by_location(city=city, limit=limit)
                    for result in results:
                        result["match_type"] = "city_match"
                        result["matched_location"] = city
                        result["location_score"] = 1.0
                    location_results.extend(results)
                
                # 去重（基于city_code + city_name）
                seen = set()
                unique_results = []
                for result in location_results:
                    key = f"{result.get('city_code')}_{result.get('city_name')}"
                    if key not in seen:
                        seen.add(key)
                        unique_results.append(result)
                
                if unique_results:
                    return {
                        "search_strategy": "location_priority",
                        "matched_locations": {
                            "provinces": found_provinces,
                            "cities": found_cities
                        },
                        "results": unique_results[:limit]
                    }
            
            # 3. 如果没有找到位置或位置搜索无结果，返回空结果
            return {
                "search_strategy": "hybrid_fallback",
                "matched_locations": {
                    "provinces": found_provinces,
                    "cities": found_cities
                },
                "results": []
            }
            
        except Exception as e:
            logger.error(f"位置优先搜索失败: {e}")
            return {
                "search_strategy": "error_fallback",
                "matched_locations": {"provinces": [], "cities": []},
                "results": []
            }
    
    def filter_by_threshold(self, results: List[Dict[str, Any]], search_type: str) -> List[Dict[str, Any]]:
        """根据阈值过滤结果"""
        if not results:
            return results
            
        filtered_results = []
        
        for result in results:
            should_keep = False
            
            if search_type == "vector":
                # 向量搜索阈值
                vector_score = result.get("score", 0)
                if vector_score >= self.VECTOR_SIMILARITY_THRESHOLD:
                    should_keep = True
                    result["filter_reason"] = f"vector_score({vector_score:.3f}) >= threshold({self.VECTOR_SIMILARITY_THRESHOLD})"
                else:
                    result["filter_reason"] = f"vector_score({vector_score:.3f}) < threshold({self.VECTOR_SIMILARITY_THRESHOLD})"
                    
            elif search_type == "keyword":
                # 关键词搜索阈值
                keyword_score = result.get("keyword_score", 0)
                if keyword_score >= self.KEYWORD_SCORE_THRESHOLD:
                    should_keep = True
                    result["filter_reason"] = f"keyword_score({keyword_score}) >= threshold({self.KEYWORD_SCORE_THRESHOLD})"
                else:
                    result["filter_reason"] = f"keyword_score({keyword_score}) < threshold({self.KEYWORD_SCORE_THRESHOLD})"
                    
            else:  # hybrid
                # 混合搜索使用RRF阈值
                rrf_score = result.get("rrf_score", 0)
                if rrf_score >= self.RRF_SCORE_THRESHOLD:
                    should_keep = True
                    result["filter_reason"] = f"rrf_score({rrf_score:.4f}) >= threshold({self.RRF_SCORE_THRESHOLD})"
                else:
                    result["filter_reason"] = f"rrf_score({rrf_score:.4f}) < threshold({self.RRF_SCORE_THRESHOLD})"
            
            if should_keep:
                filtered_results.append(result)
        
        return filtered_results
    
    def search(self, query: str, limit: int = 1, search_type: str = "hybrid", 
               vector_weight: float = 0.7, keyword_weight: float = 1.5,
               use_threshold: bool = True) -> List[Dict[str, Any]]:
        """智能搜索：位置优先 + 混合检索降级 + 阈值过滤"""
        try:
            if search_type == "vector":
                results = self.vector_search(query, limit * 2)  # 多获取一些用于过滤
            elif search_type == "keyword":
                results = self.keyword_search(query, limit * 2)
            elif search_type == "location":
                location_result = self.location_priority_search(query, limit)
                return location_result["results"]
            else:
                # 智能混合搜索
                location_result = self.location_priority_search(query, limit)
                
                if location_result["results"]:
                    # 位置搜索成功，直接返回（位置搜索不需要阈值过滤）
                    for result in location_result["results"]:
                        result["search_strategy"] = location_result["search_strategy"]
                        result["matched_locations"] = location_result["matched_locations"]
                    return location_result["results"]
                
                # 位置搜索失败，降级到混合检索
                logger.info(f"位置搜索无结果，降级到混合检索: {query}")
                
                vector_results = self.vector_search(query, limit * 3)  # 更多结果用于过滤
                keyword_results = self.keyword_search(query, limit * 3)
                
                if not vector_results and not keyword_results:
                    return []
                elif not vector_results:
                    results = keyword_results
                    search_type = "keyword"
                elif not keyword_results:
                    results = vector_results
                    search_type = "vector"
                else:
                    results = self.reciprocal_rank_fusion(
                        vector_results, keyword_results, 
                        vector_weight, keyword_weight
                    )
                
                # 为混合检索结果添加策略信息
                for result in results:
                    result["search_strategy"] = "hybrid_fallback"
                    result["matched_locations"] = location_result["matched_locations"]
            
            # 应用阈值过滤
            if use_threshold:
                results = self.filter_by_threshold(results, search_type)
                logger.info(f"阈值过滤后剩余结果数: {len(results)}")
            
            return results[:limit]
                    
        except Exception as e:
            logger.error(f"智能搜索失败: {e}")
            return []
    
    def vector_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """纯向量检索（原来的search方法）"""
        try:
            # 确保连接正常
            self.ensure_connection()
            
            # 获取查询向量
            query_embedding = self.get_embedding(query)
            if not query_embedding:
                return []
            
            # 搜索参数
            search_params = {"metric_type": "IP", "params": {}}
            
            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["city_code", "city_name", "province_name", "content"]
            )
            
            # 格式化结果
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "city_code": hit.entity.get("city_code"),
                        "city_name": hit.entity.get("city_name"), 
                        "province_name": hit.entity.get("province_name"),
                        "content": hit.entity.get("content"),
                        "score": float(hit.score),
                        "search_type": "vector",
                        "location": f"{hit.entity.get('province_name')}-{hit.entity.get('city_name')}"
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def search_by_location(self, province: str = None, city: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """根据地理位置搜索"""
        try:
            # 确保连接正常
            self.ensure_connection()
            
            # 构建过滤条件
            filter_expr = []
            if province:
                filter_expr.append(f'province_name == "{province}"')
            if city:
                filter_expr.append(f'city_name like "%{city}%"')
            
            expr = " and ".join(filter_expr) if filter_expr else ""
            
            # 执行查询
            results = self.collection.query(
                expr=expr,
                output_fields=["city_code", "city_name", "province_name", "content"],
                limit=limit
            )
            
            # 格式化结果
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "city_code": result.get("city_code"),
                    "city_name": result.get("city_name"),
                    "province_name": result.get("province_name"), 
                    "content": result.get("content"),
                    "location": f"{result.get('province_name')}-{result.get('city_name')}"
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"按位置搜索失败: {e}")
            return []

# 全局服务实例
rag_service = None

def get_rag_service():
    """获取RAG服务实例（懒加载）"""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        service = get_rag_service()
        return jsonify({
            "status": "healthy",
            "service": "RAG Travel Guide API",
            "collection_size": service.collection.num_entities if service.collection else 0
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/search', methods=['POST'])
def search():
    """混合搜索接口 - 支持权重调节"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "error": "缺少query参数",
                "status": "error"
            }), 400
        
        query = data['query']
        limit = data.get('limit', 5)
        search_type = data.get('search_type', 'hybrid')
        
        # 新增权重参数
        vector_weight = data.get('vector_weight', 1.0)
        keyword_weight = data.get('keyword_weight', 1.5)
        
        if not query.strip():
            return jsonify({
                "error": "查询内容不能为空",
                "status": "error"
            }), 400
        
        # 执行搜索
        service = get_rag_service()
        results = service.search(query, limit, search_type, vector_weight, keyword_weight)
        
        # 分析搜索策略和匹配信息
        search_strategy = "hybrid"  # 默认
        matched_locations = {"provinces": [], "cities": []}
        
        if results:
            first_result = results[0]
            search_strategy = first_result.get("search_strategy", "hybrid")
            matched_locations = first_result.get("matched_locations", {"provinces": [], "cities": []})
        
        return jsonify({
            "status": "success",
            "query": query,
            "search_type": search_type,
            "search_strategy": search_strategy,
            "matched_locations": matched_locations,
            "weights": {
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight
            },
            "limit": limit,
            "results_count": len(results),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"搜索接口错误: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/search_by_location', methods=['POST'])
def search_by_location():
    """按地理位置搜索接口"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "请提供搜索参数",
                "status": "error"
            }), 400
        
        province = data.get('province')
        city = data.get('city')
        limit = data.get('limit', 10)
        
        if not province and not city:
            return jsonify({
                "error": "请至少提供省份或城市参数",
                "status": "error"
            }), 400
        
        # 执行搜索
        service = get_rag_service()
        results = service.search_by_location(province, city, limit)
        
        return jsonify({
            "status": "success",
            "filters": {
                "province": province,
                "city": city
            },
            "limit": limit,
            "results_count": len(results),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"按位置搜索接口错误: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """获取数据库统计信息"""
    try:
        service = get_rag_service()
        total_count = service.collection.num_entities
        
        # 获取省份统计（简单查询前100条记录进行统计）
        sample_results = service.collection.query(
            expr="",
            output_fields=["province_name"],
            limit=100
        )
        
        provinces = {}
        for result in sample_results:
            province = result.get("province_name")
            if province:
                provinces[province] = provinces.get(province, 0) + 1
        
        return jsonify({
            "status": "success",
            "total_travel_guides": total_count,
            "sample_province_distribution": provinces,
            "collection_name": COLLECTION_NAME
        })
        
    except Exception as e:
        logger.error(f"获取统计信息错误: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "接口不存在",
        "status": "error",
        "available_endpoints": [
            "GET /health - 健康检查",
            "POST /search - 混合搜索 (支持vector/keyword/hybrid模式)",
            "POST /search_by_location - 按位置搜索",
            "GET /stats - 统计信息"
        ]
    }), 404

if __name__ == '__main__':
    print("启动RAG旅游攻略查询服务...")
    print("可用接口:")
    print("- GET  /health - 健康检查")
    print("- POST /search - 混合搜索 (支持vector/keyword/hybrid模式)")
    print("- POST /search_by_location - 按位置搜索") 
    print("- GET  /stats - 统计信息")
    
    # 启用多线程支持以提升并发处理能力
    app.run(host='0.0.0.0', port=8010, debug=False, use_reloader=False, threaded=True)
