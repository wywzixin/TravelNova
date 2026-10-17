# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_main import call_llm
from typing import List, Dict, Any
import json
import random

def get_hotel_recommendations(user_requirements: str, model_name: str = None) -> List[Dict[str, Any]]:
    """
    获取酒店推荐列表
    
    Args:
        user_requirements: 用户的需求描述，如"上海商务酒店，预算1000元以内，近地铁站"
        model_name: 使用的LLM模型名称，默认为qwen-plus
        
    Returns:
        List[Dict]: 推荐的酒店列表，每个酒店包含名字、位置、价格、房型等信息
    """
    
    model_name = model_name or os.getenv(
        "HOTEL_LLM_MODEL",
        "deepseek-chat" if os.getenv("TRAVEL_ASSISTANT_BACKEND", "local_qwen").lower() in {"deepseek", "online"} else "local-qwen",
    )

    system_prompt = """你是一个专业的酒店推荐专家。根据用户需求，推荐合适的酒店。
请严格按照以下JSON格式返回酒店推荐列表：

[
  {
    "hotel_name": "酒店名称",
    "location": "具体位置/地址"
  }
]

请返回1个推荐酒店，确保数据真实合理。只返回JSON数组格式，不要其他文字。请用中文填写内容。也可以返回为空，表示没有找到合适的酒店。但是整体JSON格式要正确，只是key对应的value为空"""

    user_prompt = f"""用户需求：{user_requirements}

请根据以上需求推荐合适的酒店，返回JSON格式的酒店列表。"""

    try:
        response = call_llm(model_name, user_prompt, system_prompt, stream=False, show_thinking=False)
        
        # 尝试解析JSON响应
        try:
            # 提取JSON部分（去除可能的markdown格式）
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                hotels = json.loads(json_str)
                return hotels
            else:
                # 如果找不到JSON数组，尝试解析整个响应
                hotels = json.loads(response.strip())
                return hotels
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回模拟数据
            return _generate_fallback_hotels(user_requirements)
            
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        return _generate_fallback_hotels(user_requirements)


def get_hotel_reviews(hotel_name: str, model_name: str = None) -> List[Dict[str, Any]]:
    """
    获取酒店评论信息
    
    Args:
        hotel_name: 酒店名称
        model_name: 使用的LLM模型名称，默认为qwen-plus
        
    Returns:
        List[Dict]: 用户评论列表，包含评分、评论内容、用户信息等
    """
    
    model_name = model_name or os.getenv(
        "HOTEL_LLM_MODEL",
        "deepseek-chat" if os.getenv("TRAVEL_ASSISTANT_BACKEND", "local_qwen").lower() in {"deepseek", "online"} else "local-qwen",
    )

    system_prompt = """你是一个酒店评论数据专家。根据酒店名称，生成真实合理，包含了正面负面的用户评论。
请严格按照以下JSON格式返回评论列表：

[
  {
    "reviewer_name": "评论者昵称",
    "rating": 4,
    "review_date": "2024-12-15",
    "review_content": "详细的评论内容，要真实自然",
  }
]

请返回1条评论，确保评论内容真实自然，直接输出评价的文字即可。请用中文填写内容。3-5句话，要真实，可以正面也可以负面"""

    user_prompt = f"""用户问题：{hotel_name}怎么样

请回答这个问题，不要输出其他文字"""

    try:
        response = call_llm(model_name, user_prompt, system_prompt, stream=False, show_thinking=False)
        
        # 尝试解析JSON响应
        try:
            # 提取JSON部分（去除可能的markdown格式）
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                reviews = json.loads(json_str)
                return reviews
            else:
                # 如果找不到JSON数组，尝试解析整个响应
                reviews = json.loads(response.strip())
                return reviews
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回模拟数据
            return _generate_fallback_reviews(hotel_name)
            
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        return _generate_fallback_reviews(hotel_name)


def _generate_fallback_hotels(user_requirements: str) -> List[Dict[str, Any]]:
    """生成后备的酒店推荐数据"""
    fallback_hotels = [
        {
            "hotel_name": "假日酒店",
            "location": "市中心商业区",
            "price_range": "400-600元/晚",
            "room_types": ["标准间", "豪华间", "套房"],
            "rating": 4.2,
            "amenities": ["免费WiFi", "健身房", "早餐", "停车场"],
            "distance_to_transport": "地铁站500米"
        },
        {
            "hotel_name": "商务精选酒店",
            "location": "金融中心",
            "price_range": "600-900元/晚",
            "room_types": ["商务间", "行政套房"],
            "rating": 4.5,
            "amenities": ["商务中心", "会议室", "免费WiFi", "接送服务"],
            "distance_to_transport": "地铁站300米"
        },
        {
            "hotel_name": "经济型连锁酒店",
            "location": "交通枢纽附近",
            "price_range": "200-350元/晚",
            "room_types": ["标准间", "大床房"],
            "rating": 3.8,
            "amenities": ["免费WiFi", "24小时前台"],
            "distance_to_transport": "火车站200米"
        }
    ]
    return fallback_hotels


def _generate_fallback_reviews(hotel_name: str) -> List[Dict[str, Any]]:
    """生成后备的酒店评论数据"""
    fallback_reviews = [
        {
            "reviewer_name": "旅行达人小王",
            "rating": 4,
            "review_date": "2024-12-15",
            "review_content": "酒店位置很好，房间干净整洁，服务人员态度友好。早餐种类丰富，性价比不错。",
        },
        {
            "reviewer_name": "商务出差人",
            "rating": 5,
            "review_date": "2024-12-10",
            "review_content": "商务设施齐全，会议室很专业，网络稳定。房间安静，适合工作和休息。",
        },
    ]
    return fallback_reviews


# 示例用法
if __name__ == "__main__":
    # 测试酒店推荐功能
    print("=== 酒店推荐测试 ===")
    user_req = "北京商务酒店，预算800元以内，近地铁站，需要会议室"
    hotels = get_hotel_recommendations(user_req)
    for i, hotel in enumerate(hotels, 1):
        print(f"\n酒店 {i}:")
        print(f"名称: {hotel['hotel_name']}")
        print(f"位置: {hotel['location']}")
        print(f"价格: {hotel['price_range']}")
        print(f"评分: {hotel['rating']}")
    
    print("\n" + "="*50)
    
    # 测试酒店评论功能
    print("=== 酒店评论测试 ===")
    hotel_name = "北京假日酒店"
    reviews = get_hotel_reviews(hotel_name)
    for i, review in enumerate(reviews, 1):
        print(f"\n评论 {i}:")
        print(f"评论者: {review['reviewer_name']}")
        print(f"评分: {review['rating']}/5")
        print(f"评论: {review['review_content']}")
        print(f"入住目的: {review['stay_purpose']}")
