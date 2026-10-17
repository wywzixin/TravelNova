#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于Function Call的旅行助手（修复重复调用问题）
使用DeepSeek模型自主判断并调用工具
"""

import json
import sys
import os
import re
import requests
import uuid
from typing import Dict, List, Any, Optional
from openai import OpenAI
from datetime import date, timedelta

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.get_route import query_routes, geocode
from tools.get_weather import get_weather_by_date
from tools.get_hotel import get_hotel_recommendations, get_hotel_reviews
from local_qwen import get_local_qwen_client
from observability import get_observability, safe_value, usage_details


def _default_travel_date_range() -> str:
    """Return a forecastable default range starting today."""
    today = date.today()
    return f"{today:%Y-%m-%d}~{today + timedelta(days=20):%Y-%m-%d}"


class TravelAssistantFuncCall:
    """基于Function Call的旅行助手"""
    
    def __init__(self, model_name: Optional[str] = None, user_name: str = "用户", user_city_id: str = "101010100", 
                 travel_date_range: Optional[str] = None, start_coordinates: str = "116.481028,39.989643",
                 backend: Optional[str] = None):
        travel_date_range = travel_date_range or _default_travel_date_range()
        self.backend = (backend or os.getenv("TRAVEL_ASSISTANT_BACKEND", "local_qwen")).strip().lower()
        self.model_name = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.conversation_history = []
        self.session_id = os.getenv("LANGFUSE_SESSION_ID", uuid.uuid4().hex)
        self.observability = get_observability()
        # Qwen3-0.6B has a small context window for following the long workflow
        # prompt.  Keep the few pieces of information collected during a
        # multi-turn trip request explicitly so a short answer such as
        # ``烟台`` is not forgotten on the next turn.
        self.trip_context: Dict[str, Any] = {
            "destination": None,
            "num_days": None,
            "date_text": None,
        }
        self.rag_api_url = os.getenv("RAG_API_URL", "http://127.0.0.1:8010").rstrip("/")
        
        # 用户信息
        self.user_info = {
            "name": user_name,
            "city_id": user_city_id,  # 城市ID，如北京101010100
            "travel_date_range": travel_date_range,  # 出发日期范围 2025-09-15～2025-10-05
            "start_coordinates": start_coordinates,  # 起点坐标 116.481028,39.989643
            "current_date": self._get_current_date()
        }
        
        # 解析出发日期范围
        self._parse_travel_dates()
        
        # 初始化模型客户端。默认使用本地Qwen，设置
        # TRAVEL_ASSISTANT_BACKEND=deepseek 可切回原来的在线模型。
        if self.backend in {"local", "local_qwen", "qwen", "qwen3"}:
            self.backend = "local_qwen"
            local_model_path = model_name if model_name and os.path.isdir(model_name) else None
            self.model_name = "qwen3-0.6b-local"
            self.client = get_local_qwen_client(local_model_path)
        elif self.backend in {"deepseek", "online"}:
            self.backend = "deepseek"
            self.client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
            )
        else:
            raise ValueError(
                f"不支持的TRAVEL_ASSISTANT_BACKEND: {self.backend}，"
                "可选 local_qwen 或 deepseek"
            )
        
        # 定义工具函数
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_travel_guide",
                    "description": "搜索旅行攻略信息，获取目的地的详细旅游信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "目的地城市名称，如'北京'、'上海'"
                            },
                            "search_mode": {
                                "type": "string",
                                "description": "搜索模式：vector(向量搜索)、keyword(关键词搜索)、hybrid(混合搜索)",
                                "enum": ["vector", "keyword", "hybrid"],
                                "default": "hybrid"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_info",
                    "description": "查询指定地点和时间段的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称，如'北京'、'上海'，或城市ID如'101010100'"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "开始日期，格式YYYY-MM-DD，如'2025-09-15'"
                            },
                            "num_days": {
                                "type": "integer",
                                "description": "查询天数，默认1天。根据旅行攻略中的行程天数确定"
                            }
                        },
                        "required": ["location", "start_date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_route",
                    "description": "查询两地之间的路线，包括步行、公交、驾车路线",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_location": {
                                "type": "string",
                                "description": "起点坐标，如'116.481028,39.989643'"
                            },
                            "end_location": {
                                "type": "string",
                                "description": "终点地址，如'颐和园'"
                            },
                            "city_code": {
                                "type": "string",
                                "description": "城市代码，默认'110000'(北京)",
                                "default": "110000"
                            }
                        },
                        "required": ["start_location", "end_location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recommend_hotels",
                    "description": "根据用户需求推荐酒店",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "requirements": {
                                "type": "string",
                                "description": "用户对酒店的需求描述，包括地点、预算、设施等"
                            }
                        },
                        "required": ["requirements"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_hotel_reviews",
                    "description": "获取指定酒店的用户评价信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hotel_name": {
                                "type": "string",
                                "description": "酒店名称"
                            }
                        },
                        "required": ["hotel_name"]
                    }
                }
            }
        ]
    
    def _get_current_date(self) -> str:
        """获取当前日期"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")
    
    def _parse_travel_dates(self):
        """解析出发日期范围"""
        try:
            if "~" in self.user_info["travel_date_range"]:
                start_str, end_str = self.user_info["travel_date_range"].split("~")
                self.user_info["travel_start_date"] = start_str.strip()
                self.user_info["travel_end_date"] = end_str.strip()
            else:
                # 如果没有范围，就作为单一日期
                self.user_info["travel_start_date"] = self.user_info["travel_date_range"]
                self.user_info["travel_end_date"] = self.user_info["travel_date_range"]
        except Exception as e:
            print(f"解析出发日期出错: {e}")
            # 使用默认值
            default_range = _default_travel_date_range()
            self.user_info["travel_start_date"], self.user_info["travel_end_date"] = default_range.split("~")
    
    def add_to_history(self, role: str, content: str):
        """添加对话历史"""
        self.conversation_history.append({"role": role, "content": content})
        # 保持对话历史在合理长度内
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _update_trip_context(self, user_input: str) -> None:
        """Extract the small amount of state needed across trip turns.

        The base Qwen model often treats a one-word answer to a follow-up
        question as a new, unrelated request.  Persisting destination and
        duration here lets the system message state the confirmed values on
        every turn.  The patterns are intentionally conservative and only run
        for travel-planning language, so route and hotel conversations are not
        accidentally changed.
        """
        text = re.sub(r"\s+", "", user_input or "")
        if not text:
            return

        # Duration in forms such as "3天" or "三日".  Keep the original text
        # for the model; this numeric value is only a useful reminder.
        duration = re.search(r"(\d{1,2})\s*(?:天|日)", text)
        if duration:
            self.trip_context["num_days"] = int(duration.group(1))

        planning_hint = any(
            marker in text for marker in ("规划", "旅行", "旅游", "行程", "攻略", "景点", "怎么玩", "游玩")
        )

        # A destination immediately following a planning verb, e.g. "规划烟台
        # 3天旅行" or "去杭州玩".
        destination = None
        match = re.search(
            r"(?:规划|安排|计划)(?:去|到|前往)?([\u4e00-\u9fff]{2,10}?)(?=(?:\d{1,2}\s*[天日]|旅行|旅游|行程|游玩|攻略|景点|$))",
            text,
        )
        if match:
            destination = match.group(1)
        if destination is None:
            match = re.search(
                r"(?:去|到|前往)([\u4e00-\u9fff]{2,10}?)(?=(?:\d{1,2}\s*[天日]|旅行|旅游|行程|游玩|攻略|景点|$))",
                text,
            )
            if match:
                destination = match.group(1)
        if destination is None and planning_hint:
            # Also accept "杭州旅游攻略" / "烟台怎么玩".
            match = re.search(
                r"([\u4e00-\u9fff]{2,10}?)(?=(?:旅游|旅行|行程|攻略|怎么玩|景点|游玩))",
                text,
            )
            if match:
                destination = match.group(1)

        # When the previous turn asked for a destination, a short city-only
        # answer is the destination.  Do not treat date-only answers as one.
        if (
            destination is None
            and self.trip_context.get("destination") is None
        ):
            if re.fullmatch(r"[\u4e00-\u9fff]{2,8}(?:市|县)?", text) and not re.search(r"(?:多少|几|天|日|月|号|日)", text):
                destination = text
        if destination:
            self.trip_context["destination"] = destination.rstrip("市县")

        if re.search(r"(?:月|日|号|日期|出发|入住|到\d+)", text):
            self.trip_context["date_text"] = user_input.strip()

    def _trip_context_text(self) -> str:
        """Render confirmed state as a compact instruction for the model."""
        values = []
        if self.trip_context.get("destination"):
            values.append(f"目的地={self.trip_context['destination']}")
        if self.trip_context.get("num_days"):
            values.append(f"天数={self.trip_context['num_days']}天")
        if self.trip_context.get("date_text"):
            values.append(f"日期={self.trip_context['date_text']}")
        if not values:
            return "尚未从对话中确认旅行目的地。"
        return (
            "已从之前对话确认旅行信息：" + "，".join(values) + "。"
            "这些信息已经确认，禁止再次询问目的地；请直接继续当前工作流。"
        )

    def search_travel_guide(self, location: str, search_mode: str = "hybrid") -> str:
        """搜索旅行攻略"""
        try:
            url = f"{self.rag_api_url}/search"
            data = {
                "query": location,
                "search_type": search_mode,
                "limit": 3
            }
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if "results" in result and result["results"]:
                    guides = []
                    for item in result["results"]:
                        if "content" in item:
                            guides.append(item["content"])
                    return "\n\n".join(guides)
                else:
                    return "未找到相关旅行攻略信息"
            else:
                return "旅行攻略查询服务暂时不可用"
        except Exception as e:
            return f"旅行攻略查询出错: {e}"
    
    def get_weather_info(self, location: str, start_date: str, num_days: int = 1) -> str:
        """获取天气信息"""
        try:
            from tools.get_weather import get_weather_by_date_range
            
            # 转换城市名到城市ID的映射
            city_id_map = {
                "北京": "101010100",
                "上海": "101020100",
                "广州": "101280101",
                "深圳": "101280601",
                "杭州": "101210101",
                "南京": "101190101",
            }
            
            # 确定城市ID
            if location in city_id_map:
                location_id = city_id_map[location]
            elif location.isdigit():
                location_id = location
            else:
                # 默认使用用户城市ID
                location_id = self.user_info["city_id"]
            
            weather_list = get_weather_by_date_range(location_id, start_date, num_days)
            
            if weather_list:
                weather_text = f"天气信息({location} - {start_date}起{num_days}天):\n"
                for i, weather in enumerate(weather_list, 1):
                    weather_text += f"\n第{i}天 ({weather['日期']}):\n"
                    weather_text += f"  白天: {weather['白天天气']}，夜间: {weather['夜间天气']}\n"
                    weather_text += f"  温度: {weather['最低温']} ~ {weather['最高温']}\n"
                return weather_text
            else:
                return f"无法获取{location}在{start_date}起{num_days}天的天气信息"
        except Exception as e:
            return f"天气查询出错: {e}"
    
    def query_route(self, start_location: str, end_location: str, city_code: str = "110000") -> str:
        """查询路线"""
        try:
            # 处理起点坐标
            if "," in start_location and len(start_location.split(",")) == 2:
                start_coords = start_location
            elif start_location == "当前位置":
                start_coords = self.user_info["start_coordinates"]
            else:
                try:
                    start_coords = geocode(start_location)
                except:
                    # 使用用户默认坐标
                    start_coords = self.user_info["start_coordinates"]
            
            try:
                routes = query_routes(start_coords, end_location, city_code)
            except ValueError as e:
                return f"路线查询出错: {e}"
            
            route_text = f"从 {start_location} 到 {end_location} 的路线:\n\n"
            
            for mode, info in routes.items():
                route_text += f"【{mode}】\n"
                if info:
                    if mode == "步行":
                        route_text += f"总时间: {info['总时间(分钟)']}分钟\n"
                        route_text += f"总距离: {info['总距离(米)']}米\n"
                    elif mode == "公交":
                        route_text += f"总时间: {info['总时间(分钟)']}分钟\n"
                        if info['票价(元)'] != "未知":
                            route_text += f"票价: {info['票价(元)']}元\n"
                        if '详细路线' in info and info['详细路线']:
                            route_text += "路线: " + " → ".join(info['详细路线'][:3]) + "\n"
                    elif mode == "驾车/打车":
                        route_text += f"总时间: {info['总时间(分钟)']}分钟\n"
                        route_text += f"总距离: {info['总距离(米)']}米\n"
                        if info['过路费(元)'] != "0":
                            route_text += f"过路费: {info['过路费(元)']}元\n"
                else:
                    route_text += "暂无路线信息\n"
                route_text += "\n"
            
            return route_text
        except Exception as e:
            return f"路线查询出错: {e}"
    
    def recommend_hotels(self, requirements: str) -> str:
        """推荐酒店"""
        try:
            hotels = get_hotel_recommendations(requirements)
            if hotels:
                hotel_text = "根据您的需求，推荐以下酒店:\n\n"
                for i, hotel in enumerate(hotels[:3], 1):
                    hotel_text += f"酒店{i}: {hotel['hotel_name']}\n"
                    hotel_text += f"位置: {hotel['location']}\n"
                    # 由于新的JSON格式可能只包含hotel_name和location，需要安全地获取其他字段
                    if hotel.get('price_range'):
                        hotel_text += f"价格: {hotel['price_range']}\n"
                    if hotel.get('rating'):
                        hotel_text += f"评分: {hotel['rating']}/5\n"
                    if hotel.get('amenities'):
                        hotel_text += f"设施: {', '.join(hotel['amenities'][:4])}\n"
                    if hotel.get('distance_to_transport'):
                        hotel_text += f"交通: {hotel['distance_to_transport']}\n"
                    hotel_text += "\n"
                return hotel_text
            else:
                return "暂时没有找到符合您需求的酒店推荐"
        except Exception as e:
            return f"酒店推荐出错: {e}"
    
    def get_hotel_reviews_func(self, hotel_name: str) -> str:
        """获取酒店评价"""
        try:
            reviews = get_hotel_reviews(hotel_name)
            if reviews:
                review_text = f"{hotel_name}的用户评价:\n\n"
                for i, review in enumerate(reviews[:2], 1):
                    review_text += f"评价{i}:\n"
                    review_text += f"评分: {review.get('rating', '未知')}/5\n"
                    review_text += f"评论: {review.get('review_content', '')}\n\n"
                return review_text
            else:
                return f"暂时没有找到{hotel_name}的用户评价信息"
        except Exception as e:
            return f"酒店评价查询出错: {e}"
    
    def _should_continue_tool_chain(self, function_name: str, result: str) -> bool:
        """判断是否需要继续工具调用链"""
        if function_name == "recommend_hotels":
            # 如果酒店推荐返回了有效结果，需要继续调用评价工具
            try:
                # 检查结果是否包含有效的酒店信息
                if "酒店" in result and ("位置" in result or "价格" in result):
                    return True
                # 检查是否是空结果
                if "没有找到" in result or "暂时没有" in result or len(result.strip()) < 20:
                    return False
            except:
                pass
            return False
        elif function_name == "search_travel_guide":
            # 如果旅行攻略返回了有效结果，需要继续调用天气工具
            try:
                if any(marker in result for marker in ("未找到", "不可用", "查询出错", "服务暂时不可用")) or len(result.strip()) < 50:
                    return False
                return True
            except:
                pass
            return False
        else:
            return False
    
    def call_function(self, function_name: str, arguments: dict) -> str:
        """调用函数，并把每次工具执行记录为 Langfuse tool observation。"""
        with self.observability.observation(
            name=f"tool.{function_name}",
            as_type="tool",
            input=arguments,
            metadata={"session_id": self.session_id, "backend": self.backend},
        ) as tool_observation:
            try:
                if function_name == "search_travel_guide":
                    result = self.search_travel_guide(
                        arguments.get("location"),
                        arguments.get("search_mode", "hybrid")
                    )
                elif function_name == "get_weather_info":
                    result = self.get_weather_info(
                        arguments.get("location"),
                        arguments.get("start_date"),
                        arguments.get("num_days", 1)
                    )
                elif function_name == "query_route":
                    result = self.query_route(
                        arguments.get("start_location"),
                        arguments.get("end_location"),
                        arguments.get("city_code", "110000")
                    )
                elif function_name == "recommend_hotels":
                    result = self.recommend_hotels(arguments.get("requirements"))
                elif function_name == "get_hotel_reviews":
                    result = self.get_hotel_reviews_func(arguments.get("hotel_name"))
                else:
                    result = f"未知函数: {function_name}"

                tool_observation.update(output=safe_value(result))
                return result
            except Exception as exc:
                tool_observation.update(
                    level="ERROR",
                    status_message=str(exc),
                    output={"error": str(exc)},
                )
                raise

    def _chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        observation_name: str,
    ) -> Any:
        """Call the configured model and record a generation observation."""
        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }
        if tools is not None:
            request_kwargs["tools"] = tools
        if tool_choice is not None:
            request_kwargs["tool_choice"] = tool_choice

        with self.observability.observation(
            name=observation_name,
            as_type="generation",
            input={"messages": messages, "tools": tools, "tool_choice": tool_choice},
            metadata={"session_id": self.session_id, "backend": self.backend},
            model=self.model_name,
        ) as generation:
            try:
                response = self.client.chat.completions.create(**request_kwargs)
                generation.update(
                    output=safe_value(response),
                    usage_details=usage_details(response),
                )
                return response
            except Exception as exc:
                generation.update(
                    level="ERROR",
                    status_message=str(exc),
                    output={"error": str(exc)},
                )
                raise

    def process_user_input(self, user_input: str) -> str:
        """处理用户输入，并创建一条包含模型与工具子节点的 trace。"""
        with self.observability.observation(
            name="travel-assistant",
            as_type="agent",
            input={"user_input": user_input},
            metadata={
                "session_id": self.session_id,
                "backend": self.backend,
                "model": self.model_name,
            },
        ) as trace:
            try:
                result = self._process_user_input_impl(user_input)
                trace.update(output=result)
                return result
            except Exception as exc:
                trace.update(
                    level="ERROR",
                    status_message=str(exc),
                    output={"error": str(exc)},
                )
                raise
            finally:
                self.observability.flush()

    def _process_user_input_impl(self, user_input: str) -> str:
        """处理用户输入"""
        # 添加用户输入到对话历史
        self.add_to_history("user", user_input)
        self._update_trip_context(user_input)
        
        # 构建消息列表
        user_info_text = f"""
## 用户信息
- 用户名: {self.user_info['name']}
- 当前城市ID: {self.user_info['city_id']}
- 出发日期: {self.user_info['travel_date_range']}
- 起点坐标: {self.user_info['start_coordinates']}

## 已确认的旅行槽位
{self._trip_context_text()}

请在处理用户请求时考虑这些信息，比如：
- 问路时如果没有明确起点，使用起点坐标{self.user_info['start_coordinates']}
- 旅行规划时根据用户的出发日期范围提供建议
- 天气查询时根据旅行攻略中的天数来确定查询天数，使用时间段查询
- 路线查询时起点优先使用起点坐标
"""

        messages = [
            {
                "role": "system", 
                "content": user_info_text + """
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
   - 如果用户说"从X到Y"、"X到Y怎么走"，则X是起点，Y是终点，直接调用工具，如果说了火车站、医院、学校等公共设施的信息则无需追问，直接调用工具查询即可，无需追问具体在哪里
   - 如果用户说"怎么回家"，可以提醒用户提供具体地址
   - 如果没有明确的终点，必须反问："请问您要去哪里？"
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

严格按照以上流程处理用户请求，确保信息完整后再调用相应工具。直接输出结果，不要思考/no_think"""
            }
        ]
        
        # 添加对话历史
        messages.extend(self.conversation_history)
        
        try:
            # 第一轮调用：检查是否需要工具调用
            response = self._chat_completion(
                messages,
                tools=self.tools,
                tool_choice="auto",
                observation_name="agent.decide-tools",
            )
            
            message = response.choices[0].message
            
            # 如果不需要工具调用，直接返回
            if not message.tool_calls:
                content = message.content
                self.add_to_history("assistant", content)
                return content
            
            # 处理工具调用
            tool_calls_info = []
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"🔧 调用工具: {function_name}")
                print(f"参数: {arguments}")
                
                # 调用工具并记录结果
                result = self.call_function(function_name, arguments)
                print(f"🔧 工具结果: {result[:200]}{'...' if len(result) > 200 else ''}")
                
                tool_calls_info.append({
                    "call": tool_call,
                    "result": result,
                    "should_continue": self._should_continue_tool_chain(function_name, result)
                })
            
            # 将第一轮工具调用添加到messages
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in message.tool_calls]
            })
            
            # 添加工具结果
            for info in tool_calls_info:
                messages.append({
                    "role": "tool",
                    "tool_call_id": info["call"].id,
                    "content": info["result"]
                })
            
            # 决定是否需要继续调用工具
            needs_more_tools = any(info["should_continue"] for info in tool_calls_info)
            
            if needs_more_tools:
                # 第二轮调用：可能触发更多工具调用
                second_response = self._chat_completion(
                    messages,
                    tools=self.tools,
                    tool_choice="auto",
                    observation_name="agent.continue-tools",
                )
                
                second_message = second_response.choices[0].message
                
                if second_message.tool_calls:
                    # 处理第二轮工具调用 - 修复重复调用问题
                    second_tool_results = []
                    for tool_call in second_message.tool_calls:
                        function_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        
                        print(f"🔧 继续调用工具: {function_name}")
                        print(f"参数: {arguments}")
                        
                        result = self.call_function(function_name, arguments)
                        print(f"🔧 工具结果: {result[:200]}{'...' if len(result) > 200 else ''}")
                        
                        second_tool_results.append({
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                    
                    # 添加第二轮工具调用到messages
                    messages.append({
                        "role": "assistant",
                        "content": second_message.content,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in second_message.tool_calls]
                    })
                    
                    # 添加第二轮工具结果（修复：不重复调用）
                    for tool_result in second_tool_results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_result["tool_call_id"],
                            "content": tool_result["content"]
                        })
            
            # 最终调用：生成最终回复
            final_response = self._chat_completion(
                messages,
                observation_name="agent.final-answer",
            )
            
            final_content = final_response.choices[0].message.content
            self.add_to_history("assistant", final_content)
            
            return final_content
                
        except Exception as e:
            error_msg = f"处理请求时出错: {e}"
            self.add_to_history("assistant", error_msg)
            return error_msg


def main():
    """主函数 - 交互式对话"""
    print("🌍 欢迎使用智能旅行助手！")
    print("我可以帮您：")
    print("1. 制定旅行计划（会自动查询天气和攻略）")
    print("2. 查询路线（问路导航）")
    print("3. 推荐酒店和查看评价")
    print("4. 回答旅行相关问题")
    
    # 获取用户信息
    user_name = input("\n请输入您的姓名（默认：旅行者）：").strip() or "旅行者"
    user_city_input = input("请输入您所在的城市（默认：北京）：").strip() or "北京"
    
    # 城市ID映射
    city_id_map = {
        "北京": "101010100", "上海": "101020100", "广州": "101280101",
        "深圳": "101280601", "杭州": "101210101", "南京": "101190101"
    }
    user_city_id = city_id_map.get(user_city_input, "101010100")
    
    default_range = _default_travel_date_range()
    travel_range = input(f"请输入出发日期范围（默认：{default_range}）：").strip() or default_range
    start_coords = input("请输入起点坐标（默认：116.481028,39.989643）：").strip() or "116.481028,39.989643"
    
    print(f"✅ 用户信息设置完成:")
    print(f"   姓名: {user_name}")
    print(f"   城市: {user_city_input} (ID: {user_city_id})")
    print(f"   出发范围: {travel_range}")
    print(f"   起点坐标: {start_coords}")
    print("输入 'quit' 或 'exit' 退出\n")
    
    assistant = TravelAssistantFuncCall(
        user_name=user_name,
        user_city_id=user_city_id,
        travel_date_range=travel_range,
        start_coordinates=start_coords
    )
    
    while True:
        try:
            user_input = input("您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', '再见']:
                print("🌍 感谢使用旅行助手，祝您旅途愉快！")
                break
            
            if not user_input:
                continue
            
            print("\n正在处理您的请求...")
            response = assistant.process_user_input(user_input)
            print(f"\n助手: {response}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n🌍 感谢使用旅行助手，祝您旅途愉快！")
            break
        except Exception as e:
            print(f"\n❌ 出现错误: {e}")
            print("请重新输入您的问题。\n")


if __name__ == "__main__":
    main()
