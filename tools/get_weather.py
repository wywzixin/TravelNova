import os

import requests
from datetime import datetime

def get_weather_by_date_range(location: str, start_date: str, num_days: int, query_range: str = "30d"):
    """
    查询指定地点和时间段的天气情况

    :param location: 地点 (城市ID如 101010100，或 '116.41,39.92')
    :param start_date: 开始日期 (格式 'YYYY-MM-DD')
    :param num_days: 查询天数（从开始日期算起几天）
    :param query_range: API查询范围，可选: 3d / 7d / 10d / 15d / 30d，默认30d
    :return: 时间段内的天气情况列表（list of dict），未找到则返回 []
    """
    from datetime import datetime, timedelta
    
    api_host = os.getenv("QWEATHER_API_HOST", "").rstrip("/")
    api_key = os.getenv("QWEATHER_API_KEY", "")
    if not api_host or not api_key:
        raise ValueError("QWEATHER_API_HOST and QWEATHER_API_KEY must be set")

    url = f"{api_host}/v7/weather/{query_range}"
    params = {"location": location, "key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[weather] request failed: {exc}")
        return []

    if resp.status_code != 200 or data.get("code") != "200":
        error_code = data.get("code", str(resp.status_code))
        print(f"[weather] request failed (HTTP {resp.status_code}, code={error_code})")
        return []

    if "daily" not in data:
        print("[weather] response has no daily data")
        return []

    # 计算目标日期范围
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    target_dates = []
    for i in range(num_days):
        target_date = start_dt + timedelta(days=i)
        target_dates.append(target_date.strftime("%Y-%m-%d"))

    # 收集目标时间段内的天气数据
    weather_list = []
    for d in data["daily"]:
        if d["fxDate"] in target_dates:
            weather_info = {
                "日期": d["fxDate"],
                "日出": d["sunrise"],
                "日落": d["sunset"],
                "白天天气": d["textDay"],
                "夜间天气": d["textNight"],
                "最高温": d["tempMax"] + "℃",
                "最低温": d["tempMin"] + "℃",
            }
            weather_list.append(weather_info)
    
    # 按日期排序
    weather_list.sort(key=lambda x: x["日期"])
    return weather_list


def get_weather_by_date(location: str, date: str, days: str = "30d"):
    """
    查询指定地点和日期的天气情况（兼容原有接口）

    :param location: 地点 (城市ID如 101010100，或 '116.41,39.92')
    :param date: 日期 (格式 'YYYY-MM-DD')
    :param days: 预报天数，可选: 3d / 7d / 10d / 15d / 30d
    :return: 当日天气情况（dict），未找到则返回 None
    """
    weather_list = get_weather_by_date_range(location, date, 1, days)
    return weather_list[0] if weather_list else None


# 示例调用
if __name__ == "__main__":
    result = get_weather_by_date("101010100", "2025-09-19")
    if result:
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        print("未找到对应日期的天气数据")
