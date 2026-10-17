import os

import requests

def _amap_key():
    key = os.getenv("AMAP_API_KEY", "")
    if not key:
        raise ValueError("AMAP_API_KEY must be set")
    return key

def geocode(address):
    """通过地名获取经纬度"""
    url = f"https://restapi.amap.com/v3/geocode/geo?address={address}&key={_amap_key()}"
    resp = requests.get(url).json()
    if resp["status"] == "1" and resp["geocodes"]:
        return resp["geocodes"][0]["location"]  # "lng,lat"
    else:
        raise ValueError(f"无法找到地址: {address}")

def get_walking(origin, destination):
    url = f"https://restapi.amap.com/v3/direction/walking?origin={origin}&destination={destination}&key={_amap_key()}"
    resp = requests.get(url).json()
    if resp["status"] == "1" and resp["route"]["paths"]:
        path = resp["route"]["paths"][0]
        steps = [s["instruction"] for s in path["steps"]]
        return {
            "总时间(分钟)": int(path["duration"]) // 60,
            "总距离(米)": int(path["distance"]),
            # "详细路线": steps
        }
    return None

def get_transit(origin, destination, city_code):
    url = f"https://restapi.amap.com/v3/direction/transit/integrated?origin={origin}&destination={destination}&city={city_code}&key={_amap_key()}"
    resp = requests.get(url).json()
    if resp["status"] == "1" and resp["route"]["transits"]:
        transit = resp["route"]["transits"][0]
        segments = []
        for seg in transit["segments"]:
            # 公交段
            if "bus" in seg and seg["bus"].get("buslines"):
                buslines = seg["bus"]["buslines"]
                if buslines:  # 确保不是空
                    busline = buslines[0]
                    segments.append(
                        f"乘坐 {busline['name']} "
                        f"({busline['departure_stop']['name']} 上车 → {busline['arrival_stop']['name']} 下车)"
                    )
            # 步行段（兼容 dict / list）
            if "walking" in seg and isinstance(seg["walking"], dict):
                if "distance" in seg["walking"]:
                    segments.append(f"步行 {seg['walking']['distance']} 米")

        return {
            "总时间(分钟)": int(transit["duration"]) // 60,
            "票价(元)": transit.get("cost", "未知"),
            "详细路线": segments
        }
    return None



def get_driving(origin, destination):
    url = f"https://restapi.amap.com/v3/direction/driving?origin={origin}&destination={destination}&extensions=all&key={_amap_key()}"
    resp = requests.get(url).json()
    if resp["status"] == "1" and resp["route"]["paths"]:
        path = resp["route"]["paths"][0]
        steps = [s["instruction"] for s in path["steps"]]
        return {
            "总时间(分钟)": int(path["duration"]) // 60,
            "总距离(米)": int(path["distance"]),
            "过路费(元)": path.get("tolls", "0"),
            # "详细路线": steps
        }
    return None

def query_routes(start_coords, end_place, city_code="110000"):
    destination = geocode(end_place)

    result = {
        "步行": get_walking(start_coords, destination),
        "公交": get_transit(start_coords, destination, city_code),
        "驾车/打车": get_driving(start_coords, destination)
    }
    return result


# 使用示例
if __name__ == "__main__":
    start = "116.481028,39.989643"   # 起点经纬度（望京SOHO）
    end = "天坛"
    routes = query_routes(start, end, city_code="110000")

    for mode, info in routes.items():
        print(f"\n【{mode}】")
        if info:
            for k, v in info.items():
                print(f"{k}: {v}")
        else:
            print("未查询到结果")
