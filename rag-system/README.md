# 旅游攻略RAG系统

基于Milvus向量数据库和阿里云Qwen Embedding的旅游攻略检索系统。

## 功能特性

- 🔍 **向量搜索**: 使用语义搜索查找相关旅游攻略
- 📍 **地理位置搜索**: 按省份或城市精确搜索
- 🚀 **高性能**: 基于Milvus向量数据库，支持高并发查询
- 🌐 **RESTful API**: 提供标准HTTP接口，易于集成
- 📊 **统计信息**: 提供数据库统计和健康检查接口

## 系统架构

```
旅游攻略文本 → Qwen Embedding → Milvus向量数据库 → API服务 → 客户端查询
```

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
conda create -n sft_rag python=3.10
conda activate sft_rag
python -m pip install -r requirements.txt

```

在项目根目录将 `.env.example` 复制为 `.env`，填写 `DASHSCOPE_API_KEY`。
`.env` 已加入 `.gitignore`，不会随源码提交。也可以直接设置同名环境变量。

### 2. 数据导入

```bash
# 批量导入旅游攻略到向量数据库
python import_travel_guides.py
```

导入过程会：
- 读取 `original_data/travel_guides/` 目录下的所有txt文件
- 使用阿里云Qwen Embedding生成向量
- 存储到Milvus向量数据库
- 每个文件作为一个独立的chunk，包含城市和省份metadata

### 3. 启动API服务

```bash
# 启动微服务
python rag_api.py
```

服务默认运行在 `http://localhost:8010`

### 4. 测试API

```bash
# 运行测试脚本
python test_api.py
```

## API接口文档

### 1. 健康检查

```http
GET /health
```

**响应**:
```json
{
    "status": "healthy",
    "service": "RAG Travel Guide API",
    "collection_size": 500
}
```

### 2. 向量搜索

```http
POST /search
```

**请求体**:
```json
{
    "query": "北京有什么好玩的景点",
    "limit": 5
}
```

**响应**:
```json
{
    "status": "success",
    "query": "北京有什么好玩的景点",
    "limit": 5,
    "results_count": 3,
    "results": [
        {
            "city_code": "110000",
            "city_name": "北京",
            "province_name": "北京市",
            "content": "# 北京市旅游攻略...",
            "score": 0.8567,
            "location": "北京市-北京"
        }
    ]
}
```

### 3. 按位置搜索

```http
POST /search_by_location
```

**请求体**:
```json
{
    "province": "浙江省",
    "city": "杭州",
    "limit": 10
}
```

**响应**:
```json
{
    "status": "success",
    "filters": {
        "province": "浙江省",
        "city": "杭州"
    },
    "results_count": 1,
    "results": [...]
}
```

### 4. 统计信息

```http
GET /stats
```

**响应**:
```json
{
    "status": "success",
    "total_travel_guides": 500,
    "sample_province_distribution": {
        "北京市": 1,
        "浙江省": 12,
        "江苏省": 15
    },
    "collection_name": "travel_guides"
}
```

## 使用示例

### Python客户端

```python
import requests

# 向量搜索
response = requests.post("http://localhost:8010/search", json={
    "query": "适合春天旅游的古城",
    "limit": 3
})
results = response.json()

# 按位置搜索
response = requests.post("http://localhost:8010/search_by_location", json={
    "province": "江苏省",
    "limit": 5
})
results = response.json()
```

### curl命令

```bash
# 向量搜索
curl -X POST http://localhost:8010/search \
  -H "Content-Type: application/json" \
  -d '{"query": "西湖旅游攻略", "limit": 3}'

# 按位置搜索
curl -X POST http://localhost:8010/search_by_location \
  -H "Content-Type: application/json" \
  -d '{"province": "浙江省", "limit": 5}'
```

## 配置说明

### 环境变量

可以通过修改脚本中的配置常量来自定义设置：

```python
# 阿里云配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://ws-iwycceiqo7wk46k5.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSIONS = 1024

# Milvus配置
MILVUS_URI = os.getenv("RAG_MILVUS_URI", "./milvus_store.db")  # 本地存储目录
COLLECTION_NAME = "travel_guides"
```

### 数据格式

旅游攻略文件命名格式：
```
{城市编码}_{城市名}_travel_guide.txt
```

例如：
- `110000_北京_travel_guide.txt`
- `330100_杭州_travel_guide.txt`

## 性能优化

1. **批量插入**: 导入脚本使用50条记录为一批进行插入
2. **索引优化**: 使用AUTOINDEX自动选择最优索引类型
3. **文本长度限制**: embedding输入文本限制为8192字符
4. **连接复用**: API使用连接池复用数据库连接

## 故障排除

### 常见问题

1. **连接失败**
   ```
   解决方案：检查Milvus服务是否启动，端口是否正确
   ```

2. **embedding获取失败**
   ```
   解决方案：检查阿里云API Key是否正确，网络连接是否正常
   ```

3. **内存不足**
   ```
   解决方案：减少batch_size，或增加系统内存
   ```

### 日志查看

API服务运行时会输出详细日志，包括：
- 请求处理情况
- 错误信息
- 性能统计

## 扩展功能

### 添加新的搜索策略

可以在`RAGService`类中添加新的搜索方法：

```python
def hybrid_search(self, query: str, location_filter: str = None):
    """混合搜索：结合向量搜索和位置过滤"""
    # 实现混合搜索逻辑
    pass
```

### 添加缓存

使用Redis缓存热门查询结果：

```python
import redis

class RAGService:
    def __init__(self):
        self.cache = redis.Redis()
        # ...
```

## 许可证

本项目仅供学习和研究使用。


import requests

# 基础搜索
response = requests.post('http://localhost:8010/search', json={
    "query": "成都熊猫基地游玩攻略",
    "limit": 3
})

# 带权重的混合搜索
response = requests.post('http://localhost:8010/search', json={
    "query": "西安兵马俑历史文化",
    "search_type": "hybrid",
    "vector_weight": 1.5,
    "keyword_weight": 0.7,
    "limit": 10
})

print(response.json())

请求体
POST：http://198.18.0.1:8010/search
{
  "query": "我想去天德湖",
  "search_type": "hybrid", 
  "vector_weight": 0.7,
  "keyword_weight": 1.5,
  "limit": 5
}
