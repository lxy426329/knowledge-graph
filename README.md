# 眼科知识图谱 API 服务

基于 FastAPI + Neo4j 构建的眼科疾病知识图谱查询接口服务。

## 功能特性

- ✅ 实体关联查询：传入疾病/症状/检查/治疗实体名称，双向返回所有关联关系
- ✅ 疾病列表获取：获取库内全部眼科疾病名称
- ✅ 环境变量配置：支持通过 `.env` 文件配置数据库连接信息

## 技术栈

- **框架**: FastAPI
- **数据库**: Neo4j (知识图谱)
- **依赖**: py2neo, python-dotenv

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn py2neo python-dotenv
```

### 2. 配置环境

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填写你的 Neo4j 密码
```

### 3. 启动服务

```bash
python kg_api.py
```

服务启动时会**自动检查并导入数据**，无需手动运行构建脚本。

服务启动后访问: `http://localhost:8001`

## API 接口

### 查询实体关联

**GET** `/api/kg/query?entity_name=实体名称`

示例：
```bash
# 查询急性结膜炎的关联信息
curl "http://localhost:8001/api/kg/query?entity_name=急性结膜炎"
```

响应示例：
```json
{
    "code": 200,
    "msg": "查询成功",
    "data": {
        "有症状": ["结膜充血", "分泌物增多"],
        "需检查": ["裂隙灯检查"],
        "治疗方式": ["抗生素滴眼液"]
    }
}
```

### 获取所有疾病

**GET** `/api/kg/all_disease`

示例：
```bash
curl http://localhost:8001/api/kg/all_disease
```

### Swagger 文档

访问 `http://localhost:8001/docs` 查看交互式 API 文档。

## 项目结构

```
├── kg_api.py         # FastAPI 接口服务
├── kg_build.py       # 知识图谱数据导入
├── kg_query.py       # 图谱查询工具
├── kg_data.json      # 疾病三元组数据
├── .env              # 环境变量配置（不上传）
├── .env.example      # 配置模板
└── .gitignore        # Git 忽略规则
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `kg_api.py` | 对外 RESTful API 接口 |
| `kg_build.py` | 初始化/导入知识图谱数据 |
| `kg_query.py` | 底层 Cypher 查询封装 |
| `kg_data.json` | 疾病-关系-内容 三元组数据 |

## 使用示例

```python
import requests

# 查询疾病关联信息
response = requests.get(
    "http://localhost:8001/api/kg/query",
    params={"entity_name": "白内障"}
)
print(response.json())

# 获取所有疾病列表
response = requests.get("http://localhost:8001/api/kg/all_disease")
print(response.json())
```

## 注意事项

1. 确保 Neo4j 数据库服务已启动
2. `.env` 文件包含敏感信息，不要上传到版本控制
3. 服务默认运行在端口 `8001`，可在 `kg_api.py` 中修改