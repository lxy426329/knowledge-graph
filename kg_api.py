from fastapi import FastAPI, Query
from py2neo import Graph
from typing import Dict, List, Any
from dotenv import load_dotenv
import os

# 拿到当前kg_api.py文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接.env完整绝对路径
env_file_path = os.path.join(current_dir, ".env")

# 先检查.env文件是否存在
if not os.path.exists(env_file_path):
    raise FileNotFoundError(f"找不到.env文件，路径：{env_file_path}")

# 方案1：手动解析.env（兼容BOM头，最稳健）
with open(env_file_path, "r", encoding="utf-8-sig") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()

# 方案2：也调用一次dotenv作为补充（override=True覆盖同名环境变量）
load_dotenv(dotenv_path=env_file_path, override=True)

DB_USER = os.getenv("NEO4J_USER")
DB_PWD = os.getenv("NEO4J_PWD")

# 调试打印
print(f".env文件路径：{env_file_path}")
print("读取用户：", DB_USER)
print("读取密码：", DB_PWD)

# 增加空值拦截，避免空参数连接数据库
if not DB_USER or not DB_PWD:
    raise FileNotFoundError(f".env文件读取失败，请检查文件内容和编码，路径：{env_file_path}")

try:
    graph = Graph("bolt://localhost:7687", auth=(DB_USER, DB_PWD))
except Exception as e:
    print("Neo4j数据库连接失败，请检查服务和账号密码", e)
    raise

# 自动检查并导入数据
def auto_init_database():
    try:
        # 检查数据库中是否已有数据
        result = graph.run("MATCH (d:Disease) RETURN COUNT(d) AS count").data()
        disease_count = result[0]["count"] if result else 0
        
        if disease_count == 0:
            print("⚠️ 数据库为空，正在自动导入数据...")
            json_full_path = os.path.join(current_dir, "kg_data.json")
            
            if not os.path.exists(json_full_path):
                print(f"❌ 找不到数据文件：{json_full_path}")
                return
            
            import json
            with open(json_full_path, "r", encoding="utf-8") as f:
                eye_kg_data = json.load(f)
            
            for disease, relation, content in eye_kg_data:
                cypher = f"""
                MERGE (d:Disease {{name: $disease}})
                SET d.cure_department = "眼科"
                MERGE (info:Content {{name: $content}})
                MERGE (d)-[r:`{relation}`]->(info)
                """
                graph.run(cypher, disease=disease, content=content)
            
            print("✅ 数据导入完成！")
        else:
            print(f"✅ 数据库已有 {disease_count} 种疾病数据")
    except Exception as e:
        print(f"⚠️ 自动导入失败（手动运行 kg_build.py 导入）：{e}")

auto_init_database()

# 实例化接口服务
app = FastAPI(
    title="眼科知识图谱查询接口",
    description="传入疾病、症状、检查、治疗实体名称，双向返回所有关联关系",
    version="1.0"
)

# 底层图谱查询逻辑
def get_entity_relation(entity_name: str) -> Dict[str, List[str]]:
    cypher = """
    MATCH (n {name: $name})-[rel]->(m)
    RETURN type(rel) AS rel_type, m.name AS target_name
    UNION ALL
    MATCH (m)-[rel]->(n {name: $name})
    RETURN type(rel) AS rel_type, m.name AS target_name
    """
    res = graph.run(cypher, name=entity_name).data()
    rel_result: Dict[str, List[str]] = {}
    for item in res:
        r_type = item["rel_type"]
        target = item["target_name"]
        if r_type not in rel_result:
            rel_result[r_type] = []
        rel_result[r_type].append(target)
    return rel_result

# 核心查询接口
@app.get("/api/kg/query", summary="实体关联查询接口")
def query_kg_entity(
    entity_name: str = Query(..., description="实体名称：疾病/症状/检查项目/治疗方式，例：急性结膜炎、结膜充血")
) -> Dict[str, Any]:
    try:
        data = get_entity_relation(entity_name.strip())
        if not data:
            return {"code": 404, "msg": "未匹配到该实体知识库数据", "data": {}}
        return {"code": 200, "msg": "查询成功", "data": data}
    except Exception as err:
        return {"code": 500, "msg": f"服务异常：{str(err)}", "data": {}}

# 辅助接口：全部疾病列表
@app.get("/api/kg/all_disease", summary="获取库内全部眼科疾病名称")
def get_all_disease() -> Dict[str, Any]:
    try:
        cypher = "MATCH (d:Disease) RETURN d.name ORDER BY d.name"
        rows = graph.run(cypher).data()
        disease_list = [row["d.name"] for row in rows]
        return {
            "code": 200,
            "msg": "获取疾病列表成功",
            "count": len(disease_list),
            "disease_list": disease_list
        }
    except Exception as err:
        return {"code": 500, "msg": f"服务异常：{str(err)}", "count":0, "disease_list":[]}

if __name__ == "__main__":
    import uvicorn
    # host=0.0.0.0 支持局域网其他设备访问
    uvicorn.run(app, host="0.0.0.0", port=8001)