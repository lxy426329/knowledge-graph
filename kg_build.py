from py2neo import Graph
import json
import os
from dotenv import load_dotenv

# 拿到当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
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

# 方案2：也调用一次dotenv作为补充
load_dotenv(dotenv_path=env_file_path, override=True)

DB_USER = os.getenv("NEO4J_USER")
DB_PWD = os.getenv("NEO4J_PWD")

if not DB_USER or not DB_PWD:
    raise FileNotFoundError(f".env文件读取失败，请检查文件内容和编码，路径：{env_file_path}")

# 连接Neo4j
graph = Graph("bolt://localhost:7687", auth=(DB_USER, DB_PWD))

# 自动获取当前文件所在目录
cur_path = os.path.dirname(os.path.abspath(__file__))
json_full_path = os.path.join(cur_path, "kg_data.json")

# 读取JSON三元组
with open(json_full_path, "r", encoding="utf-8") as f:
    eye_kg_data = json.load(f)

# 初始化数据库（MERGE自动去重）
def init_database():
    for disease, relation, content in eye_kg_data:
        cypher = """
        MERGE (d:Disease {name: $disease})
        SET d.cure_department = "眼科"
        MERGE (info:Content {name: $content})
        MERGE (d)-[r:`%s`]->(info)
        """ % relation
        graph.run(cypher, disease=disease, content=content)
    print("✅ 导入完成！已存在数据自动跳过，所有疾病标记为眼科")

# 查询函数完全保留你原有逻辑
def query_disease_knowledge(disease_name):
    cypher = """
    MATCH (dis:Disease {name: $name})-[rel]->(msg)
    RETURN dis.name, type(rel) AS rel_type, msg.name
    """
    res = graph.run(cypher, name=disease_name).data()

    result = {"有症状": [], "需检查": [], "治疗方式": []}
    for item in res:
        rel = item["rel_type"]
        if rel in result:
            result[rel].append(item["msg.name"])

    text = f"\n===== {disease_name} =====\n"
    if result["有症状"]:
        text += "【常见症状】\n" + "、".join(result["有症状"]) + "\n\n"
    if result["需检查"]:
        text += "【需要检查】\n" + "、".join(result["需检查"]) + "\n\n"
    if result["治疗方式"]:
        text += "【治疗方式】\n" + "、".join(result["治疗方式"]) + "\n"
    return text

if __name__ == "__main__":
    # 首次导入打开，用完注释
    init_database()

    print(query_disease_knowledge("急性结膜炎"))
    print(query_disease_knowledge("白内障"))
    print(query_disease_knowledge("麦粒肿"))
    print(query_disease_knowledge("上睑下垂"))