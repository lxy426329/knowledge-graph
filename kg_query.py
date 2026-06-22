# kg_query.py
from py2neo import Graph
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

# 连接库
graph = Graph("bolt://localhost:7687", auth=(DB_USER, DB_PWD))

def query_entity_relation(entity_name: str):
    cypher = """
    MATCH (n {name: $name})-[rel]->(m)
    RETURN type(rel) AS rel_type, m.name AS target_name
    UNION ALL
    MATCH (m)-[rel]->(n {name: $name})
    RETURN type(rel) AS rel_type, m.name AS target_name
    """
    res = graph.run(cypher, name=entity_name).data()
    relation_dict = {}
    for item in res:
        rt = item["rel_type"]
        tn = item["target_name"]
        if rt not in relation_dict:
            relation_dict[rt] = []
        relation_dict[rt].append(tn)
    return relation_dict

# 格式化打印疾病信息
def query_disease_knowledge(disease_name):
    rel_map = query_entity_relation(disease_name)
    text = f"\n===== {disease_name} =====\n"
    if rel_map.get("有症状"):
        text += "【常见症状】\n" + "、".join(rel_map["有症状"]) + "\n\n"
    if rel_map.get("需检查"):
        text += "【需要检查】\n" + "、".join(rel_map["需检查"]) + "\n\n"
    if rel_map.get("治疗方式"):
        text += "【治疗方式】\n" + "、".join(rel_map["治疗方式"]) + "\n"
    return text

def query_all_disease():
    cypher = "MATCH (d:Disease) RETURN d.name ORDER BY d.name"
    data = graph.run(cypher).data()
    name_list = [row["d.name"] for row in data]
    print(f"全部疾病共{len(name_list)}种：{name_list}")
    return name_list

if __name__ == "__main__":
    # 自测
    print(query_entity_relation("急性结膜炎"))
    print(query_disease_knowledge("白内障"))