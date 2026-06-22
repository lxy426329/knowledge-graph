#测外眼
from predict_module import predict_text
print(predict_text(r"E:\KG\qimo\fuxian\LangChain-ReAct-Agent-main\temp_images\913aaf879e75433cb81626e48b546460.png"))

# 测内眼
# from odir_api import ODIRPredictor
# p = ODIRPredictor()
# print(p.predict(r"E:\KG\qimo\fuxian\LangChain-ReAct-Agent-main\data\img\Im322_g_ACRIMA.jpg"))

# 测知识图谱（需要Neo4j跑起来之后）
# from kg_query import query_disease_knowledge
# print(query_disease_knowledge("白内障"))