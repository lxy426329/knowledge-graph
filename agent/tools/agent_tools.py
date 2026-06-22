from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService
from datetime import datetime

rag = RagSummarizeService()

from odir_api import ODIRPredictor

_odir_predictor = None

def _get_odir_predictor():
    global _odir_predictor
    if _odir_predictor is None:
        _odir_predictor = ODIRPredictor()
    return _odir_predictor

# 知识图谱连接（延迟初始化，Neo4j不可用时降级为本地数据）
_kg_graph = None
_kg_available = False

def _get_kg_graph():
    global _kg_graph, _kg_available
    if _kg_graph is not None:
        return _kg_graph if _kg_available else None
    try:
        from py2neo import Graph
        import os
        from dotenv import load_dotenv
        load_dotenv(override=True)
        db_user = os.getenv("NEO4J_USER")
        db_pwd = os.getenv("NEO4J_PWD")
        if db_user and db_pwd:
            _kg_graph = Graph("bolt://localhost:7687", auth=(db_user, db_pwd))
            _kg_available = True
            return _kg_graph
    except Exception:
        pass
    _kg_available = False
    return None

# 本地降级数据（从kg_data.json提取的核心疾病）
_KG_FALLBACK = {
    "糖尿病性视网膜病变": {"有症状": ["飞蚊症","视力下降","视物变形","视野缺损","视力模糊","眼前有黑影飞舞","眼前有黑云遮挡","失明"], "需检查": ["体格检查","血糖测定","血生化","凝血功能检测","裂隙灯显微镜","眼底镜","眼底荧光血管造影","视力检查","眼压测定","血压测定"], "治疗方式": ["严格控制血糖","严格控制血压","严格控制血脂","药物治疗","激光治疗","手术治疗","降糖药物治疗","二甲双胍","羟苯磺酸钙","玻璃体切割手术","眼底激光治疗"]},
    "青光眼": {"有症状": ["眼胀","眼痛","畏光","流泪","视物模糊","虹视","视力减退","夜盲","视野缺损","头痛","恶心","呕吐"], "需检查": ["视野检查","眼压测量","房角镜检查","裂隙灯检查","超声检查","眼底照相机检查"], "治疗方式": ["药物治疗","缩瞳药物","毛果芸香碱","噻吗洛尔","激光治疗","激光周边虹膜切除术","手术治疗","小梁切除术","视神经保护治疗"]},
    "白内障": {"有症状": ["视力下降","视物模糊","重影","眩光","色觉改变","固定性黑影","视野缺损","眼前有黑影"], "需检查": ["视力检查","眼压测量","裂隙灯显微镜检查","对比敏感度试验","Amsler表检查","色觉检查"], "治疗方式": ["药物治疗","抗氧化剂","谷胱甘肽","吡诺克辛滴眼液","手术治疗","超声乳化白内障吸除术","人工晶状体植入术"]},
    "高血压视网膜病变": {"有症状": ["视力下降","进行性视力下降","视物模糊","头痛","头晕","复视","恶心","呕吐","心悸"], "需检查": ["体格检查","血压测量","血常规","血生化","眼底检查","眼底镜","荧光素眼底血管造影"], "治疗方式": ["一般治疗","控制体重","限制钠盐摄入","戒烟限酒","降压药物治疗","卡托普利","缬沙坦","硝苯地平","手术治疗"]},
    "病理性近视": {"有症状": ["远视力不清","飞蚊症","闪光","视力明显下降","视野缺损","视物变形","夜盲","白内障","青光眼","视网膜脱离"], "需检查": ["视力检查","睫状肌麻痹验光检查","视野检查","彩色眼底照相","光学相干断层扫描检查","眼底荧光素血管造影检查","裂隙灯检查"], "治疗方式": ["激光光凝治疗","抗血管内皮生长因子治疗","后巩膜加固术","玻璃体切除手术","巩膜扣带术"]},
    "急性结膜炎": {"有症状": ["眼部异物感","灼热感","畏光","流泪","眼疼痛","眼部发红","分泌物增多","结膜充血水肿","眼睑红肿"], "需检查": ["体格检查","眼部专科检查","细胞学检查","结膜刮片","病原学检查","细菌培养","药物敏感试验"], "治疗方式": ["一般治疗","生理盐水洗眼","药物治疗","抗生素","左氧氟沙星","抗病毒药物","阿昔洛韦","人工泪液"]},
    "麦粒肿": {"有症状": ["眼睑红肿","疼痛","眼睑硬结","压痛","皮肤出现脓点","流泪","畏光"], "需检查": ["眼科检查","裂隙灯检查","血液学检查"], "治疗方式": ["热敷","抗生素","左氧氟沙星滴眼液","红霉素眼膏","止痛药","布洛芬","手术治疗","切开排脓"]},
    "上睑下垂": {"有症状": ["上睑位置下移","遮盖瞳孔","额纹变深","仰头视物","视力下降"], "需检查": ["视力检查","眼球运动检查","上睑提肌功能测试","新斯的明试验"], "治疗方式": ["口服新斯的明","维生素B₁","肾上腺皮质激素","上睑提肌缩短术","额肌悬吊术"]},
}

def _kg_query_neo4j(entity: str, relation_type: str = "") -> str:
    """通过Neo4j查询知识图谱"""
    graph = _get_kg_graph()
    if graph is None:
        return None
    try:
        cypher = """
        MATCH (n {name: $name})-[rel]->(m)
        RETURN type(rel) AS rel_type, m.name AS target_name
        UNION ALL
        MATCH (m)-[rel]->(n {name: $name})
        RETURN type(rel) AS rel_type, m.name AS target_name
        """
        res = graph.run(cypher, name=entity).data()
        if not res:
            return None
        rel_map = {}
        for item in res:
            rt = item["rel_type"]
            tn = item["target_name"]
            if rt not in rel_map:
                rel_map[rt] = []
            rel_map[rt].append(tn)
        # 按relation_type过滤
        if relation_type:
            type_map = {"symptoms": "有症状", "diagnosis": "鉴别诊断", "examination": "需检查", "treatment": "治疗方式"}
            cn_type = type_map.get(relation_type, relation_type)
            if cn_type in rel_map:
                return f"{entity}的{cn_type}：{'、'.join(rel_map[cn_type])}"
            else:
                return f"未找到{entity}的{relation_type}关联信息"
        # 返回全部
        text = f"{entity}知识图谱信息：\n"
        if rel_map.get("有症状"):
            text += f"【常见症状】{'、'.join(rel_map['有症状'])}\n"
        if rel_map.get("需检查"):
            text += f"【需要检查】{'、'.join(rel_map['需检查'])}\n"
        if rel_map.get("治疗方式"):
            text += f"【治疗方式】{'、'.join(rel_map['治疗方式'])}\n"
        return text
    except Exception:
        return None

def _kg_query_fallback(entity: str, relation_type: str = "") -> str:
    """本地降级查询"""
    data = _KG_FALLBACK.get(entity)
    if not data:
        return f"未检索到与'{entity}'相关的知识图谱信息"
    if relation_type:
        type_map = {"symptoms": "有症状", "diagnosis": "鉴别诊断", "examination": "需检查", "treatment": "治疗方式"}
        cn_type = type_map.get(relation_type, relation_type)
        items = data.get(cn_type, [])
        if items:
            return f"{entity}的{cn_type}：{'、'.join(items)}"
        else:
            return f"未找到{entity}的{relation_type}关联信息"
    text = f"{entity}知识图谱信息：\n"
    if data.get("有症状"):
        text += f"【常见症状】{'、'.join(data['有症状'])}\n"
    if data.get("需检查"):
        text += f"【需要检查】{'、'.join(data['需检查'])}\n"
    if data.get("治疗方式"):
        text += f"【治疗方式】{'、'.join(data['治疗方式'])}\n"
    return text

@tool
def kg_query(entity: str, relation_type: str = "") -> str:
    """从眼科知识图谱中查询疾病、症状、检查项目之间的结构化关联关系。

    Args:
        entity: 标准医学术语，如"青光眼"、"糖尿病性视网膜病变"、"白内障"、"高血压视网膜病变"、"病理性近视"、"急性结膜炎"、"麦粒肿"、"上睑下垂"
        relation_type: 可选，关系类型。可选值：symptoms（症状）、examination（检查）、treatment（治疗）

    Returns:
        字符串类型的结构化关联信息
    """
    # 疾病名称别名映射（用户可能用简称/不同写法）
    _ENTITY_ALIASES = {
        "糖尿病视网膜病变": "糖尿病性视网膜病变",
        "糖网": "糖尿病性视网膜病变",
        "高血压视网膜病变": "高血压视网膜病变",
        "近视": "病理性近视",
        "结膜炎": "急性结膜炎",
        "麦粒肿": "麦粒肿",
        "针眼": "麦粒肿",
    }

    # 标准化实体名
    canonical = _ENTITY_ALIASES.get(entity, entity)

    # 优先查询Neo4j
    result = _kg_query_neo4j(canonical, relation_type)
    if result is not None:
        return result
    # Neo4j不可用时降级为本地数据
    return _kg_query_fallback(canonical, relation_type)


@tool
def eye_image_analysis(image_path: str, image_type: str) -> str:
    """对患者上传的眼部图像进行AI辅助分析。根据图像类型自动调用对应的识别模型。

    Args:
        image_path: 患者上传的眼部图像文件路径
        image_type: 图像类型，必须为以下之一：
            - "external": 外眼图像（眼表、结膜、角膜、眼睑等）
            - "internal": 内眼图像（眼底照片、OCT图像等）

    Returns:
        字符串类型的图像分析结果，包含识别到的异常特征及初步判断
    """
    if image_type == "external":
        from predict_module import predict_text
        try:
            return predict_text(image_path)
        except Exception as e:
            return f"外眼图像分析失败：{str(e)}"
    elif image_type == "internal":
        predictor = _get_odir_predictor()
        result = predictor.predict(image_path)
        if result["status"] == "success":
            pred = result["prediction"]
            return f"内眼图像分析结果：{pred['disease_name']}，置信度{pred['confidence']*100:.1f}%"
        else:
            return f"内眼分析失败：{result['message']}"
    else:
        return "无法识别图像类型，请指定image_type为'external'（外眼）或'internal'（内眼）"


@tool
def fill_context_for_report():
    """无入参，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息"""
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    return f"报告模式已激活。当前就诊时间：{now}，请在报告的「基本信息」中填写该就诊时间。"

# 新增缺失的 rag_summarize 工具
@tool
def rag_summarize(text: str) -> str:
    """调用RAG服务对医学长文本进行摘要，提取病历、检查报告核心信息。

    Args:
        text: 待总结的问诊记录、检查报告、病历长文本

    Returns:
        精简关键医学摘要
    """
    return rag.rag_summarize(text)