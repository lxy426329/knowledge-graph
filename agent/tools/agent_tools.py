from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService

rag = RagSummarizeService()


@tool
def rag_summarize(query: str) -> str:
    """从知识库检索回答"""
    return rag.rag_summarize(query)


@tool
def kg_query(entity: str, relation_type: str = "") -> str:
    """从眼科知识图谱中查询疾病、症状、检查项目之间的结构化关联关系。

    Args:
        entity: 标准医学术语，如"青光眼"、"糖尿病视网膜病变"
        relation_type: 可选，关系类型。可选值：symptoms、diagnosis、examination、treatment

    Returns:
        字符串类型的结构化关联信息
    """
    # TODO: 对接Neo4j，当前mock
    mock_data = {
        "青光眼": "关联症状：眼压升高、视野缺损、头痛、虹视；鉴别诊断：白内障、视神经炎；推荐检查：眼压测量、视野检查、OCT、房角镜检查；治疗方案：药物降眼压、激光治疗、手术治疗",
        "白内障": "关联症状：视力渐进性下降、畏光、视物模糊、色觉改变；鉴别诊断：青光眼、屈光不正；推荐检查：裂隙灯检查、视力检查、眼底检查；治疗方案：手术摘除联合人工晶体植入",
        "干眼症": "关联症状：干涩、异物感、烧灼感、视疲劳；鉴别诊断：结膜炎、睑缘炎；推荐检查：泪液分泌试验、泪膜破裂时间；治疗方案：人工泪液、湿房镜、生活习惯调整",
    }
    return mock_data.get(entity, f"未检索到与'{entity}'相关的知识图谱信息")


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
        # TODO: 对接外眼模型接口，当前mock
        return "外眼图像分析结果：未见明显结膜充血，角膜透明，瞳孔等大等圆，对光反射灵敏。未检测到明显外眼异常。（mock数据，待对接外眼识别模型）"
    elif image_type == "internal":
        # TODO: 对接内眼模型接口，当前mock
        return "内眼图像分析结果：视盘边界清晰，杯盘比约0.3，黄斑区反光可见，未见明显出血点或渗出。（mock数据，待对接内眼识别模型）"
    else:
        return "无法识别图像类型，请指定image_type为'external'（外眼）或'internal'（内眼）"


@tool
def fill_context_for_report():
    """无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息"""
    return "fill_context_for_report已调用"
