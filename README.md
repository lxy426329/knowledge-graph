<div align="center">

# 明眸 · 眼科智能医生

**基于 LangChain + ReAct 范式 + RAG 检索增强的眼科智能问诊系统**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
&nbsp;
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://www.langchain.com/)
&nbsp;
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)](https://github.com/langchain-ai/langgraph)
&nbsp;
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)](https://streamlit.io/)
&nbsp;
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

</div>

---

## 项目简介

明眸眼科智能医生是一款基于 **LangChain ReAct Agent** 构建的专业眼科问诊系统。系统模拟真实眼科医生的问诊流程，能够：

- 理解患者的眼部症状描述
- 通过智能追问收集必要信息
- 检索专业眼科知识库获取准确信息
- 分析上传的眼部图像
- 生成个性化的筛查建议和检查报告

系统采用流式输出技术，实时展示 Agent 的思考过程、工具调用和推理逻辑，让用户能够清晰了解 AI 的决策依据。

## 系统流程

```
用户提问 → 输入分发（文本/图片）
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 内眼图像    外眼图像     RAG知识检索
  分析        分析      (Chroma向量库)
    │           │           │
    └───────────┼───────────┘
                ▼
          结果整合 + 知识图谱查询
                │
                ▼
          信息充分性判断
           ┌────┴────┐
           ▼         ▼
        信息足够   信息不足
           │         │
           ▼         ▼
       生成回答   智能追问
           │
     ┌─────┴─────┐
     ▼           ▼
  常规回答    医疗报告
```

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端交互层（Streamlit）                  │
│              用户输入 → 流式渲染 → 对话历史管理               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  Agent推理层（LangGraph ReAct）            │
│                                                          │
│   ┌─────────┐    条件判断    ┌─────────┐                │
│   │  agent  │ ─────────────→ │  tools  │                │
│   │ (LLM)  │ ←─── 返回结果 ── │ (执行)  │                │
│   └─────────┘                └─────────┘                │
│   动态切换提示词：普通问诊 / 报告生成                         │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   工具执行层（4个工具）                      │
│                                                          │
│  rag_summarize    kg_query    eye_image_analysis         │
│  (向量检索摘要)   (知识图谱查询)  (眼部图像分析)              │
│                                                          │
│  fill_context_for_report (报告模式切换触发器)               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                   知识服务层                               │
│                                                          │
│  Chroma向量库        Neo4j知识图谱      图像识别模型        │
│  (眼科文档检索)       (结构化关系查询)    (ODIR疾病分类)     │
└─────────────────────────────────────────────────────────┘
```

### 核心特性

| 特性 | 说明 |
|---|---|
| **ReAct 范式** | Thought → Action → Observation 循环，Agent 自主推理并决定调用哪个工具 |
| **RAG 检索增强** | Chroma 向量库 + DashScope Embedding，MD5 文件去重，支持 txt/pdf 混合加载 |
| **知识图谱查询** | 预留 Neo4j 接口，支持疾病-症状-检查-治疗结构化关联查询 |
| **眼部图像分析** | 支持外眼/内眼图像上传，内眼对接 ODIR 疾病分类模型 |
| **智能追问** | 患者描述模糊时先追问症状维度，收集足够信息后再调用工具 |
| **动态提示词切换** | 根据运行时上下文自动切换「问诊筛查」与「报告生成」两套 System Prompt |
| **思考链可视化** | 可折叠展示 Agent 的思考过程、工具调用和返回结果 |
| **流式对话界面** | Streamlit 构建，医疗风格界面，支持流式输出、图片上传、历史对话管理 |
| **PDF 报告生成** | 支持生成包含问诊记录和图片的 PDF 格式医疗报告 |

## 技术栈

| 层级 | 技术 |
|---|---|
| LLM | 通义千问 qwen3.7-plus（DashScope / ChatOpenAI 兼容模式） |
| Embedding | DashScope text-embedding-v4 |
| Agent 框架 | LangChain + LangGraph |
| 向量数据库 | Chroma |
| 知识图谱 | Neo4j（预留接口，当前 Mock 数据） |
| 图像识别 | ODIRPredictor（内眼疾病分类） |
| 文档处理 | PyPDF + RecursiveCharacterTextSplitter |
| 前端 | Streamlit |
| PDF 生成 | ReportLab |
| 配置 | YAML 驱动（Agent / RAG / Chroma / Prompts） |

## 快速开始

### 环境要求

- **Python** >= 3.10
- **DashScope API Key**（[阿里云百炼](https://bailian.console.aliyun.com/) 申请）

### 1. 克隆仓库

```bash
git clone https://github.com/lhh737/LangChain-ReAct-Agent.git
cd LangChain-ReAct-Agent
```

### 2. 创建虚拟环境

```bash
conda create -n fuxian python=3.10
conda activate fuxian
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 API Key

参考 `.env.example`，在项目根目录创建 `.env` 文件：

```
DASHSCOPE_API_KEY=your-api-key
```

> 申请地址：[阿里云百炼控制台](https://bailian.console.aliyun.com/)

### 5. 初始化知识库（首次运行）

```bash
python -c "from rag.vector_store import VectorStoreService; VectorStoreService().load_document()"
```

等待日志输出"内容加载成功"即可。

### 6. 启动应用

```bash
streamlit run app.py
```

浏览器自动打开 http://localhost:8501

### 验证运行

启动后在聊天框输入以下测试问题：

- *我最近眼睛干涩，看电脑时间长了更明显，是怎么回事？*（RAG 知识库问答）
- *青光眼有哪些早期症状？*（知识图谱查询）
- *请帮我生成一份眼科检查报告*（报告生成 + 工具调用）

## 项目结构

```
LangChain-ReAct-Agent/
│
├── agent/                          # Agent 核心
│   ├── react_agent.py              #   ReAct Agent 主逻辑（StateGraph · 流式执行 · 思考链追踪）
│   └── tools/
│       └── agent_tools.py          #   工具函数（RAG检索 / 知识图谱 / 图像分析 / 报告切换）
│
├── rag/                            # RAG 检索增强
│   ├── vector_store.py             #   Chroma 向量库 · 文档加载 · MD5 去重
│   └── rag_service.py              #   RAG 检索 → LLM 总结服务
│
├── model/
│   └── factory.py                  # 模型工厂（ChatOpenAI + DashScopeEmbedding）
│
├── config/                         # YAML 配置文件
│   ├── agent.yml                   #   Agent 外部数据路径
│   ├── chroma.yml                  #   向量库与检索参数
│   ├── prompts.yml                 #   提示词模板路径
│   └── rag.yml                     #   模型名称配置
│
├── prompts/                        # 提示词模板
│   ├── main_prompt.txt             #   问诊筛查 System Prompt
│   ├── rag_summarize.txt           #   RAG 总结 Prompt
│   └── report_prompt.txt           #   报告生成 System Prompt
│
├── utils/                          # 工具函数
│   ├── config_handler.py           #   YAML 配置加载
│   ├── file_handler.py             #   文件解析（PDF/TXT）
│   ├── logger_handler.py           #   日志管理
│   ├── path_tool.py                #   路径工具
│   └── prompt_loader.py            #   提示词加载
│
├── data/                           # 知识库文档（眼科相关）
├── temp_images/                    # 临时图片存储
├── chat_history/                   # 对话历史存储
├── app.py                          # Streamlit 应用入口
├── requirements.txt
└── README.md
```

## 配置说明

项目通过 `config/` 目录下的 YAML 文件统一管理配置：

| 文件 | 说明 |
|---|---|
| `rag.yml` | 对话模型名称（qwen3.7-plus）、Embedding 模型名称（text-embedding-v4） |
| `chroma.yml` | Chroma 持久化路径、分块大小（200）、检索 Top-K（3）、支持的文件类型 |
| `prompts.yml` | 各场景提示词模板文件路径 |
| `agent.yml` | 外部数据路径等 |

首次运行只需确保 **DashScope API Key 已设置** 且 `data/` 目录下有知识库文档即可。

## 使用说明

### 基本操作

1. **发送消息**：在底部输入框输入问题，按 Enter 键或点击发送按钮
2. **上传图片**：点击输入框左侧的 ➕ 或 📎 按钮，选择图片并指定图片类型（外眼/内眼）
3. **新建问诊**：点击左侧边栏的"新建问诊"按钮
4. **查看历史**：点击左侧边栏的历史对话项查看过往问诊记录
5. **生成报告**：在对话过程中，系统会根据情况询问是否需要生成报告，确认后自动生成 PDF 报告

### 思考链查看

每条 AI 回复下方都有一个可折叠的"AI 思考过程"区域，展开后可以看到：

- **思考步骤**：Agent 的推理过程
- **工具调用**：调用的工具名称和输入参数
- **工具结果**：工具返回的结果

### 报告生成

系统会在问诊过程中判断是否需要生成报告，当收集到足够信息后，会询问用户是否需要生成报告。生成的报告包含：

- 患者基本信息
- 问诊记录与建议
- 上传的图片资料
- 免责声明

## License

MIT
