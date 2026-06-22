import streamlit as st
import os
import uuid
import json
import re
from datetime import datetime
from PIL import Image
from agent.react_agent import ReactAgent
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO

st.set_page_config(page_title="明眸 · 眼科智能医生", page_icon="👁️", layout="wide")

# ==================== 历史记录持久化 ====================
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history")
os.makedirs(HISTORY_DIR, exist_ok=True)

def load_all_chats():
    """从磁盘加载所有历史对话"""
    chats = []
    if not os.path.exists(HISTORY_DIR):
        return chats
    for filename in os.listdir(HISTORY_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(HISTORY_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # JSON反序列化后trace_map的key变成字符串，需转回整数
                    if "trace_map" in data:
                        data["trace_map"] = {int(k): v for k, v in data["trace_map"].items()}
                    chats.append(data)
            except Exception:
                pass
    # 按时间倒序排列
    chats.sort(key=lambda x: x.get("time", ""), reverse=True)
    return chats

def save_chat_to_disk(chat_data):
    """保存单个对话到磁盘"""
    filepath = os.path.join(HISTORY_DIR, f"{chat_data['id']}.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"保存对话失败: {e}")

def delete_chat_from_disk(chat_id):
    """从磁盘删除对话"""
    filepath = os.path.join(HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)

# ==================== 全局样式 ====================
st.markdown("""
    <style>
        /* ---------- 侧边栏：医院淡蓝绿 ---------- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e6f2ef 0%, #d4e8e3 100%) !important;
            border-right: 1px solid #b8d5cc !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,0.6);
            color: #2c6e5a !important;
            border: 1px solid #b8d5cc;
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 13px;
            transition: all 0.2s;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(255,255,255,0.9);
            border-color: #5ba88e;
        }
        [data-testid="stSidebar"] hr {
            border-color: #b8d5cc;
        }

        /* ---------- 主区域背景 ---------- */
        .stApp {
            background: #f2f7f5;
        }

        /* ---------- 历史记录标题 ---------- */
        .history-title {
            font-size: 12px;
            font-weight: 600;
            color: #6a9e8e !important;
            letter-spacing: 1px;
            margin: 12px 0 8px 0;
            padding-left: 4px;
        }

        /* ---------- 历史记录项 ---------- */
        .history-item {
            padding: 10px 12px;
            margin-bottom: 4px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }
        .history-item:hover {
            background: rgba(255,255,255,0.6);
        }
        .history-item.active {
            background: rgba(255,255,255,0.85);
            border-color: #5ba3d9;
        }
        .history-item-title {
            font-size: 13px;
            color: #1a5276;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .history-item-time {
            font-size: 11px;
            color: #8bb8d8;
            margin-top: 2px;
        }

        /* ---------- 聊天气泡 ---------- */
        .chat-message-user {
            background: linear-gradient(135deg, #3d9b7e 0%, #2c7d63 100%);
            color: #ffffff;
            border-radius: 20px 20px 4px 20px;
            padding: 14px 18px;
            margin: 10px 0 10px auto;
            max-width: 65%;
            width: fit-content;
            box-shadow: 0 3px 12px rgba(44, 125, 99, 0.2);
            font-size: 14px;
            line-height: 1.7;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        .chat-message-assistant {
            background: #ffffff;
            color: #1f2937;
            border-radius: 20px 20px 20px 4px;
            padding: 14px 18px;
            margin: 10px 0;
            max-width: 65%;
            width: fit-content;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
            border: 1px solid #d4e8e3;
            font-size: 14px;
            line-height: 1.7;
            word-wrap: break-word;
            white-space: pre-wrap;
        }

        /* ---------- 欢迎区 ---------- */
        .welcome-area {
            text-align: center;
            padding: 80px 20px 40px 20px;
        }
        .welcome-area h1 {
            font-size: 28px;
            font-weight: 700;
            color: #2c6e5a;
            margin: 0 0 8px 0;
        }
        .welcome-area p {
            font-size: 15px;
            color: #6a9e8e;
            margin: 0 0 32px 0;
        }
        .welcome-cards {
            display: flex;
            gap: 16px;
            justify-content: center;
            flex-wrap: wrap;
            max-width: 700px;
            margin: 0 auto;
        }
        .welcome-card {
            background: #ffffff;
            border-radius: 14px;
            padding: 20px 24px;
            width: 200px;
            text-align: left;
            box-shadow: 0 2px 12px rgba(0,0,0,0.04);
            border: 1px solid #d4e8e3;
            cursor: pointer;
            transition: all 0.25s;
        }
        .welcome-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(44,125,99,0.12);
            border-color: #5ba88e;
        }
        .welcome-card .card-icon {
            font-size: 24px;
            margin-bottom: 10px;
        }
        .welcome-card .card-title {
            font-size: 14px;
            font-weight: 600;
            color: #2c6e5a;
            margin-bottom: 4px;
        }
        .welcome-card .card-desc {
            font-size: 12px;
            color: #6a9e8e;
        }

        /* ---------- 底部输入栏（固定在底部） ---------- */
        .bottom-input {
            position: fixed;
            bottom: 0;
            left: 280px;
            right: 0;
            padding: 12px 32px 20px 32px;
            background: linear-gradient(180deg, rgba(242,247,245,0) 0%, #f2f7f5 25%);
            z-index: 100;
        }
        .input-wrapper {
            max-width: 860px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 10px;
            background: #ffffff;
            border-radius: 28px;
            padding: 6px 6px 6px 20px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.06);
            border: 1px solid #c8ddd5;
        }
        .input-wrapper:focus-within {
            border-color: #3d9b7e;
            box-shadow: 0 4px 24px rgba(61,155,126,0.15);
        }

        /* ---------- 滚动条 ---------- */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #b8d5cc; border-radius: 3px; }

        /* ---------- 对话区底部留白 ---------- */
        .chat-spacer { height: 100px; }

        /* ---------- 对话标题 ---------- */
        .chat-header-bar {
            padding: 16px 32px;
            background: #ffffff;
            border-bottom: 1px solid #d4e8e3;
            margin: -1.125rem -1.125rem 0 -1.125rem;
        }
        .chat-header-bar h3 {
            font-size: 16px;
            font-weight: 600;
            color: #2c6e5a;
            margin: 0;
        }

        /* ---------- 发送/新建按钮 ---------- */
        [data-testid="stBaseButton-primary"] {
            background: #5ba88e !important;
            border: none !important;
            color: white !important;
        }
        [data-testid="stBaseButton-primary"]:hover {
            background: #4a9278 !important;
        }

        /* ---------- AI流程追踪 ---------- */
        .trace-container {
            margin-bottom: 12px;
            border-left: 3px solid #5ba88e;
            padding-left: 12px;
        }
        .trace-title {
            font-size: 14px;
            font-weight: 600;
            color: #2c6e5a;
            margin-bottom: 8px;
            padding: 4px 8px;
            background: #e6f2ef;
            border-radius: 4px;
            display: inline-block;
        }
        .trace-step {
            margin-bottom: 8px;
            border-radius: 8px;
            overflow: hidden;
            font-size: 14px;
        }
        .trace-think {
            background: #fff8e6;
            border: 1px solid #f0d98d;
        }
        .trace-call {
            background: #e6f2ef;
            border: 1px solid #b8d5cc;
        }
        .trace-result {
            background: #f0f7f4;
            border: 1px solid #d4e8e3;
        }
        .trace-header {
            padding: 8px 12px;
            font-weight: 600;
            color: #2c6e5a;
            font-size: 14px;
            border-bottom: 1px solid #d4e8e3;
        }
        .trace-think .trace-header {
            background: #fef3cd;
            color: #856404;
            border-bottom-color: #f0d98d;
        }
        .trace-call .trace-header {
            background: #d4e8e3;
        }
        .trace-result .trace-header {
            background: #e6f2ef;
        }
        .trace-body {
            padding: 10px 12px;
            color: #374151;
            font-size: 14px;
            line-height: 1.6;
        }
        .trace-body pre {
            margin: 6px 0 0 0;
            padding: 8px 10px;
            background: #ffffff;
            border-radius: 4px;
            font-size: 13px;
            color: #4b5563;
            white-space: pre-wrap;
            word-wrap: break-word;
            border: 1px solid #e5e7eb;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== Session State ====================
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

if "show_upload_modal" not in st.session_state:
    st.session_state["show_upload_modal"] = False

if "chat_history_list" not in st.session_state:
    st.session_state["chat_history_list"] = load_all_chats()

if "current_chat_id" not in st.session_state:
    st.session_state["current_chat_id"] = None

# 追踪步骤：{msg_index: [step1, step2, ...]}
if "trace_map" not in st.session_state:
    st.session_state["trace_map"] = {}

# 追踪本次问诊中所有上传的图片 [{path, type, analysis_result}]
if "uploaded_images" not in st.session_state:
    st.session_state["uploaded_images"] = []

# 报告生成触发标志（与button的key分开）
if "_show_report" not in st.session_state:
    st.session_state["_show_report"] = False

os.makedirs("temp_images", exist_ok=True)

# ==================== 工具函数 ====================
def escape_html(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

def new_chat():
    """新建对话"""
    # 先保存当前对话
    save_current_chat()
    chat_id = datetime.now().strftime("%m%d_%H%M%S")
    title = f"问诊 {chat_id}"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    chat_data = {
        "id": chat_id,
        "title": title,
        "time": now,
        "messages": [],
        "trace_map": {}
    }
    st.session_state["chat_history_list"].insert(0, chat_data)
    save_chat_to_disk(chat_data)
    st.session_state["current_chat_id"] = chat_id
    st.session_state["messages"] = []
    st.session_state["trace_map"] = {}
    st.session_state["uploaded_images"] = []

def switch_chat(chat_id):
    """切换到某个历史对话"""
    # 先保存当前对话
    save_current_chat()
    for item in st.session_state["chat_history_list"]:
        if item["id"] == chat_id:
            st.session_state["current_chat_id"] = chat_id
            st.session_state["messages"] = item.get("messages", []).copy()
            # 确保trace_map的key为整数
            raw_map = item.get("trace_map", {})
            st.session_state["trace_map"] = {int(k): v for k, v in raw_map.items()}
            # 从聊天历史中恢复上传图片（存路径，不存二进制）
            msgs = st.session_state["messages"]
            st.session_state["uploaded_images"] = [
                {"path": m["image_path"], "type": m.get("image_type", "")}
                for m in msgs if m.get("image_path")
            ]
            break

def save_current_chat():
    """保存当前对话到历史列表和磁盘"""
    if st.session_state["current_chat_id"] and st.session_state["messages"]:
        for item in st.session_state["chat_history_list"]:
            if item["id"] == st.session_state["current_chat_id"]:
                item["messages"] = st.session_state["messages"].copy()
                item["trace_map"] = st.session_state.get("trace_map", {}).copy()
                item["uploaded_images"] = st.session_state.get("uploaded_images", []).copy()
                # 用第一条用户消息更新标题
                for msg in item["messages"]:
                    if msg["role"] == "user":
                        content = msg["content"][:20]
                        item["title"] = content + ("..." if len(msg["content"]) > 20 else "")
                        break
                save_chat_to_disk(item)
                break

def send_message(content):
    st.session_state["messages"].append({"role": "user", "content": content})
    st.session_state["messages"].append({"role": "assistant", "content": "正在思考中..."})
    save_current_chat()

def on_send():
    """text_input的on_change回调，处理消息发送"""
    user_input = st.session_state.get("chat_input", "").strip()
    if user_input and not st.session_state.get("_pending_msg"):
        st.session_state["_pending_msg"] = user_input
        st.rerun()

# ==================== 侧边栏 ====================
with st.sidebar:
    if st.button("+ 新建问诊", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    st.markdown('<div class="history-title">过往对话</div>', unsafe_allow_html=True)

    if not st.session_state["chat_history_list"]:
        st.markdown("""
            <div style="text-align:center; padding:40px 0; color:#6a9e8e; font-size:13px;">
                暂无历史记录<br>点击上方按钮开始问诊
            </div>
        """, unsafe_allow_html=True)
    else:
        for item in st.session_state["chat_history_list"]:
            col_chat, col_del = st.columns([5, 1])
            with col_chat:
                if st.button(
                    f"{item['title']}",
                    key=f"history_{item['id']}",
                    use_container_width=True
                ):
                    switch_chat(item["id"])
                    st.rerun()
            with col_del:
                if st.button("x", key=f"del_{item['id']}"):
                    delete_chat_from_disk(item["id"])
                    st.session_state["chat_history_list"] = [
                        c for c in st.session_state["chat_history_list"] if c["id"] != item["id"]
                    ]
                    if st.session_state["current_chat_id"] == item["id"]:
                        st.session_state["current_chat_id"] = None
                        st.session_state["messages"] = []
                        st.session_state["trace_map"] = {}
                    st.rerun()

    st.divider()



# ==================== 主区域 ====================
st.markdown("""
    <div class="chat-header-bar" style="display:flex; align-items:center; justify-content:space-between;">
        <h3>当前问诊</h3>
        <span style="font-size:11px; color:#6a9e8e;">仅供初步筛查参考，不构成最终诊断</span>
    </div>
""", unsafe_allow_html=True)

if not st.session_state["messages"]:
    st.markdown("""
        <div class="welcome-area">
            <h1>您好，我是明眸眼科助手</h1>
            <p>请描述您的眼部症状，或选择左侧常见问题开始问诊</p>
            <div class="welcome-cards">
                <div class="welcome-card" onclick="document.querySelector('[data-testid=\\'stTextInput\\'] input').focus()">
                    <div class="card-title">症状自查</div>
                    <div class="card-desc">描述眼部不适获取初步建议</div>
                </div>
                <div class="welcome-card" onclick="document.querySelector('[data-testid=\\'stFileUploader\\'] input').click()">
                    <div class="card-title">图片分析</div>
                    <div class="card-desc">上传眼部图片AI辅助判读</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

else:
    for i, message in enumerate(st.session_state["messages"]):
        if message["role"] == "user":
            # 如果有图片，先展示图片
            if message.get("image_path"):
                st.image(message["image_path"], caption=message.get("image_type", ""), width=300)
            st.markdown(
                f'<div class="chat-message-user">{escape_html(message["content"])}</div>',
                unsafe_allow_html=True
            )
        else:
            content = message["content"]
            if content == "正在思考中...":
                continue
            # 渲染助手消息区域：思考链 + 回复内容
            # 先渲染可折叠的思考链（紧贴在回复上方）
            if i in st.session_state.get("trace_map", {}):
                steps = st.session_state["trace_map"][i]
                with st.expander("AI 思考过程", expanded=False):
                    for step in steps:
                        if step["type"] == "think":
                            st.markdown(
                                f'<div class="trace-step trace-think">'
                                f'<div class="trace-header">Step {step.get("step", "")} - 思考</div>'
                                f'<div class="trace-body">{escape_html(step.get("thought", ""))}</div>'
                                f'</div>', unsafe_allow_html=True
                            )
                        elif step["type"] == "call":
                            args_str = json.dumps(step["args"], ensure_ascii=False, indent=2)
                            st.markdown(
                                f'<div class="trace-step trace-call">'
                                f'<div class="trace-header">调用工具: {escape_html(step["name"])}</div>'
                                f'<div class="trace-body">输入参数:<br><pre>{escape_html(args_str)}</pre></div>'
                                f'</div>', unsafe_allow_html=True
                            )
                        elif step["type"] == "result":
                            output_str = str(step["output"])
                            if len(output_str) > 500:
                                output_str = output_str[:500] + "..."
                            st.markdown(
                                f'<div class="trace-step trace-result">'
                                f'<div class="trace-header">工具返回: {escape_html(step["name"])}</div>'
                                f'<div class="trace-body"><pre>{escape_html(output_str)}</pre></div>'
                                f'</div>', unsafe_allow_html=True
                            )
            # 再渲染消息气泡（保留换行格式）
            formatted_content = escape_html(content).replace("\n", "<br>")
            st.markdown(
                f'<div class="chat-message-assistant">{formatted_content}</div>',
                unsafe_allow_html=True
            )
    st.markdown('<div class="chat-spacer"></div>', unsafe_allow_html=True)

# ==================== 处理"正在思考中"的消息 ====================
if st.session_state["messages"] and st.session_state["messages"][-1]["content"] == "正在思考中..." and st.session_state["messages"][-1]["role"] == "assistant":
    last_user_msg = ""
    for msg in reversed(st.session_state["messages"]):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    trace_steps = []
    final_response = ""
    raw_chunks = []  # 调试用
    try:
        res_stream = st.session_state["agent"].execute_stream(
            last_user_msg, st.session_state["messages"][:-1]
        )
        for chunk in res_stream:
            raw_chunks.append(chunk)
            # 循环解析 chunk 中所有的标签（可能有多个）
            remaining = chunk
            while remaining:
                found = False
                # 解析思考步骤
                if "[TRACE_THINK]" in remaining:
                    try:
                        start = remaining.index("[TRACE_THINK]") + len("[TRACE_THINK]")
                        end = remaining.index("[/TRACE_THINK]")
                        data = json.loads(remaining[start:end])
                        trace_steps.append({"type": "think", "step": data.get("step", 0), "thought": data.get("thought", "")})
                        remaining = remaining[end + len("[/TRACE_THINK]"):]
                        found = True
                    except Exception as e:
                        trace_steps.append({"type": "think", "step": 0, "thought": f"解析错误: {e}"})
                        break
                # 解析工具调用
                elif "[TRACE_TOOL_CALL]" in remaining:
                    try:
                        start = remaining.index("[TRACE_TOOL_CALL]") + len("[TRACE_TOOL_CALL]")
                        end = remaining.index("[/TRACE_TOOL_CALL]")
                        data = json.loads(remaining[start:end])
                        trace_steps.append({"type": "call", "name": data["name"], "args": data["args"]})
                        remaining = remaining[end + len("[/TRACE_TOOL_CALL]"):]
                        found = True
                    except Exception as e:
                        trace_steps.append({"type": "call", "name": "parse_error", "args": str(e)})
                        break
                # 解析工具结果
                elif "[TRACE_TOOL_RESULT]" in remaining:
                    try:
                        start = remaining.index("[TRACE_TOOL_RESULT]") + len("[TRACE_TOOL_RESULT]")
                        end = remaining.index("[/TRACE_TOOL_RESULT]")
                        data = json.loads(remaining[start:end])
                        trace_steps.append({"type": "result", "name": data["name"], "output": data["output"]})
                        remaining = remaining[end + len("[/TRACE_TOOL_RESULT]"):]
                        found = True
                    except Exception as e:
                        trace_steps.append({"type": "result", "name": "parse_error", "output": str(e)})
                        break
                # 解析最终回答
                elif "[TRACE_ANSWER]" in remaining:
                    try:
                        start = remaining.index("[TRACE_ANSWER]") + len("[TRACE_ANSWER]")
                        end = remaining.index("[/TRACE_ANSWER]")
                        final_response = remaining[start:end]
                        remaining = remaining[end + len("[/TRACE_ANSWER]"):]
                        found = True
                    except Exception:
                        break
                # 没有找到任何标签，退出循环
                if not found:
                    break
    except Exception as e:
        trace_steps.append({"type": "think", "step": 0, "thought": f"AI处理异常: {str(e)}"})
        final_response = f"抱歉，AI处理过程中出现异常，请稍后重试。（{str(e)}）"

    # 如果没有工具调用，添加一条"直接回答"的思考步骤
    if not trace_steps:
        trace_steps.append({"type": "think", "step": 1, "thought": "根据已有信息直接回答，无需调用工具"})

    # 追踪步骤存入trace_map，消息内容只存纯文本
    msg_index = len(st.session_state["messages"]) - 1
    st.session_state["trace_map"][msg_index] = trace_steps

    # 从 trace_steps 中提取 eye_image_analysis 的结果，更新 uploaded_images[].analysis
    pending_image_path = None
    for step in trace_steps:
        if step["type"] == "call" and step["name"] == "eye_image_analysis":
            pending_image_path = step["args"].get("image_path")
        elif step["type"] == "result" and step["name"] == "eye_image_analysis" and pending_image_path:
            # 匹配 uploaded_images 中的图片
            for img in st.session_state.get("uploaded_images", []):
                if img["path"] == pending_image_path:
                    img["analysis"] = step["output"]
                    break
            pending_image_path = None
        # 检测 fill_context_for_report 工具调用，触发报告生成
        elif step["type"] == "call" and step["name"] == "fill_context_for_report":
            st.session_state["_show_report"] = True

    # 保存调试信息
    st.session_state["_debug_raw_chunks"] = raw_chunks
    st.session_state["_debug_trace_steps"] = trace_steps
    st.session_state["_debug_final_response"] = final_response
    # 清理回答中的多余空行
    cleaned_response = re.sub(r'\n{3,}', '\n\n', final_response.strip()) if final_response.strip() else "（未获取到回答）"
    # 如果回答为空但raw_chunks有内容，尝试从raw_chunks中提取
    if cleaned_response == "（未获取到回答）" and raw_chunks:
        for c in reversed(raw_chunks):
            if "[TRACE_ANSWER]" in c:
                try:
                    s = c.index("[TRACE_ANSWER]") + len("[TRACE_ANSWER]")
                    e = c.index("[/TRACE_ANSWER]")
                    cleaned_response = re.sub(r'\n{3,}', '\n\n', c[s:e].strip())
                except:
                    pass
                break
    st.session_state["messages"][-1] = {"role": "assistant", "content": cleaned_response}
    save_current_chat()
    st.rerun()

# ==================== 底部输入栏 ====================
st.markdown('<div class="bottom-input"><div class="input-wrapper">', unsafe_allow_html=True)

if st.session_state.get("_pending_msg"):
    st.session_state["chat_input"] = ""

col_left, col_upload, col_mid, col_right = st.columns([1, 1, 14, 1])
with col_left:
    if st.button("➕", key="upload_btn"):
        st.session_state["show_upload_modal"] = True
        st.rerun()
with col_upload:
    if st.button("📎", key="upload_btn2", help="上传图片"):
        st.session_state["show_upload_modal"] = True
        st.rerun()
with col_mid:
    prompt = st.text_input(
        "chat_input",
        placeholder="请描述您的眼部症状...",
        label_visibility="collapsed",
        key="chat_input",
        on_change=on_send
    )
with col_right:
    send_btn = st.button("发送", key="send_btn", type="primary")

st.markdown('</div></div>', unsafe_allow_html=True)

if send_btn:
    user_input = st.session_state.get("chat_input", "").strip()
    if user_input and not st.session_state.get("_pending_msg"):
        st.session_state["_pending_msg"] = user_input
        st.rerun()

if st.session_state.get("_pending_msg"):
    pending_msg = st.session_state.pop("_pending_msg")
    if not st.session_state["current_chat_id"]:
        new_chat()
    send_message(pending_msg)
    st.rerun()

# ==================== 对话框定义（提前定义，确保可被调用） ====================

@st.dialog("问诊报告", width="large")
def report_dialog():
    user_msgs = [m for m in st.session_state["messages"] if m["role"] == "user"]
    assistant_msgs = [m for m in st.session_state["messages"] if m["role"] == "assistant" and m["content"] != "正在思考中..."]

    col_left, col_right = st.columns([1, 1])
    with col_left:
        hospital = st.text_input("医院名称", value="明眸眼科医院")
    with col_right:
        department = st.text_input("科室", value="眼科")
    doctor = st.text_input("医生姓名", value="AI智能助手")
    patient_name = st.text_input("患者姓名（选填）", value="")
    visit_date = st.text_input("就诊日期", value=datetime.now().strftime("%Y-%m-%d"))

    st.divider()

    conversation_text = ""
    for m in user_msgs + assistant_msgs:
        role = "患者" if m["role"] == "user" else "医生"
        conversation_text += f"【{role}】{m['content']}\n\n"

    with st.spinner("正在生成报告..."):
        try:
            from utils.prompt_loader import load_report_prompts
            from model.factory import chat_model
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            prompt_text = load_report_prompts()
            prompt = ChatPromptTemplate.from_template(prompt_text)
            chain = prompt | chat_model | StrOutputParser()
            from agent.tools.agent_tools import fill_context_for_report
            time_info = fill_context_for_report.invoke({"conversation_context": ""})
            report_content = chain.invoke({
                "conversation_context": conversation_text[:3000],
                "current_time": time_info.replace("报告模式已激活。", "").strip()
            })
        except Exception as e:
            report_content = f"报告生成失败：{str(e)}\n\n请确保对话内容足够完整后重试。"

    st.markdown("### 📋 报告预览")
    st.markdown(report_content)
    st.divider()

    if st.session_state.get("uploaded_images"):
        st.markdown("### 🖼️ 图片资料")
        img_cols = st.columns(len(st.session_state["uploaded_images"]))
        for idx, img_info in enumerate(st.session_state["uploaded_images"]):
            with img_cols[idx]:
                st.image(img_info["path"], caption=img_info.get("type", ""), width=200)

    st.divider()

    def build_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                leftMargin=20*mm, rightMargin=20*mm,
                                topMargin=15*mm, bottomMargin=15*mm)

        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os

        chinese_font_path = None
        font_candidates = [
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "/usr/share/fonts/noto/NotoSansCJK-SC.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-SC.otf",
        ]
        for candidate in font_candidates:
            if os.path.exists(candidate):
                chinese_font_path = candidate
                break

        if chinese_font_path:
            pdfmetrics.registerFont(TTFont("SimSun", chinese_font_path))
            pdfmetrics.registerFont(TTFont("SimSun-Bold", chinese_font_path))
            pdfmetrics.registerFontFamily("SimSun", normal="SimSun", bold="SimSun-Bold")
            font_name = "SimSun"
        else:
            font_name = "Helvetica"

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("title", parent=styles["Normal"],
                                    fontSize=18, alignment=TA_CENTER,
                                    textColor=colors.HexColor("#2c6e5a"),
                                    spaceAfter=6, fontName=font_name)
        header_style = ParagraphStyle("header", parent=styles["Normal"],
                                     fontSize=10, alignment=TA_CENTER,
                                     textColor=colors.grey, spaceAfter=12, fontName=font_name)
        section_style = ParagraphStyle("section", parent=styles["Normal"],
                                       fontSize=13, textColor=colors.HexColor("#2c6e5a"),
                                       fontName=font_name, spaceBefore=10, spaceAfter=4)
        body_style = ParagraphStyle("body", parent=styles["Normal"],
                                    fontSize=10, leading=16, spaceAfter=4, fontName=font_name)

        story = []
        story.append(Paragraph("眼 科 门 诊 病 历", title_style))
        story.append(Paragraph(f"{hospital} · {department}", header_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#b8d5cc"), spaceAfter=10))

        info_data = [
            ["患者姓名", patient_name or "—", "就诊日期", visit_date],
            ["医院", hospital, "科室", department],
            ["医生", doctor, "AI辅助", "明眸眼科助手"],
        ]
        info_table = Table(info_data, colWidths=[22*mm, 58*mm, 22*mm, 58*mm])
        info_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7faf9")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b8d5cc")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 8*mm))

        if st.session_state.get("uploaded_images"):
            story.append(Paragraph("图片资料", section_style))
            row = []
            for img_info in st.session_state["uploaded_images"]:
                if os.path.exists(img_info["path"]):
                    rl_img = RLImage(img_info["path"], width=40*mm, height=30*mm)
                    caption = Paragraph(img_info.get("type", ""), body_style)
                    row.append([rl_img, Spacer(1, 2*mm), caption])
            if row:
                img_table = Table([row], colWidths=[80*mm]*len(row))
                img_table.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.append(img_table)
            story.append(Spacer(1, 6*mm))

        story.append(Paragraph("问诊记录与建议", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#b8d5cc"), spaceAfter=6))
        for line in report_content.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 4*mm))
            else:
                story.append(Paragraph(line, body_style))

        story.append(Spacer(1, 10*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#b8d5cc")))
        story.append(Paragraph(
            "本报告由AI辅助生成，仅供初步筛查参考，不构成最终诊断，如有疑问请咨询专业眼科医生。",
            ParagraphStyle("disclaimer", parent=styles["Normal"], fontSize=8,
                          textColor=colors.grey, alignment=TA_CENTER, spaceBefore=4, fontName=font_name)
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer

    col_pdf, col_close = st.columns([1, 1])
    with col_pdf:
        st.download_button(
            "⬇️ 下载PDF报告",
            data=build_pdf(),
            file_name=f"眼科问诊报告_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
    with col_close:
        if st.button("关闭", use_container_width=True):
            st.session_state["_show_report"] = False
            st.rerun()

@st.dialog("上传眼部图片")
def upload_dialog():
    uploaded_file = st.file_uploader(
        "选择图片",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="modal_upload"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="预览", use_column_width=True)

    image_type = st.radio(
        "图片类型",
        ["external", "internal"],
        format_func=lambda x: "外眼图像" if x == "external" else "内眼图像",
        horizontal=True,
        key="modal_image_type"
    )

    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button("取消", key="cancel_upload", use_container_width=True):
            st.session_state["show_upload_modal"] = False
            st.rerun()
    with col_confirm:
        if st.button("发送分析", key="confirm_upload", use_container_width=True, type="primary"):
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                file_ext = uploaded_file.name.split(".")[-1]
                filename = f"{uuid.uuid4().hex}.{file_ext}"
                file_path = os.path.join("temp_images", filename)
                image.save(file_path)

                # 内部使用英文标识，UI显示中文
                image_type_eng = image_type  # "external" or "internal"
                image_type_desc = "外眼图像" if image_type_eng == "external" else "内眼图像"

                # 用户消息内容：简洁，只告诉 agent 路径和类型，让 agent 自己调用工具
                message_content = (
                    f"图片路径: {file_path}\n"
                    f"图片类型: {image_type_eng}\n"
                    f"请分析这张眼部图片"
                )

                if not st.session_state["current_chat_id"]:
                    new_chat()

                st.session_state["messages"].append({
                    "role": "user",
                    "content": message_content,
                    "image_path": file_path,
                    "image_type": image_type_desc,
                    "image_type_eng": image_type_eng
                })
                st.session_state["uploaded_images"].append({
                    "path": file_path,
                    "type": image_type_desc,
                    "type_eng": image_type_eng,
                    "analysis": None  # 等 agent 分析完再填充
                })
                st.session_state["messages"].append({"role": "assistant", "content": "正在思考中..."})

                st.session_state["show_upload_modal"] = False
                st.rerun()

# ==================== 对话框调用（互斥逻辑） ====================
if st.session_state.get("_show_report"):
    report_dialog()
elif st.session_state.get("show_upload_modal"):
    upload_dialog()
