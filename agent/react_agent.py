import json
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts, load_report_prompts
from agent.tools.agent_tools import rag_summarize, fill_context_for_report, kg_query, eye_image_analysis
from utils.logger_handler import logger


# 定义 Agent 状态
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    is_report: bool


# 所有工具列表
tools = [rag_summarize, kg_query, eye_image_analysis, fill_context_for_report]

# 工具名称到工具对象的映射
tool_map = {t.name: t for t in tools}

# 绑定工具到模型
model_with_tools = chat_model.bind_tools(tools)


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """判断是否需要继续调用工具"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"


def agent_node(state: AgentState) -> dict:
    """Agent 主节点：调用模型，动态切换提示词"""
    logger.info(f"[agent_node]即将调用模型，带有{len(state['messages'])}条消息。")

    messages = state["messages"]

    # 根据上下文动态切换提示词
    if state.get("is_report", False):
        system_prompt = load_report_prompts()
    else:
        system_prompt = load_system_prompts()

    # 构建带系统提示词的消息列表
    system_message = SystemMessage(content=system_prompt)
    all_messages = [system_message] + messages

    response = model_with_tools.invoke(all_messages)

    return {"messages": [response]}


def tool_node_with_monitor(state: AgentState) -> dict:
    """带监控的工具执行节点（手动实现，不依赖 ToolNode）"""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    tool_messages = []
    is_report = False

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        logger.info(f"[tool monitor]执行工具：{tool_name}")
        logger.info(f"[tool monitor]传入参数：{tool_args}")

        try:
            tool_obj = tool_map.get(tool_name)
            if tool_obj:
                result = tool_obj.invoke(tool_args)
            else:
                result = f"错误：未找到工具 {tool_name}"
        except Exception as e:
            result = f"工具执行错误：{str(e)}"
            logger.error(f"[tool monitor]工具{tool_name}执行失败：{str(e)}")

        # 确保 result 是字符串
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)

        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tool_id, name=tool_name)
        )

        # 检查是否调用了 fill_context_for_report
        if tool_name == "fill_context_for_report":
            is_report = True
            logger.info("[tool monitor]检测到 fill_context_for_report 调用，切换为报告生成模式")

        logger.info(f"[tool monitor]工具{tool_name}调用完成")

    output = {"messages": tool_messages}
    if is_report:
        output["is_report"] = True

    return output


def create_agent():
    """创建 ReAct Agent 图"""
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node_with_monitor)

    # 设置入口
    graph.add_edge(START, "agent")

    # 添加条件边：agent -> tools 或 END
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})

    # 工具执行后回到 agent
    graph.add_edge("tools", "agent")

    return graph.compile()


class ReactAgent:
    def __init__(self):
        self.agent = create_agent()

    def execute_stream(self, query: str, history: list[dict] = None):
        # 构建历史消息列表（history已包含当前用户消息，不再重复添加）
        messages = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})

        # 如果history为空，则添加当前提问
        if not messages:
            messages.append({"role": "user", "content": query})

        input_dict = {
            "messages": messages,
            "is_report": False,
        }

        step_num = 0
        # 记录已处理的消息数量，只处理新增的消息
        prev_msg_count = len(messages)

        for chunk in self.agent.stream(input_dict, stream_mode="values", config={"recursion_limit": 10}):
            all_messages = chunk["messages"]
            # 只看新增的消息（跳过输入的历史消息）
            if len(all_messages) <= prev_msg_count:
                continue

            # 处理所有新增的消息（可能有多条工具调用和工具结果）
            for msg in all_messages[prev_msg_count:]:
                # 追踪工具调用：当AI决定调用工具时
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    step_num += 1
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        yield f"[TRACE_THINK]{json.dumps({'step': step_num, 'thought': f'需要调用工具 {tool_name} 获取信息'}, ensure_ascii=False)}[/TRACE_THINK]\n"
                        yield f"[TRACE_TOOL_CALL]{json.dumps({'name': tool_name, 'args': tool_args}, ensure_ascii=False)}[/TRACE_TOOL_CALL]\n"

                # 追踪工具结果：当工具执行完成时
                elif isinstance(msg, ToolMessage):
                    tool_name = msg.name if msg.name else "unknown"
                    tool_output = msg.content
                    yield f"[TRACE_TOOL_RESULT]{json.dumps({'name': tool_name, 'output': tool_output}, ensure_ascii=False)}[/TRACE_TOOL_RESULT]\n"

                # 最终文本回答
                elif isinstance(msg, AIMessage) and msg.content:
                    yield f"[TRACE_ANSWER]{msg.content.strip()}[/TRACE_ANSWER]\n"

            # 更新已处理的消息数量
            prev_msg_count = len(all_messages)


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
