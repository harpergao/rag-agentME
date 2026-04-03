import operator
import json
import re
from typing import Annotated, Dict, List, Optional, TypedDict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.prebuilt import ToolNode # 引入标准工具节点

from dotenv import load_dotenv
load_dotenv()

# 导入你的本地组件
from model_client import LocalMultimodalModel 
from langchain_wrappers import LocalQwenChatModel 
from tools import text_retrieval, image_retrieval, cross_doc_compare, citation_formatter, save_session_summary, save_session_summary_sync
# 引入 LangGraph 和 LangChain 的消息处理机制
from langgraph.graph.message import add_messages
from langchain_core.messages import RemoveMessage, AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver # 用于持久化记忆
from langgraph.types import Command

import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from typing import Literal
# --- 1. 状态定义 ---

class SubAgentState(TypedDict):
    """子图(Agent Subgraph)的独立状态"""
    messages: Annotated[list[AnyMessage], add_messages]  # 存储对话和工具调用消息
    question: str                            # 当前子任务的问题
    question_index: int                      # 子问题的序号（用于汇总排序）
    context_summary: str                     # 压缩后的上下文
    iteration_count: int                     # 循环次数
    tool_call_count: int                     # 工具调用次数
    # 新增：防死循环的动作记忆
    retrieval_keys: Annotated[list[str], operator.add]
    agent_answers: list
class MainState(TypedDict):
    """主图(Main Graph)的全局状态"""
    messages: Annotated[list[AnyMessage], add_messages]
    originalQuery: str
    conversation_summary: str
    questionIsClear: bool
    rewrittenQuestions: List[str]
    # 收集子图返回的答案，格式如 [{"index": 0, "answer": "..."}]
    agent_answers: Annotated[List[Dict[str, Any]], operator.add]
    
# --- 2. 路由逻辑 (Edges) ---

def route_after_rewrite(state: MainState) -> Literal["request_clarification", "agent"]:
    """判断是否清晰，清晰则派发给多个子图，不清晰则请求澄清"""
    if not state.get("questionIsClear", False):
        return "request_clarification"
    else:
        # 核心：使用 Send API 将多个子问题并行分发给子图的 orchestrator
        return [
                Send("agent", {
                    "question": query, 
                    "question_index": idx, 
                    "messages": [], 
                    "iteration_count": 0,
                    "tool_call_count": 0,
                    "context_summary": "",
                    "retrieval_keys": [] # 补充此项，保证状态机初始化安全
                })
                for idx, query in enumerate(state.get("rewrittenQuestions", []))
                ]

def route_after_orchestrator(state: SubAgentState) -> Literal["tools", "fallback_response", "collect_answer"]:
    """Orchestrator 之后的路由：检查是否达到上限或是否需要调用工具"""
    iteration = state.get("iteration_count", 0)
    tool_count = state.get("tool_call_count", 0)
    
    # 你可以自己设定最大值
    MAX_ITERATIONS = 5
    MAX_TOOL_CALLS = 5

    if iteration >= MAX_ITERATIONS or tool_count >= MAX_TOOL_CALLS:
        return "fallback_response"

    last_message = state["messages"][-1]
    # 判断模型是否生成了工具调用请求 (需要你的 LocalQwenTextLLM 支持 tool_calls)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "collect_answer"

# def should_compress_context(state: SubAgentState) -> Literal["compress_context", "orchestrator"]:
#     """纯粹的路由函数：只负责判断 Token 长度并指路，绝不修改 State"""
#     messages = state.get("messages", [])
    
#     # 提取 summary，并安全转换为字符串
#     summary_obj = state.get("context_summary", "")
#     summary_str = summary_obj.content if hasattr(summary_obj, "content") else str(summary_obj)
    
#     # 安全拼接
#     current_text = str([m.content for m in messages if hasattr(m, "content")]) + summary_str
#     TOKEN_LIMIT = 25000 
    
#     if len(current_text) > TOKEN_LIMIT:
#         print("🔀 [Router] 上下文超限，路由至 compress_context 节点")
#         return "compress_context"
    
#     print("🔀 [Router] 上下文安全，路由回 orchestrator 节点")
#     # 🚨 核心修复：只返回字符串，绝对不要 return Command(...)
#     return "orchestrator"
def estimate_tokens(text: str) -> int:
    """粗略估算 Token 数量：中英文混合情况下，大约 2 个字符算 1 个 Token"""
    return len(str(text)) // 2

def should_compress_context(state: SubAgentState) -> Literal["compress_context", "orchestrator"]:
    """纯粹的路由函数：使用动态阈值判断是否超载"""
    messages = state.get("messages", [])
    summary_str = state.get("context_summary", "")
    
    # 1. 计算当前的 Token 负载
    current_token_messages = estimate_tokens("".join([str(m.content) for m in messages if hasattr(m, "content")]))
    current_token_summary = estimate_tokens(summary_str)
    total_tokens = current_token_messages + current_token_summary

    # 2. 动态阈值公式 (BASE + 摘要长度的膨胀系数)
    # 基础允许 8000 token (约 1.6 万字符)，摘要每增加 1 token，容忍度增加 1.2
    BASE_TOKEN_THRESHOLD = 8000 
    TOKEN_GROWTH_FACTOR = 1.2
    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    if total_tokens > max_allowed:
        print(f"🔀 [Router] 负载超标 ({total_tokens}/{max_allowed} tokens)，路由至 compress_context 节点")
        return "compress_context"
    
    print(f"🔀 [Router] 负载安全 ({total_tokens}/{max_allowed} tokens)，继续交由 orchestrator 处理")
    return "orchestrator"
# --- 3. 节点实现 (Nodes) ---

class AdvancedResearchGraph:
    # def __init__(self):
    #     self.llm = LocalQwenChatModel(temperature=0.1)
    #     self.vlm = LocalMultimodalModel.get_shared()
        
    #     # 将你的工具列表准备好
    #     self.tools_list = [text_retrieval, image_retrieval, cross_doc_compare, citation_formatter, save_session_summary, save_session_summary_sync]
    #     # 注意：此处要求 LocalQwenTextLLM 支持 bind_tools。如果不原生支持，你需要自己写一个判断工具调用的 Node。
    #     if hasattr(self.llm, "bind_tools"):
    #         self.llm_with_tools = self.llm.bind_tools(self.tools_list)
    #     else:
    #         self.llm_with_tools = self.llm # 兜底方案
            
    #     self.tool_node = ToolNode(self.tools_list)
    def __init__(self):
        # 你的硅基流动 API Key
        # 建议生产环境用 os.getenv("SILICONFLOW_API_KEY") 读取环境变量
        
        siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
        # 直接使用标准的 ChatOpenAI 调用硅基流动
        self.llm = ChatOpenAI(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507", # 硅基流动上的模型 ID
            api_key=siliconflow_api_key,
            base_url="https://api.siliconflow.cn/v1", # 硅基流动的 API 地址
            temperature=0.1,
            max_tokens=2048
        )
        print(f"🌟 [System Init] 成功连接并初始化大模型: {self.llm.model_name}")
        # 将你的工具列表准备好
        self.tools_list = [text_retrieval, image_retrieval, cross_doc_compare, citation_formatter]
        
        # 💡 因为你现在用的是标准 ChatOpenAI，它原生完美支持 bind_tools！
        # 再也不用担心模型自己乱改参数或格式错乱了
        self.llm_with_tools = self.llm.bind_tools(self.tools_list)
        self.tool_node = ToolNode(self.tools_list)
    # ======== 主图节点 ========
    def summarize_history(self, state: MainState):
        """提取对话历史，并清空冗余的原始 Message"""
        messages = state.get("messages", [])
        current_summary = state.get("conversation_summary", "")
        
        # 假设：如果消息少于 3 条（比如只有最新的 HumanMessage），不压缩
        if len(messages) <= 3:
            return {"conversation_summary": current_summary}
            
        print("🧹 [Main] 正在压缩全局对话历史并清理 Token...")
        
        # 提取需要被总结的旧消息（排除最后一条用户的新问题）
        old_messages = messages[:-1]
        old_text = "\n".join([f"{m.type}: {m.content}" for m in old_messages if m.content])
        
        prompt = (
            f"请将以下之前的对话摘要与最新的一轮对话合并，提炼出核心要点和用户的偏好。\n"
            f"【已有摘要】：{current_summary}\n"
            f"【最新对话】：{old_text}"
        )
        new_summary = self.llm.invoke(prompt)
        
        # 核心机制：阅后即焚！生成 RemoveMessage 列表，依据 ID 删除旧消息
        # 这样主图的状态里就永远只有 current_summary 和最新的一条提问
        delete_messages = [RemoveMessage(id=m.id) for m in old_messages if m.id]

        return {
            "conversation_summary": new_summary,
            "messages": delete_messages # 将删除指令返回给状态机
        }

    def rewrite_query(self, state: MainState):
        """拆解问题（替代你之前的 Planner），并安全解析 JSON 输出"""
        original_query = state.get('originalQuery', '')
        summary = state.get('conversation_summary', '')
        
        print("🔍 [Main] 正在评估和拆解用户问题...")
        
        # 1. 构造严格的 Prompt
        prompt = f"""
        你是一名专业的学术问题拆解专家。
        【历史对话摘要】：{summary}
        【用户当前问题】：{original_query}
        
        请分析用户的当前问题是否清晰。
        - 如果清晰，请将其拆解为 1 到 3 个独立、自包含的子查询（以便并行在文献库中检索）。
        - 如果不清晰（比如代指不明，或者过于宽泛），请将 is_clear 设为 false。
        
        你必须严格输出以下 JSON 格式，不要包含任何其他废话：
        ```json
        {{
            "is_clear": true,
            "sub_queries": ["子查询1", "子查询2"]
        }}
        ```
        """
        
        # 2. 调用大模型
        response_text = self.llm.invoke(prompt)
        # 如果你的 invoke 返回的是 AIMessage 对象，请使用 response_text.content
        if hasattr(response_text, "content"):
            response_text = response_text.content
            
        # 3. 设置安全的默认兜底值 (Fallback)
        is_clear = True
        sub_queries = [original_query] # 兜底：如果解析失败，就把原问题当成唯一子问题
        
        # 4. 鲁棒的 JSON 解析提取逻辑
        try:
            # 找到第一个左大括号和最后一个右大括号的位置
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            # 如果成功找到了配对的大括号
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx : end_idx + 1]
            else:
                # 兜底：如果没有找到大括号，说明模型可能直接输出了纯文本，尝试直接解析
                json_str = response_text
                
            # 解析字符串为 Python 字典
            data = json.loads(json_str)
            
            # 安全地获取字段
            is_clear = data.get("is_clear", True)
            parsed_queries = data.get("sub_queries", [])
            
            if parsed_queries and isinstance(parsed_queries, list):
                sub_queries = parsed_queries
                
            print(f"✅ [Main] 问题拆解成功: {sub_queries}")
            
        except json.JSONDecodeError as e:
            print(f"⚠️ [Main] JSON 解析失败，已回退到原问题。模型原始输出: {response_text}\n错误信息: {e}")
        except Exception as e:
            print(f"⚠️ [Main] 发生未知错误，已回退到原问题。错误信息: {e}")
        return {
            "questionIsClear": is_clear, 
            "rewrittenQuestions": sub_queries
        }
        
    def request_clarification(self, state: MainState):
        """问题不清晰时向用户提问"""
        return {"messages": ["请问您具体指的是哪篇论文？"]}

    def aggregate_answers(self, state: MainState):
        """汇总多个子图的答案"""
        answers = state.get("agent_answers", [])
        
        original_query = state.get("originalQuery", "")
        
        # 兜底：如果没收到子图答案
        if not answers:
            return {"messages": [AIMessage(content="抱歉，未能收集到子问题的有效检索结果。")]}
            
        # 根据 index 排序
        answers.sort(key=lambda x: x["index"])
        combined_text = "\n\n".join([f"【子问题】: {ans['question']}\n【检索结论】: {ans['answer']}" for ans in answers])
        
        # prompt = f"请根据以下各个子问题的检索结论，整合出一份逻辑连贯、重点突出的最终回答：\n\n{combined_text}"
        prompt = (
            f"用户提出的原始问题是：'{original_query}'\n\n"
            f"请根据以下各个子问题的检索结论，整合出一份逻辑连贯、重点突出的最终回答：\n\n{combined_text}\n\n"
            f"【⚠️重要要求】：请必须使用与用户原始问题相同的语言（如果问题是英文，请用英文回答；如果是中文，请用中文回答）来进行总结！"
            f"【绝对约束条件】："
            f"1. 核心回溯（关键！）：你的回答第一句话必须包含原问题中的核心主语、专有名词和前提条件，使其成为一个完整的陈述句。"
            f"2. 极简原则：问题问什么就只回答什么，严禁在罗列事实后附加任何关于“为什么这样做”或“有什么优缺点/作用”的总结和解释。"
            f"3. 语气要求：像一个冷酷的数据库一样返回数据，绝不要使用“综上所述”、“总之”等废话。"
        )
        # 建议此处用纯文本 LLM (self.llm) 而不是 VLM，除非你的结果里有图片需要识别
        response = self.llm.invoke([HumanMessage(content=prompt)])
        final_answer = response.content if hasattr(response, "content") else str(response)
        
        return {"messages": [AIMessage(content=final_answer)]}

    # ======== 子图节点 ========
    def orchestrator(self, state: SubAgentState):
        current_iter = state.get("iteration_count", 0) + 1
        print(f"🧠 [Sub] Orchestrator 思考中... (迭代次数: {current_iter})")
        
        if current_iter == 1:
            print(f"⚡ [Sub] 第一轮强制接管：不让模型思考，直接派发检索任务！")
            from langchain_core.messages import AIMessage
            
            # 我们伪造一个完美的大模型工具调用输出
            forced_tool_call = AIMessage(
                content="",
                tool_calls=[{
                    "name": "text_retrieval",
                    "args": {"query": state["question"]}, # 直接把子问题当查询词
                    "id": f"call_forced_{current_iter}",
                    "type": "tool_call"
                }]
            )
            
            # 记录到动作记忆，并更新状态
            return {
                "messages": [forced_tool_call], 
                "iteration_count": current_iter,
                "tool_call_count": state.get("tool_call_count", 0) + 1,
                "retrieval_keys": [f"text_retrieval::{state['question']}"]
            }
        
        # 1. 从持久化状态中获取过去查过的 keys (无惧阅后即焚)
        past_keys = state.get("retrieval_keys", [])
        keys_str = ", ".join(past_keys) if past_keys else "无"
        
        sys_msg = SystemMessage(
            content=(
                f"你是一名严谨的研究助手，当前正在解决子问题: '{state['question']}'。\n\n"
                f"【已有知识摘要】:\n{state.get('context_summary', '无')}\n\n"
                f"【⚠️最高指令（防死循环）】：\n"
                f"1. 你已经执行过以下检索关键词: {keys_str}\n"
                f"2. 绝对禁止再次使用上述任何一个关键词进行检索！\n"
                f"3. 如果你发现你要检索的词已经在列表中，说明你找不到更多信息了，请立刻停止调用工具！基于【已有知识摘要】直接输出一段最终答案（即使信息不全）。\n"
            )
        )
        
        messages_to_model = [sys_msg] + state.get("messages", [])
        response = self.llm_with_tools.invoke(messages_to_model)
        
        print(f"🐛 [Debug] 模型原始文本回复: {response.content}")
        print(f"🐛 [Debug] 模型解析出的 Tool Calls: {getattr(response, 'tool_calls', '无')}")
        
        # 2. 判断是否调用工具
        is_tool_call = 1 if (hasattr(response, "tool_calls") and response.tool_calls) else 0
        
        # 3. 核心机制：一旦决定调用工具，立刻提取关键词存入长期记忆
        new_keys = []
        if is_tool_call:
            for tc in response.tool_calls:
                if tc["name"] in ["text_retrieval", "image_retrieval"]:
                    query = tc["args"].get("query", "")
                    if query:
                        new_keys.append(f"{tc['name']}::{query}")
                        print(f"🧠 [Sub] 记录检索动作到长期记忆: {query}")

        return {
            "messages": [response], 
            "iteration_count": current_iter,
            "tool_call_count": state.get("tool_call_count", 0) + is_tool_call,
            "retrieval_keys": new_keys # 依靠 operator.add 自动追加，不会被删除
        }

    def fallback_response(self, state: SubAgentState):
        """当系统达到最大搜索限制时的强制兜底回答节点"""
        # prompt = f"尽力回答：{state['question']}，基于：{state.get('context_summary')}"
        prompt = """You are an expert synthesis assistant. The system has reached its maximum research limit.

                Your task is to provide the most complete answer possible using ONLY the information provided below.

                Input structure:
                - "Compressed Research Context": summarized findings from prior search iterations — treat as reliable.
                - "Retrieved Data": raw tool outputs from the current iteration — prefer over compressed context if conflicts arise.
                Either source alone is sufficient if the other is absent.

                Rules:
                1. Source Integrity: Use only facts explicitly present in the provided context. Do not infer, assume, or add any information not directly supported by the data.
                2. Handling Missing Data: Cross-reference the USER QUERY against the available context.
                Flag ONLY aspects of the user's question that cannot be answered from the provided data.
                Do not treat gaps mentioned in the Compressed Research Context as unanswered
                unless they are directly relevant to what the user asked.
                3. Tone: Professional, factual, and direct.
                4. Output only the final answer. Do not expose your reasoning, internal steps, or any meta-commentary about the retrieval process.
                5. Do NOT add closing remarks, final notes, disclaimers, summaries, or repeated statements after the Sources section.
                The Sources section is always the last element of your response. Stop immediately after it.

                Formatting:
                - Use Markdown (headings, bold, lists) for readability.
                - Write in flowing paragraphs where possible.
                - Conclude with a Sources section as described below.

                Sources section rules:
                - Include a "---\\n**Sources:**\\n" section at the end, followed by a bulleted list of file names.
                - List ONLY entries that have a real file extension (e.g. ".pdf", ".docx", ".txt").
                - Any entry without a file extension is an internal chunk identifier — discard it entirely, never include it.
                - Deduplicate: if the same file appears multiple times, list it only once.
                - If no valid file names are present, omit the Sources section entirely.
                - THE SOURCES SECTION IS THE LAST THING YOU WRITE. Do not add anything after it.
                """
        ans = self.llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [ans]}

    # def compress_context(self, state: SubAgentState):
    #     """当检索文档过长时：压缩知识，并删除原始的冗长 ToolMessage"""
    #     print(" [Sub] 正在压缩局部工作记忆...")
    #     messages = state.get("messages", [])
    #     current_summary = state.get("context_summary", "")
        
    #     # 把所有的消息内容拼起来（这里面包含了几万字的文献）
    #     full_text = "\n".join([str(m.content) for m in messages if m.content])
        
    #     # 告诉模型之前已经查过什么，防止它失忆后重复查
    #     past_keys = state.get("retrieval_keys", [])
    #     keys_str = ", ".join(past_keys) if past_keys else "无"
        
    #     prompt = (
    #         f"请从以下海量检索结果中提取与子问题 '{state['question']}' 相关的核心事实。\n"
    #         f"【注意：你已经查过以下关键词，后续无需重复查找】：{keys_str}\n"
    #         f"【已有知识摘要】：{current_summary}\n"
    #         f"【最新检索原始文本】：{full_text}"
    #     )
    #     compressed_knowledge_msg = self.llm.invoke([HumanMessage(content=prompt)])
    
    #     # 🚨 核心修复：提取 content 变成纯文本！
    #     compressed_text = compressed_knowledge_msg.content if hasattr(compressed_knowledge_msg, "content") else str(compressed_knowledge_msg)
        
    #     delete_messages = [RemoveMessage(id=m.id) for m in messages if m.id]

    #     return {
    #         "context_summary": compressed_text, # 这里存入纯字符串
    #         "messages": delete_messages 
    #     }
    # def compress_context(self, state: SubAgentState):
    #     """当检索文档过长时：压缩知识，并安全清理冗长的工具消息"""
    #     print("🗜️ [Sub] 正在压缩局部工作记忆...")
    #     messages = state.get("messages", [])
    #     current_summary = state.get("context_summary", "")
        
    #     # 提取所有文本准备压缩
    #     full_text = "\n".join([str(m.content) for m in messages if m.content])
        
    #     past_keys = state.get("retrieval_keys", [])
    #     keys_str = ", ".join(past_keys) if past_keys else "无"
        
    #     # 🌟 优化 Prompt：强保真压缩，防止丢弃“干草堆里的针”
    #     prompt = (
    #         f"你是一个严谨的学术资料整理员。请从以下海量检索结果中提取与问题 '{state['question']}' 相关的核心内容。\n"
    #         f"【严格约束】：\n"
    #         f"1. 必须原封不动地保留所有与问题相关的具体数值、指标（如 F1 分数）、超参数（如 a=0.30）、模块缩写及全称。\n"
    #         f"2. 保留不同算法之间的对比优缺点，不要做模糊化概括。\n"
    #         f"【已检索关键词，后续避开】：{keys_str}\n"
    #         f"【已有知识摘要】：{current_summary}\n"
    #         f"【最新检索文本】：\n{full_text}"
    #     )
    #     compressed_knowledge_msg = self.llm.invoke([HumanMessage(content=prompt)])
    #     compressed_text = compressed_knowledge_msg.content if hasattr(compressed_knowledge_msg, "content") else str(compressed_knowledge_msg)
        
    #     # 🌟 核心修复：精准清理内存，绝不误伤用户问题
    #     delete_messages = []
    #     for m in messages:
    #         # 只删除极其占地方的 ToolMessage (检索结果)
    #         # 以及没有工具调用的普通 AIMessage (过期的思考过程)
    #         if isinstance(m, ToolMessage) and m.id:
    #             delete_messages.append(RemoveMessage(id=m.id))
    #         elif isinstance(m, AIMessage) and not m.tool_calls and m.id:
    #             delete_messages.append(RemoveMessage(id=m.id))

    #     return {
    #         "context_summary": compressed_text, 
    #         "messages": delete_messages 
    #     }
    def compress_context(self, state: SubAgentState):
        """当检索文档过长时：高保真压缩知识，注入防重放机制，清理内存"""
        print("🗜️ [Sub] 正在执行动态局部工作记忆压缩...")
        messages = state.get("messages", [])
        existing_summary = state.get("context_summary", "").strip()

        # ==========================================
        # 1. 提取刚才调用过的工具和关键词 (防死循环核心)
        # ==========================================
        new_keys = set()
        for msg in reversed(messages):
            # 寻找 AI 调用工具的痕迹
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    tool_name = tc["name"]
                    args = tc.get("args", {})
                    # 适配你自己的工具名称
                    if tool_name in ["text_retrieval", "cross_doc_compare", "image_retrieval"]:
                        query = args.get("query", "")
                        if query:
                            new_keys.add(f"{tool_name}::{query}")
                break # 只抓取最近一次的调用，防止过度累积
                
        # 合并新旧查询记录
        updated_keys = set(state.get("retrieval_keys", [])) | new_keys

        # ==========================================
        # 2. 构建待压缩的对话回忆录
        # ==========================================
        conversation_text = f"【当前核心问题】:\n{state.get('question')}\n\n【需要压缩的对话内容】:\n\n"
        if existing_summary:
            conversation_text += f"[之前的摘要记忆]\n{existing_summary}\n\n"

        for msg in messages[1:]: # 跳过最开始的 SystemPrompt
            if isinstance(msg, AIMessage):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in getattr(msg, "tool_calls", []))
                tool_info = f" | 执行了工具: {calls}" if calls else ""
                conversation_text += f"[助手思考{tool_info}]\n{msg.content or '(仅调用工具)'}\n\n"
            elif isinstance(msg, ToolMessage):
                conversation_text += f"[工具返回的大段文献 — {msg.name}]\n{msg.content}\n\n"

        # ==========================================
        # 3. 强约束 Prompt (保护你的 Golden Dataset 细节)
        # ==========================================
        compression_prompt = (
            "你是一个严谨的遥感学术资料整理员。你的任务是将冗长的对话记录压缩为精炼的【工作记忆摘要】。\n"
            "【绝对遵守的纪律】：\n"
            "1. 数据保真：必须原封不动地保留上下文中出现的具体数值（如 F1: 97.96%）、超参数（如 λ=0.01）、模块英文缩写（如 DCAU）。绝对不许用“多项指标”这种废话概括！\n"
            "2. 剔除废话：删掉助手的寒暄、思考过程的废话，只保留文献中的事实结论和对比差异。\n"
        )
        # compression_prompt = (
        #     """You are an expert research context compressor.

        #     Your task is to compress retrieved conversation content into a concise, query-focused, and structured summary that can be directly used by a retrieval-augmented agent for answer generation.

        #     Rules:
        #     1. Keep ONLY information relevant to answering the user's question.
        #     2. Preserve exact figures, names, versions, technical terms, and configuration details.
        #     3. Remove duplicated, irrelevant, or administrative details.
        #     4. Do NOT include search queries, parent IDs, chunk IDs, or internal identifiers.
        #     5. Organize all findings by source file. Each file section MUST start with: ### filename.pdf
        #     6. Highlight missing or unresolved information in a dedicated "Gaps" section.
        #     7. Limit the summary to roughly 400-600 words. If content exceeds this, prioritize critical facts and structured data.
        #     8. Do not explain your reasoning; output only structured content in Markdown.

        #     Required Structure:

        #     # Research Context Summary

        #     ## Focus
        #     [Brief technical restatement of the question]

        #     ## Structured Findings

        #     ### filename.pdf
        #     - Directly relevant facts
        #     - Supporting context (if needed)

        #     ## Gaps
        #     - Missing or incomplete aspects

        #     The summary should be concise, structured, and directly usable by an agent to generate answers or plan further retrieval.
        #     """
        # )
        summary_response = self.llm.invoke([
            SystemMessage(content=compression_prompt), 
            HumanMessage(content=conversation_text)
        ])
        new_summary = summary_response.content if hasattr(summary_response, "content") else str(summary_response)

        # ==========================================
        # 4. 强制注入“已执行动作”，防止 Agent 失忆重复搜索
        # ==========================================
        if updated_keys:
            block = "\n\n---\n**【系统强制警告：以下检索已执行，绝对禁止再次用相同关键词查知识库！】**\n"
            block += "\n".join(f"- {k}" for k in updated_keys) + "\n"
            new_summary += block

        # ==========================================
        # 5. 精准清理内存 (安全删除)
        # ==========================================
        delete_messages = []
        for m in messages:
            # 清理冗长的工具返回和不含工具调用的中间思考过程，保留原始提问
            if isinstance(m, ToolMessage) and m.id:
                delete_messages.append(RemoveMessage(id=m.id))
            elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None) and m.id:
                delete_messages.append(RemoveMessage(id=m.id))

        return {
            "context_summary": new_summary, 
            "retrieval_keys": list(updated_keys), # 更新查询历史状态
            "messages": delete_messages 
        }
    def collect_answer(self, state: SubAgentState):
        """收集子图的答案，并提取所有真实的检索文献供 Ragas 评测"""
        messages = state.get("messages", [])
        
        # 1. 提取最终回答
        if messages:
            last_msg = messages[-1]
            answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        else:
            answer = "未获取到有效回答。"

        # 2. 🚨 核心修复：遍历对话历史，把工具真正查回来的原文提取出来
        retrieved_docs = []
        
        # 加上可能存在的压缩摘要
        if state.get("context_summary"):
            retrieved_docs.append(state["context_summary"])
            
        # 遍历消息，找出 ToolMessage (或者是带有 name 属性的返回结果)
        for msg in messages:
            if hasattr(msg, "name") and msg.name in ["text_retrieval", "image_retrieval", "cross_doc_compare"]:
                if msg.content and "Error" not in str(msg.content): # 排除报错信息
                    retrieved_docs.append(str(msg.content))
                    
        # 拼成一个大字符串
        full_context = "\n\n".join(retrieved_docs)
        if not full_context.strip():
            full_context = "未检索到外部知识。"

        return {
            "agent_answers": [{
                "index": state.get("question_index", 0),
                "question": state.get("question", ""),
                "answer": answer,
                "context": full_context # 现在 Ragas 能看到真东西了！
            }]
        }
        

# --- 4. 构建与编译图 ---

def create_research_graph():
    agent_instance = AdvancedResearchGraph()
    
    # ------------------ 构建子图 (Agent Subgraph) ------------------
    agent_builder = StateGraph(SubAgentState)
    
    agent_builder.add_node("orchestrator", agent_instance.orchestrator)
    agent_builder.add_node("tools", agent_instance.tool_node) # 直接使用 LangGraph 标准工具节点
    agent_builder.add_node("compress_context", agent_instance.compress_context)
    agent_builder.add_node("fallback_response", agent_instance.fallback_response)
    agent_builder.add_node("collect_answer", agent_instance.collect_answer)
    
    agent_builder.add_edge(START, "orchestrator")    
    agent_builder.add_conditional_edges("orchestrator", route_after_orchestrator)
    # tools 节点调用完后检查是否需要压缩
    agent_builder.add_conditional_edges("tools", should_compress_context)
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)
    
    agent_subgraph = agent_builder.compile()

    # ------------------ 构建主图 (Main Graph) ------------------
    graph_builder = StateGraph(MainState)
    
    graph_builder.add_node("summarize_history", agent_instance.summarize_history)
    graph_builder.add_node("rewrite_query", agent_instance.rewrite_query)
    graph_builder.add_node("request_clarification", agent_instance.request_clarification)
    # 关键：将编译好的子图作为一个普通 Node 添加到主图中！
    graph_builder.add_node("agent", agent_subgraph) 
    graph_builder.add_node("aggregate_answers", agent_instance.aggregate_answers)
    
    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    # 这里通过 Send API 实现并行 Map
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query") # 也可以指向 END
    
    # 子图 (Send任务) 全部执行完毕后，流入汇总节点 (Reduce)
    graph_builder.add_edge("agent", "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)
    
    # 创建内存检查点（在生产环境中可以替换为 Postgres 或 Redis Checkpointer）
    memory = MemorySaver()
    # 编译主图时传入 checkpointer
    app = graph_builder.compile(checkpointer=memory)
    return app

if __name__ == "__main__":
    app = create_research_graph()
    
    # 必须指定 thread_id，LangGraph 靠这个识别是哪一次对话
    config = {"configurable": {"thread_id": "harper_session_001"}}
    
    # 第一轮对话
    inputs1 = {"messages": [HumanMessage(content="你好，能介绍一下变化检测常用的损失函数吗？")]}
    res1 = app.invoke(inputs1, config=config)
    
    # 第二轮对话
    # 此时系统会自动触发 summarize_history，将第一轮对话压缩，并移除第一轮的原始 Message
    inputs2 = {"messages": [HumanMessage(content="那我应该怎么在 ChangeFormer 里使用它？")]}
    res2 = app.invoke(inputs2, config=config)