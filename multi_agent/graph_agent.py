import operator
import json
from typing import Annotated, Dict, List, Optional, TypedDict, Any
from langgraph.graph import StateGraph, END

# 导入现有组件
from rag_pipeline import build_retriever, format_documents 
from model_client import LocalMultimodalModel 
from langchain_wrappers import LocalQwenTextLLM 
# 导入你在 tools.py 中新写的工具
from tools import text_retrieval, image_retrieval, cross_doc_compare, citation_formatter

# 1. 完善状态定义
class AgentState(TypedDict):
    question: str
    query_type: str 
    task_list: List[Dict[str, Any]]  # 例如: [{"type": "text_retrieval", "query": "..."}]
    context: Annotated[List[str], operator.add] 
    answer: str
    review_feedback: Optional[str]
    retry_count: Annotated[int, operator.add]

# 2. 节点逻辑实现
class GraduateResearchGraph:
    def __init__(self):
        self.llm = LocalQwenTextLLM()
        self.vlm = LocalMultimodalModel.get_shared()
        # 将工具映射表放入类内部，方便调用
        self.tools_map = {
            "text_retrieval": text_retrieval,
            "image_retrieval": image_retrieval,
            "cross_doc_compare": cross_doc_compare,
            "citation_formatter": citation_formatter
        }

    def planner_node(self, state: AgentState):
        """
        职责：分析问题类型，生成严格的 JSON 任务列表
        """
        prompt = f"""
                你是一名学术规划专家。请分析用户问题："{state['question']}"
                将其拆解为执行步骤。
                
                1. 如果涉及“对比”、“差异”、“关联”，必须先调用 'text_retrieval' 检索相关文档。
                2. 在检索完成后，再调用 'cross_doc_compare' 进行总结。
                3. 严禁在没有检索步骤的情况下直接进行对比。
                
                可选工具（必须从以下列表中选择）：
                1. text_retrieval: 用于搜索论文文字观点。
                2. image_retrieval: 用于查找图表、实验结果图。
                3. cross_doc_compare: 用于多篇论文之间的横向对比分析。
                
                必须输出 JSON 数组格式。示例：
                [
                    {{"step": 1, "type": "text_retrieval", "query": "变化检测损失函数"}},
                    {{"step": 2, "type": "cross_doc_compare", "query": "对比不同损失函数的收敛速度"}}
                ]
                """
        # response = self.llm.generate(prompt)
        response = self.llm.invoke(prompt)
        # 尝试解析 JSON（实际工程中建议增加异常处理）
        try:
            # 找到 JSON 数组的起始位置，防止模型输出废话
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            tasks = json.loads(response[start_idx:end_idx])
        except:
            # 兜底：如果解析失败，默认进行一次文字检索
            tasks = [{"step": 1, "type": "text_retrieval", "query": state['question']}]
        
        return {"task_list": tasks}

# multi_agent/graph_agent.py

    def manager_node(self, state: AgentState):
        tasks = state.get("task_list", [])
        # 我们需要一个临时变量来存储当前节点产生的最新 context
        current_accumulated_context = "\n".join(state.get("context", []))
        new_results = []
        
        for task in tasks:
            tool_type = task.get("type")
            query = task.get("query")
            
            if tool_type in self.tools_map:
                tool_func = self.tools_map[tool_type]
                
                try:
                    # 如果是对比任务，把之前检索到的 context 传进去
                    if tool_type == "cross_doc_compare":
                        result = tool_func.invoke({
                            "query": query, 
                            "context": current_accumulated_context # 核心：把米给厨师
                        })
                    else:
                        # 普通检索任务
                        result = tool_func.invoke({"query": query})
                        # 实时更新临时上下文，以便后续步骤（如对比）能用到
                        current_accumulated_context += f"\n{result}"
                    
                    new_results.append(f"--- {tool_type} 结果 ---\n{result}")
                except Exception as e:
                    new_results.append(f"❌ {tool_type} 调用失败: {str(e)}")
            
        return {"context": new_results}

    def response_node(self, state: AgentState):
        """
        职责：综合所有 context 生成最终回答
        """
        current_retry = state.get("retry_count", 0)
        print(f"✍️ [Node: Responder] 正在生成回答 (第 {current_retry + 1} 次尝试)...")
        context_str = "\n\n".join(state["context"])
        feedback = f"\n\n【Review 反馈（请针对此改进）】：{state['review_feedback']}" if state.get("review_feedback") else ""
        
        prompt = f"""
                你是一名严格的文献搬运工。
                【准则】：你只能根据提供的【资料内容】回答。资料中没有的内容，你必须回答不知道，严禁使用你自己的知识。
        
        【资料内容】：
        {context_str}
        
        【用户问题】：
        {state['question']}
        {feedback}
        """
        answer = self.vlm.generate(prompt=prompt)
        # answer = self.vlm.invoke(prompt)
        return {"answer": answer, "retry_count": 1}

    def reviewer_node(self, state: AgentState):
            """
            职责：纯粹的路由函数，不再尝试修改 State
            """
            # 注意：这里的 state 已经是 responder_node 增加过后的了
            if state.get("retry_count", 0) > 1: # 如果已经尝试过 2 次（初始 0 + 两次 responder），则强制结束
                print("🛑 [Reviewer] 已达到最大重试次数，强制输出结果。")
                return "pass"
                
            print("⚖️ [Reviewer] 正在质检回答质量...")
            check_prompt = f"判断以下回答是否准确引用了来源：{state['answer']}。只需输出 PASS 或 RETRY。"
            # review_result = self.llm.generate(check_prompt)
            review_result = self.llm.invoke(check_prompt)
            
            if "PASS" in review_result.upper():
                return "pass"
            else:
                print(f"⚠️ [Reviewer] 质检未通过，准备重试。反馈：{review_result}")
                return "retry"

# 3. 编译图逻辑（保持不变）
def create_research_graph():
    agent = GraduateResearchGraph()
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", agent.planner_node)
    workflow.add_node("manager", agent.manager_node)
    workflow.add_node("responder", agent.response_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "manager")
    workflow.add_edge("manager", "responder")

    workflow.add_conditional_edges(
        "responder",
        agent.reviewer_node,
        {
            "pass": END,
            "retry": "responder"
        }
    )
    return workflow.compile()