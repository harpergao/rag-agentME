"""MCP 风格工具集合：用于多智能体调用的本地工具。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from langchain.tools import tool  # type: ignore

REPORT_DIR = Path("./output/reports")
from langchain.tools import tool
from typing import List, Dict, Any
import json
# 导入你已有的组件
from model_client import LocalMultimodalModel
from rag_pipeline import build_retriever, format_documents

# 1. 文字检索工具 (封装原有 RAG 逻辑)
@tool("text_retrieval")
def text_retrieval(query: str) -> str:
    """
    针对论文文字内容进行 FAISS+BM25 混合检索。
    适用于：单点查询某个观点、查找特定技术细节。
    """
    retriever = build_retriever() # 使用 rag_pipeline.py 中的构建函数
    if not retriever:
        return "知识库为空，无法检索。"
    docs = retriever.invoke(query)
    return format_documents(docs) # 使用 rag_pipeline.py 中的格式化函数

# 2. 图表检索工具 (多模态核心)
@tool("image_retrieval")
def image_retrieval(query: str) -> str:
    """
    检索论文中的图片、图表或实验结果。
    适用于：查找特定折线图、消融实验表或模型架构图。
    """
    client = LocalMultimodalModel.get_shared() # 使用 model_client.py 中的多模态模型
    # 模拟逻辑：在实际生产中，这里应接一个视觉向量数据库 (如 Milvus + CLIP)
    # 目前可以先让模型根据 query 在已知图表描述中搜索
    prompt = f"请在论文图表库中查找与以下描述最相关的图表路径及描述：{query}"
    res = client.generate(prompt) 
    return f"检索到相关图表：{res}"

# 3. 跨文献对比工具
# multi_agent/tools.py
@tool("cross_doc_compare")
def cross_doc_compare(query: str, context: str) -> str:
    """
    针对给定的文献内容（context）进行对比分析。
    必须严格基于提供的 context 进行对比，禁止虚构不存在的文献。
    """
    if not context or "未检索到" in context:
        return "错误：没有找到相关文献内容，无法进行对比分析。请先进行文本检索。"

    client = LocalMultimodalModel.get_shared()
    
    # 强化 Prompt：明确禁止幻觉
    prompt = f"""
    你是一名严谨的学术论文评价专家。
    【任务】请根据下方提供的【参考资料】，完成以下对比任务："{query}"
    
    【参考资料】：
    {context}
    
    【约束条件】：
    1. 严禁编造任何作者、年份或结论。
    2. 所有的对比观点必须在【参考资料】中找到原文依据。
    3. 如果资料中没有提到某个维度，请直接说明“资料未提及”，不要自行推测。
    4. 必须使用 [Doc X] 标注来源。
    
    请输出 Markdown 对比表格：
    """
    return client.generate(prompt)

# 4. 引用格式化工具
@tool("citation_formatter")
def citation_formatter(metadata_list: List[Dict[str, Any]]) -> str:
    """
    将检索到的原始元数据（标题、作者、页码）格式化为标准学术引用格式。
    """
    citations = []
    for meta in metadata_list:
        source = meta.get("source", "未知来源")
        page = meta.get("page", "N/A")
        # 这里可以根据你的需求定制格式，例如 [Paper: XXX, Page: XX]
        citations.append(f"[{source}, p.{page}]")
    return "\n".join(set(citations))

# 保留你原有的 save_session_summary
# ... (原有的 save_session_summary 代码)

def _ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]+", "-", text.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "session"


@tool("save_session_summary")
def save_session_summary(content: str, title: str = "对话总结") -> str:
    """保存会话纪要到 Markdown，模仿 MCP server 的归档指令。"""

    _ensure_report_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}_{_slugify(title)}.md"
    file_path = REPORT_DIR / filename
    header = f"# {title}\n\n"
    with file_path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.write(content.strip())
        f.write("\n")
    return str(file_path.resolve())


def save_session_summary_sync(content: str, title: str = "对话总结") -> str:
    """便捷函数：直接调用 LangChain 工具的同步实现。"""

    return save_session_summary.invoke({"content": content, "title": title})
