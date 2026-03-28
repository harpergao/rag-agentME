"""MCP 风格工具集合：用于多智能体调用的本地工具。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from langchain.tools import tool  # type: ignore
from typing import List, Dict, Any
import json

REPORT_DIR = Path("./output/reports")

# 导入你已有的组件
from model_client import LocalMultimodalModel
from rag_pipeline import build_retriever, format_documents, get_knowledge_hash, DEFAULT_KNOWLEDGE_BASE

# 1. 文字检索工具 (封装原有 RAG 逻辑)
@tool("text_retrieval")
def text_retrieval(query: str) -> str:
    """
    针对文档文字内容进行混合检索。
    适用于：单点查询某个观点、查找特定技术细节、定义或方法论。
    【重要输入要求】：query 必须是具体的实体或简短的关键词组合，不要输入完整的长句子,也可以输入自然语言短句，系统会自动处理语义匹配。
    例如输入："Transformer 模型架构"、"本文提出的主要创新点" 或 "实验部分的评估指标"。
    """
    current_hash = get_knowledge_hash(DEFAULT_KNOWLEDGE_BASE)
    retriever = build_retriever(dir_hash=current_hash) 
    if not retriever:
        return "知识库为空，无法检索。"
    docs = retriever.invoke(query)
    return format_documents(docs) 

# 2. 图表检索工具 (多模态核心)
@tool("image_retrieval")
def image_retrieval(query: str) -> str:
    """
    检索文档中的图片、图表或实验结果展示。
    适用于：查找特定的折线图、性能对比表、流程图或架构图。
    【重要输入要求】：query 需要明确指明找什么图。
    例如输入："模型整体架构图"、"不同算法在各数据集上的性能对比表" 或 "消融实验结果图"。
    """
    client = LocalMultimodalModel.get_shared() 
    prompt = f"请在文档图表库中查找与以下描述最相关的图表路径及描述：{query}"
    res = client.generate(prompt) 
    return f"检索到相关图表：{res}"

# 3. 跨文献对比工具
@tool("cross_doc_compare")
def cross_doc_compare(query: str) -> str:
    """
    针对给定的文档概念进行对比分析。
    【重要输入要求】：query 需要明确说明对比的对象，例如：“对比不同文献中关于数据预处理的方法” 或 “对比这几种主流算法的优缺点”。
    注意：此工具内部会自动在知识库中检索相关内容进行对比，你无需提供全文。
    """
    current_hash = get_knowledge_hash(DEFAULT_KNOWLEDGE_BASE)
    retriever = build_retriever(dir_hash=current_hash)
    if not retriever:
        return "错误：知识库为空，无法进行对比分析。"
    
    docs = retriever.invoke(query)
    context = format_documents(docs)

    if not context or "未检索到" in context:
        return "错误：没有找到相关文献内容，无法进行对比分析。可能知识库中缺乏该领域资料。"

    client = LocalMultimodalModel.get_shared()
    
    prompt = f"""
    你是一名严谨的分析专家。
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
    将检索到的原始元数据（标题、作者、页码）格式化为标准引用格式。
    【重要输入要求】：metadata_list 是一个包含 source 和 page 的字典列表。
    """
    citations = []
    for meta in metadata_list:
        source = meta.get("source", "未知文献")
        section_id = meta.get("parent_id", "未知段落")
        citations.append(f"[{source}, §{section_id}]")
    return "\n".join(set(citations))


def _ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]+", "-", text.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "session"

@tool("save_session_summary")
def save_session_summary(content: str, title: str = "对话总结") -> str:
    """
    保存会话纪要到本地 Markdown 文件，用于归档和记录。
    【重要输入要求】：content 必须是包含排版的 Markdown 长文本。
    """
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