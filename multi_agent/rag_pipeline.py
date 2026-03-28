"""
RAG（检索增强生成）流水线：
- 加载本地知识库目录下的文本/Markdown文件；
- 进行分块（避免长文丢失召回）；
- 使用句向量模型嵌入并构建 FAISS 向量索引；
- 暴露检索器以便 Agent 组件查询相关片段；
- 提供格式化函数将检索结果转为可阅读字符串。
"""
 
import os
import hashlib
import pymupdf4llm
import json

from pathlib import Path
from functools import lru_cache
from typing import Any, Iterable, List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, PyMuPDFLoader  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
try:  # LangChain 0.1+ 拆分
    from langchain_core.documents import Document  # type: ignore
except ImportError:  # 回退旧版本
    try:
        from langchain.schema import Document  # type: ignore
    except ImportError:
        from typing import Any as Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter  # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

try:
    from sentence_transformers import CrossEncoder  # type: ignore

    _CROSS_ENCODER_AVAILABLE = True
except ImportError:  # sentence-transformers 缺失时降级
    CrossEncoder = None  # type: ignore
    _CROSS_ENCODER_AVAILABLE = False

try:  # BM25 依赖 rank_bm25，若缺失则降级为纯向量检索
    from langchain_community.retrievers import BM25Retriever  # type: ignore

    _BM25_AVAILABLE = True
except ImportError:
    BM25Retriever = None  # type: ignore
    _BM25_AVAILABLE = False

from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

DEFAULT_KNOWLEDGE_BASE = "./knowledge_base"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
BM25_TOP_K = 8
VECTOR_TOP_K = 8
RERANK_TOP_N = 4

FAISS_INDEX_PATH = "./faiss_index_cache"

PARENT_STORE_DIR = "./parent_chunks_cache"
QDRANT_PATH = "./qdrant_local_db"

TEXT_ENCODINGS = ("utf-8", "utf-16", "gbk", "gb2312", "latin-1")
TEXT_EXTENSIONS = {".txt", ".md", ".mdx", ".csv", ".log"}
PDF_EXTENSIONS = {".pdf"}

import torch
# 动态判断是否有 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=1)
def build_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})


def _load_text_file(path: Path) -> List[Document]:
    """尝试多种编码打开文本文件，失败则忽略异常继续。"""
    for encoding in TEXT_ENCODINGS:
        try:
            loader = TextLoader(str(path), encoding=encoding)
            return loader.load()
        except UnicodeDecodeError:
            continue
    # 最后兜底，以忽略模式读取，避免完全丢失内容
    try:
        loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=False)
        return loader.load()
    except Exception:
        with open(path, "rb") as f:
            content = f.read().decode("utf-8", errors="ignore")
        return [Document(page_content=content, metadata={"source": str(path)})]


def _load_pdf_file_as_md(path: Path) -> List[Document]:
    """使用 pymupdf4llm 提取保留排版和表格的 Markdown"""
    try:
        md_text = pymupdf4llm.to_markdown(str(path), write_images=False)
        return [Document(page_content=md_text, metadata={"source": str(path)})]
    except Exception as e:
        print(f"[RAG] PDF转MD失败 {path}: {e}")
        return []


def _iter_knowledge_files(knowledge_dir: str) -> Iterable[Path]:
    base = Path(knowledge_dir)
    if not base.exists():
        return []
    return (file for file in base.rglob("*") if file.is_file())

# 构建检索器：加载 knowledge_base/ 目录下的 txt/md/pdf 等文件
def load_documents(knowledge_dir: str = DEFAULT_KNOWLEDGE_BASE) -> List[Document]:
    if not os.path.isdir(knowledge_dir):
        return []

    documents: List[Document] = []
    for file_path in _iter_knowledge_files(knowledge_dir):
        suffix = file_path.suffix.lower()
        try:
            if suffix in TEXT_EXTENSIONS:
                documents.extend(_load_text_file(file_path))
            elif suffix in PDF_EXTENSIONS:
                documents.extend(_load_pdf_file_as_md(file_path))
            else:
                continue
        except Exception as exc:  # noqa: BLE001 - 打印即可
            print(f"[RAG] 无法解析文件 {file_path}: {exc}")
            continue
    return documents

# @lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str) -> Any:
    if CrossEncoder is None:
        raise RuntimeError("sentence-transformers 未安装，无法使用 CrossEncoder reranker。")
    return CrossEncoder(model_name, max_length=512, device=device)


class SimpleCrossEncoderReranker:
    def __init__(self, model_name: str, top_n: int) -> None:
        self._model_name = model_name
        self._top_n = top_n
        self._model = _load_cross_encoder(model_name)

    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        if not documents:
            return []
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda item: item[0], reverse=True)
        return [doc for _, doc in ranked[: self._top_n]]


# class HybridRetriever:
#     """轻量混合检索器，不依赖 langchain.retrievers 子模块。"""

#     def __init__(
#         self,
#         keyword_retriever: Optional[Any],
#         semantic_retriever: Any,
#         *,
#         k: int,
#         bm25_weight: float = BM25_WEIGHT,
#         semantic_weight: float = VECTOR_WEIGHT,
#     reranker: Optional[Any] = None,
#     ) -> None:
#         self._keyword = keyword_retriever
#         self._semantic = semantic_retriever
#         self._k = k
#         self._weights = (bm25_weight, semantic_weight)
#         self._reranker = reranker

#     def _run_retriever(self, retriever: Any, weight: float, query: str) -> List[tuple]:
#         if retriever is None or weight <= 0:
#             return []
#         documents: List[Document] = []
#         if hasattr(retriever, "get_relevant_documents"):
#             documents = retriever.get_relevant_documents(query)
#         elif hasattr(retriever, "invoke"):
#             result = retriever.invoke(query)
#             if isinstance(result, list):
#                 documents = result
#             elif result:
#                 documents = [result]
#         elif callable(retriever):
#             result = retriever(query)
#             if isinstance(result, list):
#                 documents = result
#             elif result:
#                 documents = [result]
#         # scored = []
#         # for idx, doc in enumerate(documents):
#         #     key = (doc.metadata.get("source"), doc.page_content[:200])
#         #     ##  idx + 1 就是文档在单路召回中的排名,权重除以排名，排名越靠前（idx越小），得分越高
#         #     ## 改进了 RRF 算法，引入了可调配的权重因子，使得不同检索器的贡献可以灵活调整，而不仅仅依赖于排名位置
#         #     score = weight / (idx + 1)
#         #     scored.append((key, score, doc))
#         # return scored
#         scored = []
#         # 引入 RRF 平滑常数 k（通常设为 60）
#         k_constant = 60 
        
#         for idx, doc in enumerate(documents):
#             # 改进 1：直接使用 hash(全文) 作为去重 key，避免 PDF 头部重复导致的误删
#             key = hash(doc.page_content)
            
#             # 改进 2：标准的加权 RRF 公式
#             score = weight / (k_constant + idx + 1)
            
#             scored.append((key, score, doc))
            
#         return scored

#     def get_relevant_documents(self, query: str) -> List[Document]:
#         scored_docs: dict = {}
#         # for key, score, doc in self._run_retriever(self._keyword, self._weights[0], query):
#         #     current = scored_docs.get(key)
#         #     if current is None or score > current[0]:
#         #         scored_docs[key] = (score, doc)
#         # for key, score, doc in self._run_retriever(self._semantic, self._weights[1], query):
#         #     current = scored_docs.get(key)
#         #     if current is None or score > current[0]:
#         #         scored_docs[key] = (score, doc)
#         # docs_sorted = [pair[1] for pair in sorted(scored_docs.values(), key=lambda item: item[0], reverse=True)]
#         # 1. 记录 BM25 召回的分数
#         for key, score, doc in self._run_retriever(self._keyword, self._weights[0], query):
#             scored_docs[key] = [score, doc]
            
#         # 2. 记录 向量 召回的分数（如果重合，则分数相加！）
#         for key, score, doc in self._run_retriever(self._semantic, self._weights[1], query):
#             if key in scored_docs:
#                 scored_docs[key][0] += score  # 核心：分数相加，提升双命中文档的权重
#             else:
#                 scored_docs[key] = [score, doc]
                
#         # 3. 排序
#         docs_sorted = [pair[1] for pair in sorted(scored_docs.values(), key=lambda item: item[0], reverse=True)]
#         if self._reranker:
#             try:
#                 docs_sorted = self._reranker.compress_documents(docs_sorted, query)  # type: ignore[arg-type]
#             except Exception as exc:  # noqa: BLE001
#                 print("[RAG] Cross-Encoder rerank 失败，退回混合检索排序：", exc)
#         return docs_sorted[: self._k]

#     def invoke(self, query: str) -> List[Document]:
#         return self.get_relevant_documents(query)

class ParentChildQdrantRetriever:
    """基于 Qdrant 的父子文档混合检索器"""
    def __init__(self, qdrant_vector_store, parent_store_dir: str, k: int, reranker=None):
        self._vector_store = qdrant_vector_store
        self._parent_store_dir = parent_store_dir
        self._k = k
        self._reranker = reranker

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 1. 让 Qdrant 进行底层的 稠密+稀疏 混合检索，召回多一些 Child (例如 2 倍的 k)
        child_docs = self._vector_store.similarity_search(query, k=self._k * 2)
        
        # 2. 提取 Parent ID 并去重 (保持检索的相关性顺序)
        parent_ids = []
        seen = set()
        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                parent_ids.append(pid)
                
        # 3. 从本地读取完整的 Parent Chunks
        parent_docs = []
        for pid in parent_ids[:self._k]: # 只取前 K 个父文档
            filepath = os.path.join(self._parent_store_dir, f"{pid}.json")
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    parent_docs.append(Document(page_content=data["page_content"], metadata=data["metadata"]))
        
        # 4. 可选：用你的 CrossEncoder 对召回的宏大 Parent Chunks 进行重排
        if self._reranker and parent_docs:
            try:
                parent_docs = self._reranker.compress_documents(parent_docs, query)
            except Exception as e:
                print(f"[RAG] 重排失败: {e}")
                
        return parent_docs

    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

def get_knowledge_hash(knowledge_dir: str) -> str:
    """计算目录下所有文件修改时间的哈希值"""
    if not os.path.exists(knowledge_dir):
        return "empty"
    
    mtimes = []
    for root, _, files in os.walk(knowledge_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # 获取文件的最后修改时间戳
            mtimes.append(str(os.path.getmtime(file_path)))
            
    # 把所有时间戳拼起来算个 MD5
    hash_md5 = hashlib.md5("".join(mtimes).encode('utf-8'))
    return hash_md5.hexdigest()

# def build_retriever(
#     knowledge_dir: str = DEFAULT_KNOWLEDGE_BASE,
#     k: int = 3,
#     dir_hash: str = "",
# ):
#     """构建并缓存检索器（首次调用会进行向量化与索引构建）。

#     参数：
#     - knowledge_dir: 知识库目录路径。
#     - k: 每次检索返回的文档片段数量。
#     返回：
#     - langchain 的检索器对象（vectorstore.as_retriever），或 None 当目录为空。
#     """
#     docs = load_documents(knowledge_dir)
#     if not docs:
#         return None
#     # 将长文按字符递归分块，兼顾语义完整性与检索粒度
#     splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
#     chunks = splitter.split_documents(docs)
#     embeddings = build_embeddings()

#     # ---- 语义检索：FAISS ----
#     vectorstore = FAISS.from_documents(chunks, embeddings)
#     semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_TOP_K})

#     # ---- 关键词检索：BM25 ----
#     keyword_retriever: Optional[Any] = None
#     if _BM25_AVAILABLE:
#         try:
#             keyword_retriever = BM25Retriever.from_documents(chunks)
#             keyword_retriever.k = BM25_TOP_K
#         except Exception as exc:  # noqa: BLE001
#             print(
#                 "[RAG] BM25Retriever 构建失败，已降级为纯向量检索。"
#                 "请确认已安装 rank_bm25 依赖。",
#                 exc,
#             )

#     # ---- 混合检索 ----
#     reranker: Optional[Any] = None
#     if _CROSS_ENCODER_AVAILABLE:
#         try:
#             reranker = SimpleCrossEncoderReranker(
#                 model_name=DEFAULT_RERANK_MODEL,
#                 top_n=max(k, RERANK_TOP_N),
#             )
#         except Exception as exc:  # noqa: BLE001
#             print(
#                 "[RAG] 重排序模型加载失败，已回退至混合检索排序。"
#                 "请确认已安装 sentence-transformers 并可访问 BAAI/bge-reranker-base。",
#                 exc,
#             )
#     else:
#         print(
#             "[RAG] sentence-transformers 未安装，无法启用交叉编码重排序，将直接返回混合检索结果。"
#         )

#     return HybridRetriever(
#         keyword_retriever=keyword_retriever,
#         semantic_retriever=semantic_retriever,
#         k=k,
#         reranker=reranker,
#     )
# ==========================================
# 新增：全局线程锁，防止多智能体并发初始化导致的 Qdrant 锁冲突
# ==========================================
import threading
_retriever_init_lock = threading.Lock()

@lru_cache(maxsize=1)
def _build_retriever_internal(knowledge_dir: str = DEFAULT_KNOWLEDGE_BASE, k: int = 3, dir_hash: str = ""):
    """内部实际执行构建的函数（原先的 build_retriever）"""
    docs = load_documents(knowledge_dir)
    if not docs:
        return None

    os.makedirs(PARENT_STORE_DIR, exist_ok=True)
    
    # --- 1. 定义分块器 ---
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # --- 2. 父子拆分与存储准备 ---
    all_child_chunks = []
    
    # 清空旧的 parent 缓存 (已修复多线程冲突)
    for f in os.listdir(PARENT_STORE_DIR):
        file_path = os.path.join(PARENT_STORE_DIR, f)
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[RAG] 无法删除缓存文件 {file_path}: {e}")

    for doc in docs:
        source_name = Path(doc.metadata.get("source", "unknown")).stem
        parent_chunks = parent_splitter.split_text(doc.page_content)
        
        for i, p_chunk in enumerate(parent_chunks):
            parent_id = f"{source_name}_p{i}"
            p_chunk.metadata["parent_id"] = parent_id
            p_chunk.metadata["source"] = source_name
            
            with open(os.path.join(PARENT_STORE_DIR, f"{parent_id}.json"), "w", encoding="utf-8") as f:
                json.dump({"page_content": p_chunk.page_content, "metadata": p_chunk.metadata}, f, ensure_ascii=False)
            
            children = child_splitter.split_documents([p_chunk])
            all_child_chunks.extend(children)

    # --- 3. 初始化 Qdrant 混合检索 ---
    dense_embeddings = build_embeddings()
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25") 
    
    client = QdrantClient(path=QDRANT_PATH)
    collection_name = "change_detection_docs"
    
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
        sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
    )

    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse"
    )
    
    qdrant_store.add_documents(all_child_chunks)

    reranker = None
    if _CROSS_ENCODER_AVAILABLE:
        reranker = SimpleCrossEncoderReranker(model_name=DEFAULT_RERANK_MODEL, top_n=max(k, RERANK_TOP_N))

    return ParentChildQdrantRetriever(
        qdrant_vector_store=qdrant_store,
        parent_store_dir=PARENT_STORE_DIR,
        k=k,
        reranker=reranker
    )

def build_retriever(knowledge_dir: str = DEFAULT_KNOWLEDGE_BASE, k: int = 3, dir_hash: str = ""):
    """
    对外暴露的线程安全检索器构建入口。
    利用互斥锁确保无论 Agent 有多少个并发工具调用，底层只初始化一次。
    """
    with _retriever_init_lock:
        return _build_retriever_internal(knowledge_dir, k, dir_hash)


def format_documents(docs: List[Document]) -> str:
    if not docs:
        return ""
    formatted_lines = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted_lines.append(f"[Doc {idx} | {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(formatted_lines)
