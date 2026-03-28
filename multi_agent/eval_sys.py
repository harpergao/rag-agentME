import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
# --- Ragas 0.3.2 指标导入 ---
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# --- Ragas 0.3.2 生成器导入 ---
try:
    # 0.3.x 推荐的导入方式
    from ragas.testset import TestsetGenerator
    # 注意：新版本可能不再直接暴露 simple, reasoning 等，
    # 而是集成在 TestsetGenerator 内部或通过不同的方式配置
except ImportError:
    # 兜底旧版本
    from ragas.testset.generator import TestsetGenerator
# --- 内部组件导入 (使用相对导入确保包一致性) ---
try:
    from .rag_pipeline import load_documents, build_embeddings
    from .graph_agent import create_research_graph
    from .langchain_wrappers import LocalQwenChatModel
except ImportError:
    from rag_pipeline import load_documents, build_embeddings
    from graph_agent import create_research_graph
    from langchain_wrappers import LocalQwenChatModel
from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode



class RAGEvaluator:
    def __init__(self, knowledge_dir="./knowledge_base", test_size=5):
        self.knowledge_dir = knowledge_dir
        self.test_size = test_size
        
        # 初始化组件
        # self.llm = LocalQwenChatModel(temperature=0.0, max_new_tokens=2048) 
        
        siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
        # 直接使用标准的 ChatOpenAI 调用硅基流动
        self.llm = ChatOpenAI(
            model="deepseek-ai/DeepSeek-V3.2", # 硅基流动上的模型 ID
            api_key=siliconflow_api_key,
            base_url="https://api.siliconflow.cn/v1", # 硅基流动的 API 地址
            temperature=0.1,
            max_tokens=2048
        )
        print(f"🌟 [System Init] 成功连接并初始化大模型: {self.llm.model_name}")
        
        self.embeddings = build_embeddings()
        self.graph = create_research_graph()
        
        self.documents = load_documents(self.knowledge_dir)
        if not self.documents:
            print(f"⚠️ 警告：知识库 {knowledge_dir} 为空。")

    def generate_test_data(self):
            print(f"🏗️ [Ragas 0.3.2] 正在从本地文档生成 {self.test_size} 组测试题目...")
            
            # 1. 初始化生成器
            generator = TestsetGenerator.from_langchain(
                llm=self.llm, 
                embedding_model=self.embeddings
            )
            
            # 2. 调用生成方法
            testset = generator.generate_with_langchain_docs(
                self.documents, 
                testset_size=self.test_size
            )
            
            # 3. 转换为 DataFrame
            test_df = testset.to_pandas()

            # --- 核心修复：把 Ragas 0.3.2 的新列名改回你的代码认的旧列名 ---
            # 这一步解决了你刚才遇到的 KeyError: 'question'
            test_df.rename(columns={
                'user_input': 'question', 
                'reference': 'ground_truth'
            }, inplace=True)

            # --- 救命逻辑：一旦生成成功，立刻存盘！ ---
            # 这样万一后面的推理或评分环节崩了，你下次运行会自动读取这个文件，不用再等生成。
            test_df.to_csv("test_data_cache.csv", index=False)
            print("💾 结果已缓存至 test_data_cache.csv，下次运行将直接加载。")
            
            return test_df

    # def run_system_inference(self, test_df):
    #     print("🚀 [System] 开始运行推理循环...")
    #     results = []
        
    #     for _, row in test_df.iterrows():
    #         question = row.get('question') or row.get('user_input')
    #         ground_truth = row.get('ground_truth') or row.get('reference')
    #         if not question:
    #             continue
    #         print(f"❓ 处理问题: {question}")
            
    #         inputs = {"question": question, "retry_count": 0, "context": []}
    #         state = self.graph.invoke(inputs)
            
    #         results.append({
    #             "question": question,
    #             "answer": state.get("answer", ""),
    #             "contexts": state.get("context", []),
    #             "ground_truth": ground_truth
    #         })
            
    #     return Dataset.from_list(results)
    def run_system_inference(self, test_df):
        print("🚀 [System] 开始运行推理循环...")
        results = []
        
        for index, row in test_df.iterrows():
            # 兼容读取（无论是你手写的还是自动生成的）
            question = row.get('question') if pd.notna(row.get('question')) else row.get('user_input')
            ground_truth = row.get('ground_truth') if pd.notna(row.get('ground_truth')) else row.get('reference')
            
            if not question:
                continue
            print(f"\n❓ 测试用例 [{index}]: {question}")
            
            inputs = {
                "originalQuery": question,
                "messages": [HumanMessage(content=question)]
            }
            
            # 为每个测试题分配独立的记忆线程，防止串条
            config = {"configurable": {"thread_id": f"eval_test_case_{index}"}}
            
            # 执行图推理
            state = self.graph.invoke(inputs, config=config)
            
            # 提取最终回答
            messages = state.get("messages", [])
            if messages:
                final_answer = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
            else:
                final_answer = "未能生成答案"
                
            # 提取检索上下文
            answers_data = state.get("agent_answers", [])
            if answers_data:
                retrieved_contexts = [ans.get("context", "") for ans in answers_data if ans.get("context")]
            else:
                retrieved_contexts = []
                
            # 如果没查到任何文献，塞入一个提示，防止 Ragas 计算 Precision 时报错
            if not retrieved_contexts:
                retrieved_contexts = ["未检索到任何外部知识。"]

            # 🚨 核心修复点：必须严格使用 Ragas 0.3.x 要求的键名！
            results.append({
                "user_input": question,               # 对应用户的查询
                "response": final_answer,             # 对应 Agent 的最终回答
                "retrieved_contexts": retrieved_contexts, # 对应 Agent 检索到的片段
                "reference": ground_truth             # 对应标准答案
            })
            
        return Dataset.from_list(results)
    
    def evaluate_and_save(self, dataset):
            print("📊 [Ragas] 正在计算评分指标...")
            
            metrics = [
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall()
            ]
            
            # --- 核心修改：配置运行参数 ---
            # timeout: 增加到 300 秒（甚至更长，给本地模型留足推理时间）
            # max_workers: 设为 1，强行让它一个一个算，防止显存爆炸或排队超时
            run_config = RunConfig(
                timeout=600,       # 延长到 10 分钟，本地推理慢是正常的
                max_retries=10,    # 失败后重试次数
                max_workers=1      # 【最关键】限制并发数为 1，适合单显卡环境
            )

            result = evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                run_config=run_config  # 将配置传进去
            )
            
            df_result = result.to_pandas()
            df_result.to_csv("rag_eval_results.csv", index=False)
            print("✅ 评估完成！结果已保存至 rag_eval_results.csv")
            return result

if __name__ == "__main__":
    evaluator = RAGEvaluator(test_size=5)
    
    # 检查是否已有缓存数据，避免重复生成（生成很慢）
    if os.path.exists("test_data_cache.csv"):
        test_df = pd.read_csv("test_data_cache.csv")
        print("📂 加载缓存的测试数据...")
    else:
        test_df = evaluator.generate_test_data()
        test_df.to_csv("test_data_cache.csv", index=False)
    
    eval_dataset = evaluator.run_system_inference(test_df)
    scores = evaluator.evaluate_and_save(eval_dataset)
    
    print("\n--- 最终评估报告 ---")
    print(scores)

# # multi_agent/eval_sys.py

# if __name__ == "__main__":
#     os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#     # 实例化评估器
#     evaluator = RAGEvaluator(test_size=1) 
    
#     # --- 核心修改：手动构造微型测试集，彻底跳过 TestsetGenerator ---
#     print("\n🧪 [快速验证模式] 正在跳过自动生成，使用手动构造的数据进行链路测试...")
    
#     # 模拟 Ragas 生成的数据格式
#     test_data = {
#         "question": ["What is the role of graph convolution in the GFMNet architecture for remote sensing change detection?"],
#         "ground_truth": ["Graph convolution is employed in the GFMNet architecture to extract features from bi-temporal remote sensing images by constructing a graph structure where each image patch is treated as a node and edges connect neighboring patches. This enables the extraction of global representations and contextual information from a non-Euclidean space. Graph convolution aggregates information from each node and its neighbors, capturing and updating each node’s feature representation while enabling effective information exchange across the graph. The process uses max-relative graph convolution to integrate information from neighboring nodes and is combined with feed-forward network (FFN) layers to enhance feature transformation capabilities and mitigate over-smoothing in GNNs."]
#     }
#     test_df = pd.DataFrame(test_data)

#     try:
#         # 1. 验证 Agent 推理链路 (Planner -> Manager -> Responder)
#         # 这步会调用你的本地 Qwen 模型和工具
#         print("🚀 第一步：运行多智能体系统进行推理...")
#         eval_dataset = evaluator.run_system_inference(test_df)
        
#         # 2. 验证 Ragas 评分链路
#         # 这步会验证 Ragas 是否能正确识别你的 LocalQwenTextLLM
#         print("📊 第二步：运行 Ragas 评分指标计算...")
#         scores = evaluator.evaluate_and_save(eval_dataset)
        
#         print("\n🎉 [全链路通畅] 验证成功！")
#         print("综合评分结果如下：")
#         print(scores)
        
#     except Exception as e:
#         import traceback
#         print(f"\n❌ [验证失败] 错误详情：\n{traceback.format_exc()}")