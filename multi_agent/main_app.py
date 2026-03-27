"""基于 Gradio 的多智能体 + RAG 流水线 Web 界面"""

import gradio as gr
from typing import List, Tuple

# from multi_agent import MultiAgentOrchestrator
from graph_agent import create_research_graph
from rag_pipeline import format_documents

KNOWLEDGE_DIR = "./knowledge_base"

# 初始化全局的 Orchestrator 实例
# orchestrator = MultiAgentOrchestrator(knowledge_dir=KNOWLEDGE_DIR)
graph = create_research_graph()

def process_query(message, chat_history, image_path, use_rag):
    query = message.strip() if message.strip() else "请分析内容"
    
    # 注意：AgentState 里的字段必须和这里完全对应
    # 如果你的 AgentState 没有 images 字段，请务必在 graph_agent.py 里添加它
    inputs = {
        "question": query,
        "retry_count": 0,
        "context": []  # 初始化 context 列表
    }

    try:
        # 1. 使用 invoke 获取完整状态，避免 stream 模式下漏掉字段
        print("🚀 [System] 开始调用 LangGraph 引擎...")
        final_state = graph.invoke(inputs) 
        
        # 2. 从 final_state 中提取数据
        answer = final_state.get("answer", "⚠️ 模型未返回答案")
        task_list = final_state.get("task_list", [])
        context_list = final_state.get("context", [])

        # 3. 构造展示用的文本
        plan_text = "\n".join([f"步骤 {i+1}: {t.get('type')} - {t.get('query')}" for i, t in enumerate(task_list)])
        rag_text = "\n\n---\n\n".join(context_list) if context_list else "未检索到相关内容。"

        # 4. 关键：构造 Gradio 'messages' 格式的历史记录
        # 如果 chat_history 是 None，初始化为空列表
        if chat_history is None:
            chat_history = []
            
        display_query = query if not image_path else f"🖼️ [图文咨询] {query}"
        
        # 追加用户消息和助手消息
        chat_history.append({"role": "user", "content": display_query})
        chat_history.append({"role": "assistant", "content": answer})

        print("✅ [System] 结果已成功返回 UI")
        # 返回值顺序：对话历史, 清空输入框, 清空图片, 计划文本, RAG 文本
        return chat_history, "", None, plan_text, rag_text

    except Exception as e:
        import traceback
        error_msg = f"推理错误: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        chat_history.append({"role": "assistant", "content": f"发生错误：{str(e)}"})
        return chat_history, "", None, "执行中断", error_msg

# ----------------- 构建 Gradio UI -----------------

with gr.Blocks(title="多智能体 RAG 助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Multi-Agent 小苔藓")
    gr.Markdown("基于 LangChain + 多模态大模型 + RAG 的多智能体助手 · 支持文本和图片输入")

    with gr.Row():
        # 左侧区域：聊天界面与输入 (对应原界面的左半部分)
        with gr.Column(scale=5):
            # 修改 Chatbot 定义，去掉过时的 bubble_full_width，显式设置 type='messages'
            chatbot = gr.Chatbot(label="交互视窗", height=500, type='messages')
            
            with gr.Row():
                with gr.Column(scale=4):
                    msg_input = gr.Textbox(
                        label="输入问题", 
                        placeholder="输入你的问题并按回车，或直接点击发送...", 
                        show_label=False,
                        lines=3
                    )
                with gr.Column(scale=1):
                    # Gradio 原生支持图片预览、拖拽和清除
                    image_input = gr.Image(label="上传图片", type="filepath")
                    
            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空上下文")

        # 右侧区域：调试信息与配置 (对应原界面的右半部分)
        with gr.Column(scale=3):
            use_rag_cb = gr.Checkbox(label="启用知识库检索 (RAG)", value=True)
            plan_output = gr.Textbox(label="调度 + 任务计划", lines=10, interactive=False)
            rag_output = gr.Textbox(label="RAG 检索参考", lines=10, interactive=False)

    # ----------------- 绑定事件 -----------------
    
    # 组合需要传入后端和更新前端的组件列表
    input_components = [msg_input, chatbot, image_input, use_rag_cb]
    output_components = [chatbot, msg_input, image_input, plan_output, rag_output]

    # 点击发送按钮
    submit_btn.click(
        fn=process_query,
        inputs=input_components,
        outputs=output_components
    )
    
    # 在输入框按回车
    msg_input.submit(
        fn=process_query,
        inputs=input_components,
        outputs=output_components
    )
    
    # 清空按钮行为：清空聊天记录和右侧面板
    clear_btn.click(
        fn=lambda: ([], "", None, "", ""),
        inputs=None,
        outputs=output_components
    )

if __name__ == "__main__":
    # server_name="0.0.0.0" 确保它能在无头 Linux 服务器上被外部访问
    # server_port 可以自行修改，默认 7860
    demo.launch(server_name="0.0.0.0", server_port=8888, share=False)