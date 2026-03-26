"""基于 Gradio 的多智能体 + RAG 流水线 Web 界面"""

import gradio as gr
from typing import List, Tuple

from multi_agent import MultiAgentOrchestrator
from rag_pipeline import format_documents

KNOWLEDGE_DIR = "./knowledge_base"

# 初始化全局的 Orchestrator 实例
orchestrator = MultiAgentOrchestrator(knowledge_dir=KNOWLEDGE_DIR)

def process_query(
    message: str, 
    chat_history: List[List[str]], 
    image_path: str, 
    use_rag: bool
) -> Tuple[List[List[str]], str, str, str, str]:
    """
    处理用户输入，调用多智能体编排器，并更新界面状态。
    """
    query = message.strip() if message.strip() else "请描述这张图片"
    images = [image_path] if image_path else None

    # 将 Gradio 的历史记录格式转换为 Orchestrator 需要的格式 [(role, content)]
    formatted_history = []
    for user_msg, bot_msg in chat_history:
        if user_msg:
            formatted_history.append(("user", user_msg))
        if bot_msg:
            formatted_history.append(("assistant", bot_msg))
            
    # 追加当前的输入到发送给 Agent 的历史中
    formatted_history.append(("user", query))

    try:
        # 调用核心推理逻辑
        result = orchestrator.run(
            question=query,
            image_paths=images,
            use_knowledge=use_rag,
            chat_history=formatted_history,
        )

        # 格式化 RAG 检索结果
        rag_text = format_documents(result.supporting_documents)
        if not rag_text:
            rag_text = "未检索到相关文档或未启用 RAG。"

        # 更新 Gradio 对话历史 (如果上传了图片，在聊天框中附带提示)
        display_message = query if not image_path else f"[已上传图片] {query}"
        chat_history.append([display_message, result.answer])

        # 返回值对应 outputs 列表中的组件更新：
        # 对话历史, 清空输入框, 清空图片上传区, 计划文本, RAG 文本
        return chat_history, "", None, result.plan, rag_text

    except Exception as e:
        error_msg = f"推理发生错误: {str(e)}"
        chat_history.append([query, error_msg])
        return chat_history, "", None, "执行失败", ""

# ----------------- 构建 Gradio UI -----------------

with gr.Blocks(title="多智能体 RAG 助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Multi-Agent 小苔藓")
    gr.Markdown("基于 LangChain + 多模态大模型 + RAG 的多智能体助手 · 支持文本和图片输入")

    with gr.Row():
        # 左侧区域：聊天界面与输入 (对应原界面的左半部分)
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(label="交互视窗", height=500, bubble_full_width=False)
            
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)