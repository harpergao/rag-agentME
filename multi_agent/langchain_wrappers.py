"""面向多智能体的 LangChain 标准 ChatModel 封装（支持 Tool Calling）。"""

import json
import re
import uuid
import copy
from typing import Any, List, Optional, Dict, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool

# 导入你现有的模型客户端
from model_client import LocalMultimodalModel


class LocalQwenChatModel(BaseChatModel):
    """对本地多模态模型进行封装，支持标准的 Chat 接口和 Tool Calling。"""

    max_new_tokens: int = 1024
    temperature: float = 0.2 # 涉及工具调用，温度建议调低
    
    # 内部存储绑定的工具
    _bound_tools: List[BaseTool] = []

    @property
    def _llm_type(self) -> str:
        return "local_qwen_chat_with_tools"

    def bind_tools(self, tools: List[Any], **kwargs) -> "LocalQwenChatModel":
        """
        拦截 LangGraph 传进来的工具列表。
        核心修复：必须返回一个新的实例（深拷贝），防止污染原有的纯文本 LLM
        """
        bound_model = copy.copy(self)
        bound_model._bound_tools = tools
        return bound_model

    def _format_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """将 LangChain 的消息对象转换为 Qwen 认识的纯文本 Prompt。"""
        prompt = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
            elif isinstance(msg, HumanMessage):
                prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
            elif isinstance(msg, AIMessage):
                if msg.content:
                    prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
                elif hasattr(msg, "tool_calls") and msg.tool_calls:
                    prompt += f"<|im_start|>assistant\n[调用工具: {msg.tool_calls[0]['name']}]<|im_end|>\n"
            elif hasattr(msg, "name") and getattr(msg, "name", None): 
                # 兼容 ToolMessage
                prompt += f"<|im_start|>user\n【工具 {msg.name} 检索结果返回】:\n{msg.content}\n请根据上述结果继续回答或停止检索。<|im_end|>\n"
        
        # --- 核心修复：更鲁棒的工具信息提取 ---
        if getattr(self, "_bound_tools", None):
            tool_descs = []
            for tool in self._bound_tools:
                # 兼容标准的 Langchain BaseTool 和 裸 Python 函数
                tool_name = getattr(tool, "name", getattr(tool, "__name__", "unknown_tool"))
                tool_desc = getattr(tool, "description", getattr(tool, "__doc__", "没有工具描述"))
                tool_args = getattr(tool, "args", {}) 
                
                tool_descs.append({
                    "name": tool_name,
                    "description": tool_desc,
                    "args": tool_args 
                })
            
            tool_instruction = (
                "<|im_start|>system\n"
                "你是一个可以使用外部工具的智能助手。你可以使用以下工具：\n"
                f"{json.dumps(tool_descs, ensure_ascii=False, indent=2)}\n"
                "如果你需要使用工具，必须严格且仅输出以下 JSON 格式（不要包含任何其他文字）：\n"
                "```json\n"
                '{"action": "tool_name", "action_input": {"query": "你要搜索的关键词"}}\n'
                "```\n"
                "如果你不需要工具，请直接正常回答。<|im_end|>\n"
            )
            prompt = tool_instruction + prompt

        # 加上最后一轮的启动符
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """核心生成逻辑"""
        
        # 1. 组装 Prompt
        full_prompt = self._format_messages_to_prompt(messages)
        
        # 2. 调用本地模型推理
        model = LocalMultimodalModel.get_shared()
        raw_output = model.generate(
            prompt=full_prompt,
            image_inputs=None,
            max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )

        # 截断 stop words
        if stop:
            for s in stop:
                if s in raw_output:
                    raw_output = raw_output.split(s)[0]
        raw_output = raw_output.strip()

        # 3. 解析模型输出，判断是否是工具调用请求
        tool_calls = []
        content = raw_output

        # 尝试使用正则表达式提取 Markdown 代码块中的 JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_output, re.DOTALL)
        if not json_match:
            # 有时模型可能不加 ```json，直接输出大括号
            json_match = re.search(r'^(\{.*?\})$', raw_output, re.DOTALL)

        if json_match and self._bound_tools:
            try:
                action_data = json.loads(json_match.group(1))
                if "action" in action_data and "action_input" in action_data:
                    # 组装 LangChain 标准的 tool_calls 字典
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}", # 伪造一个唯一的调用ID
                        "name": action_data["action"],
                        "args": action_data["action_input"]
                    })
                    # 如果调用了工具，通常模型此轮的纯文本回复视为空
                    content = "" 
            except json.JSONDecodeError:
                pass # 解析失败退回纯文本模式

        # 4. 返回标准 AIMessage
        message = AIMessage(
            content=content,
            tool_calls=tool_calls # 如果为空列表，LangGraph 会认为不需要调用工具
        )
        
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": "Qwen2-VL-Chat-ToolReady",
            "temperature": self.temperature,
        }