"""面向多智能体的 LangChain 标准 LLM 封装。"""

from __future__ import annotations

from typing import Any, List, Optional, Dict

# 导入 LangChain 的基类
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# 确保导入路径正确
from model_client import LocalMultimodalModel


class LocalQwenTextLLM(LLM):
    """对本地多模态模型进行再封装，使其符合 LangChain 标准接口。"""

    # 注意：在 LangChain 的 LLM 类中，非私有属性必须在类定义中声明
    max_new_tokens: int = 512
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型标识。"""
        return "local_qwen_vl"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        同步调用逻辑。LangChain 会自动将其包装为异步方法 agenerate_prompt。
        """
        model = LocalMultimodalModel.get_shared()
        
        # 获取合并后的参数
        tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temp = kwargs.get("temperature", self.temperature)

        # 调用你原有的模型推理接口
        response = model.generate(
            prompt=prompt,
            image_inputs=None, # 纯文本模式
            max_new_tokens=tokens,
            temperature=temp,
        )
        
        # 如果模型输出了 stop 词，手动截断（可选）
        if stop:
            for s in stop:
                if s in response:
                    response = response.split(s)[0]
                    
        return response

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回用于识别该 LLM 的参数字典。"""
        return {
            "model_name": "Qwen2-VL-Local",
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }