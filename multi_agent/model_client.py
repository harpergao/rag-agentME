"""
本模块封装本地微调的多模态模型（以 Qwen3-VL 为例），统一提供：
- 模型与处理器的单例加载（避免重复占用显存/内存）；
- 可选 4-bit 量化（当有 CUDA 时降低显存占用）；
- 文本 + 可选图片的推理接口；

用法示例：
    from model_client import LocalMultimodalModel
    client = LocalMultimodalModel.get_shared()
    text = client.generate("用中文描述这张图片", image_inputs=["demo.jpg"]) 
"""

import os
import threading
from typing import Iterable, List, Optional

import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info  # 关键新增：引入官方图文处理工具


class LocalMultimodalModel:
    """对本地多模态模型的轻量封装，仅加载一次并提供生成接口。"""

    _instance_lock = threading.Lock()
    _global_instance: Optional["LocalMultimodalModel"] = None

    def __init__(
        self,
        model_root: str = "../qwen3-vl-4b-instruct",
        lora_checkpoint: str = "../output/Qwen3-VL-4Blora/checkpoint-310",
        load_in_4bit: bool = True,
    ) -> None:
        # 保存模型根目录与 LoRA 权重路径（与 test.py 保持同级/同配置）
        self._model_root = model_root
        self._lora_checkpoint = lora_checkpoint
        
        # 核心修改 1：明确指定使用 cuda:0，避免多卡切分导致 2D RoPE 位置编码冲突
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 只有在 CUDA 可用时才启用 4bit 量化，避免 CPU 环境报错
        self._load_in_4bit = load_in_4bit and torch.cuda.is_available()
        # 先构建处理器，再构建模型
        self._processor = self._build_processor()
        self._model = self._build_model()

    @classmethod
    def get_shared(cls) -> "LocalMultimodalModel":
        if cls._global_instance is None:
            with cls._instance_lock:
                if cls._global_instance is None:
                    cls._global_instance = cls()
        return cls._global_instance

    def _build_processor(self) -> AutoProcessor:
        # 加载图文统一处理器，设置左侧 padding 以兼容生成场景
        processor = AutoProcessor.from_pretrained(
            self._model_root,
            trust_remote_code=True,
            padding_side="left",
        )
        return processor

    def _build_model(self) -> AutoModelForImageTextToText:
        quant_config = None
        if self._load_in_4bit:
            # bitsandbytes 的 4bit 配置：NF4、双量化、FP16 计算
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        # 加载基础模型；
        # 核心修改 2：固定 device_map 为 cuda:0，并使用正确的 torch_dtype 参数名
        model = AutoModelForImageTextToText.from_pretrained(
            self._model_root,
            quantization_config=quant_config,
            device_map="cuda:0" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        # 叠加 LoRA 微调权重（PeftModel 封装） - 目前被注释以绕过加载
        # lora_config_path = os.path.join(self._lora_checkpoint, "adapter_config.json")
        # if os.path.exists(lora_config_path):
        #     print(f"检测到 LoRA 权重，正在加载: {self._lora_checkpoint}")
        #     model = PeftModel.from_pretrained(model, self._lora_checkpoint)
        # else:
        #     print(f"未检测到 LoRA 权重或路径错误 ({self._lora_checkpoint})，使用基础模型运行。")
        
        # 核心修改 3：【彻底删除 model = model.to(self._device)】
        # 绝对不能手动 to()，否则会破坏 accelerate 的硬件映射 hook！
        
        model.eval()
        return model

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def model(self) -> AutoModelForImageTextToText:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def generate(
        self,
        prompt: str,
        image_inputs: Optional[Iterable[str]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """执行推理（支持可选的图片输入）。"""

        content: List[dict] = []
        if image_inputs:
            for image_path in image_inputs:
                if not image_path:
                    continue
                if image_path.startswith("http://") or image_path.startswith("https://"):
                    content.append({"type": "image", "image": image_path})
                else:
                    resolved = os.path.abspath(image_path)
                    if not os.path.isfile(resolved):
                        continue
                    # 核心修改 4：直接传递绝对路径，qwen_vl_utils 会更安全地处理底层图像读取
                    content.append({"type": "image", "image": resolved})
                    
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # 核心修改 5：分离图文预处理，保障 Qwen-VL 多模态输入的完整性
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 将所有张量移动到正确的显卡（cuda:0）
        inputs = inputs.to(self._device)

        # 关闭梯度节省显存，进行生成
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        # 仅保留新增的生成部分（去掉输入前缀）
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        # 解码为字符串，跳过特殊符号并保留中文空格
        output = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output.strip()


def describe_image(prompt: str, image_path: str) -> str:
    """便捷函数：给出描述图片的中文回答。"""
    model = LocalMultimodalModel.get_shared()
    return model.generate(prompt=prompt, image_inputs=[image_path])