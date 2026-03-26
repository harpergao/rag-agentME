from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import torch
from peft import PeftModel
import os
from qwen_vl_utils import process_vision_info  # 关键新增：Qwen 官方图像处理库

model_id = "./qwen3-vl-4b-instruct"

# 4-bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda:0",  # 核心修复 1：强制单卡加载，完全避开多卡 RoPE 通信 Bug
    dtype=torch.float16,
    trust_remote_code=True
)

lora_checkpoint = "output/Qwen3-VL-4Blora/checkpoint-310"
if os.path.exists(os.path.join(lora_checkpoint, "adapter_config.json")):
    print(f"检测到 LoRA 权重，正在加载: {lora_checkpoint}")
    model = PeftModel.from_pretrained(model, lora_checkpoint)
else:
    print("未检测到 LoRA，使用基础模型进行测试。")

processor = AutoProcessor.from_pretrained(
    model_id,
    trust_remote_code=True,
    padding_side="left"
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "用中文描述这张图片"}
        ],
    }
]

# 核心修复 2：分离文本与图像的预处理逻辑，确保视觉特征正确送入模型
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)

# 将所有输入张量对齐到同一张显卡
inputs = inputs.to("cuda:0")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print("\n======== 模型回答 ========")
print(output_text)