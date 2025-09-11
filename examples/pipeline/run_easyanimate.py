import torch
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    EasyAnimateTransformer3DModel,
    EasyAnimatePipeline,
)
from diffusers.utils import export_to_video

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = EasyAnimateTransformer3DModel.from_pretrained(
    "alibaba-pai/EasyAnimateV5.1-12b-zh",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = EasyAnimatePipeline.from_pretrained(
    "alibaba-pai/EasyAnimateV5.1-12b-zh",
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "A cat walks on the grass, realistic style."
negative_prompt = "bad detailed"
video = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=49,
    num_inference_steps=30,
).frames[0]
export_to_video(video, "cat.mp4", fps=8)
