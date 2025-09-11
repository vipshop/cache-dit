import torch
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    AllegroTransformer3DModel,
    AllegroPipeline,
)
from diffusers.utils import export_to_video
from transformers import (
    BitsAndBytesConfig as BitsAndBytesConfig,
    T5EncoderModel,
)

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    "rhymes-ai/Allegro",
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = AllegroTransformer3DModel.from_pretrained(
    "rhymes-ai/Allegro",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = AllegroPipeline.from_pretrained(
    "rhymes-ai/Allegro",
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = (
    "A seaside harbor with bright sunlight and sparkling seawater, with many boats in the water. From an aerial view, "
    "the boats vary in size and color, some moving and some stationary. Fishing boats in the water suggest that this "
    "location might be a popular spot for docking fishing boats."
)
video = pipeline(prompt, guidance_scale=7.5, max_sequence_length=512).frames[0]
export_to_video(video, "harbor.mp4", fps=15)
