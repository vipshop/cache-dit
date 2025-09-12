import torch
from diffusers import ConsisIDPipeline
from diffusers.pipelines.consisid.consisid_utils import (
    prepare_face_models,
    process_face_embeddings_infer,
)
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BestWishYsh/ConsisID-preview",
    local_dir="BestWishYsh/ConsisID-preview",
)
(
    face_helper_1,
    face_helper_2,
    face_clip_model,
    face_main_model,
    eva_transform_mean,
    eva_transform_std,
) = prepare_face_models(
    "BestWishYsh/ConsisID-preview", device="cuda", dtype=torch.bfloat16
)
pipe = ConsisIDPipeline.from_pretrained(
    "BestWishYsh/ConsisID-preview", torch_dtype=torch.bfloat16
)
pipe.to("cuda")

# ConsisID works well with long and well-described prompts. Make sure the face in the image is clearly visible (e.g., preferably half-body or full-body).
prompt = "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel."
image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_input.png?download=true"

id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(
    face_helper_1,
    face_clip_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    face_main_model,
    "cuda",
    torch.bfloat16,
    image,
    is_align_face=True,
)

video = pipe(
    image=image,
    prompt=prompt,
    num_inference_steps=50,
    guidance_scale=6.0,
    use_dynamic_cfg=False,
    id_vit_hidden=id_vit_hidden,
    id_cond=id_cond,
    kps_cond=face_kps,
    generator=torch.Generator("cpu").manual_seed(42),
)
export_to_video(video.frames[0], "output.mp4", fps=8)
