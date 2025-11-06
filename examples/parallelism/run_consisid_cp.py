import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import ConsisIDPipeline
from diffusers.pipelines.consisid.consisid_utils import (
    prepare_face_models,
    process_face_embeddings_infer,
)
from diffusers.utils import export_to_video
from utils import (
    cachify,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    strify,
)
import cache_dit


args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

model_id = os.environ.get("CONSISID_DIR", "BestWishYsh/ConsisID-preview")

(
    face_helper_1,
    face_helper_2,
    face_clip_model,
    face_main_model,
    eva_transform_mean,
    eva_transform_std,
) = prepare_face_models(model_id, device="cuda", dtype=torch.bfloat16)
pipe = ConsisIDPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

pipe.enable_model_cpu_offload(device=device)
pipe.vae.enable_tiling()

# ConsisID works well with long and well-described prompts. Make sure the face in the image is clearly visible (e.g., preferably half-body or full-body).
prompt = "The video captures a boy walking along a city street, filmed in black and white on a classic 35mm camera. His expression is thoughtful, his brow slightly furrowed as if he's lost in contemplation. The film grain adds a textured, timeless quality to the image, evoking a sense of nostalgia. Around him, the cityscape is filled with vintage buildings, cobblestone sidewalks, and softly blurred figures passing by, their outlines faint and indistinct. Streetlights cast a gentle glow, while shadows play across the boy's path, adding depth to the scene. The lighting highlights the boy's subtle smile, hinting at a fleeting moment of curiosity. The overall cinematic atmosphere, complete with classic film still aesthetics and dramatic contrasts, gives the scene an evocative and introspective feel."
# image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/consisid/consisid_input.png?download=true"
image = "../data/consisid_input.png"

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

del face_helper_1
del face_helper_2
del face_clip_model
del face_main_model
del eva_transform_mean
del eva_transform_std
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


def run_pipe(warmup: bool = False):
    video = pipe(
        image=image,
        prompt=prompt,
        num_inference_steps=50 if not warmup else 5,
        guidance_scale=6.0,
        use_dynamic_cfg=False,
        id_vit_hidden=id_vit_hidden,
        id_cond=id_cond,
        kps_cond=face_kps,
        generator=torch.Generator("cpu").manual_seed(42),
    ).frames[0]
    return video


# warmup
_ = run_pipe(warmup=True)

start = time.time()
video = run_pipe()
end = time.time()

if rank == 0:
    cache_dit.summary(pipe, details=True)

    time_cost = end - start
    save_path = f"consisid.{strify(args, pipe)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(video, save_path, fps=8)

maybe_destroy_distributed()
