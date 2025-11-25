export HCCL_OP_EXPANSION_MODE="AIV"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD


FLUX_DIR=/home/weights/FLUX.1-dev/ torchrun --nproc_per_node=1 run_flux_cp_npu.py --attn "_native_npu" --height 1024 --width 1024

# WAN_2_2_DIR=/home/weights/Wan2.1-T2V-14B-Diffusers/ torchrun --nproc_per_node=8 run_wan_cp_npu.py --attn "_native_npu" --height 1024 --width 1024 --steps 10 --parallel ulysses --vae-dp