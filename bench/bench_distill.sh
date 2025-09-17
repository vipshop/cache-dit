export QWEN_IMAGE_DIR="${QWEN_IMAGE_DIR:-$HF_MODELS/Qwen-Image}"
export QWEN_IMAGE_LIGHT_DIR="${QWEN_IMAGE_LIGHT_DIR:-$HF_MODELS/Qwen-Image-Lightning}"
export CLIP_MODEL_DIR="${CLIP_MODEL_DIR:-$HF_MODELS/cache-dit-eval/CLIP-ViT-g-14-laion2B-s12B-b42K}"
export IMAGEREWARD_MODEL_DIR="${IMAGEREWARD_MODEL_DIR:-$HF_MODELS/cache-dit-eval/ImageReward}"

function run_qwen_draw_bench_distill_8_steps() {
  local test_num=200
  local save_dir="./tmp/DrawBench200_DBCache_Distill_8_Steps"
  local base_params="--test-num ${test_num} --save-dir ${save_dir} --flops"

  # steps 8
  local rdt=0.5
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench_distill.py ${base_params} --steps 8
  python3 bench_distill.py ${base_params} --cache --Fn 8 --Bn 8 --max-warmup-steps 4 --rdt ${rdt} --steps 8 --mcc 2
  python3 bench_distill.py ${base_params} --cache --Fn 12 --Bn 12 --max-warmup-steps 4 --rdt ${rdt} --steps 8 --mcc 2 
  python3 bench_distill.py ${base_params} --cache --Fn 16 --Bn 16 --max-warmup-steps 4 --rdt ${rdt} --steps 8 --mcc 2 
  python3 bench_distill.py ${base_params} --cache --Fn 24 --Bn 24 --max-warmup-steps 4 --rdt ${rdt} --steps 8 --mcc 2 
  python3 bench_distill.py ${base_params} --cache --Fn 16 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --steps 8 --mcc 2 
  python3 bench_distill.py ${base_params} --cache --Fn 24 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --steps 8 --mcc 2 
  python3 bench_distill.py ${base_params} --cache --Fn 32 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --steps 8 --mcc 2 
  python3 bench_distill.py ${base_params} --cache --Fn 48 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --steps 8 --mcc 2 
}


function run_qwen_draw_bench_distill_4_steps() {
  local test_num=200
  local save_dir="./tmp/DrawBench200_DBCache_Distill_4_Steps"
  local base_params="--test-num ${test_num} --save-dir ${save_dir} --flops"

  # steps 4
  rdt=0.8
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench_distill.py ${base_params} --steps 4
  python3 bench_distill.py ${base_params} --cache --Fn 8 --Bn 8 --max-warmup-steps 2 --rdt ${rdt} --steps 4 --mcc 1
  python3 bench_distill.py ${base_params} --cache --Fn 12 --Bn 12 --max-warmup-steps 2 --rdt ${rdt} --steps 4 --mcc 1 
  python3 bench_distill.py ${base_params} --cache --Fn 16 --Bn 16 --max-warmup-steps 2 --rdt ${rdt} --steps 4 --mcc 1 
  python3 bench_distill.py ${base_params} --cache --Fn 24 --Bn 24 --max-warmup-steps 2 --rdt ${rdt} --steps 4 --mcc 1 
  python3 bench_distill.py ${base_params} --cache --Fn 16 --Bn 0 --max-warmup-steps 2 --rdt ${rdt} --steps 4 --mcc 1
  python3 bench_distill.py ${base_params} --cache --Fn 24 --Bn 0 --max-warmup-steps 2 --rdt ${rdt} --steps 4 --mcc 1 
  python3 bench_distill.py ${base_params} --cache --Fn 32 --Bn 0 --max-warmup-steps 2 --rdt ${rdt} --steps 4 --mcc 1 
  python3 bench_distill.py ${base_params} --cache --Fn 48 --Bn 0 --max-warmup-steps 2 --rdt ${rdt} --steps 4 --mcc 1 
}

bench_type=$1

if [[ "${bench_type}" == "8_steps" ]]; then
  echo "bench_type: ${bench_type}"
  run_qwen_draw_bench_distill_8_steps
else 
  echo "bench_type: ${bench_type}"
  run_qwen_draw_bench_distill_4_steps
fi

# export CUDA_VISIBLE_DEVICES=6,7 && nohup bash bench_distill.sh 8_steps > log/cache_dit_bench_distill_8_steps.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=2,3 && nohup bash bench_distill.sh 4_steps > log/cache_dit_bench_distill_4_steps.log 2>&1 &
