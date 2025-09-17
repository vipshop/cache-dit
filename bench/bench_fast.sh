export FLUX_DIR="${FLUX_DIR:-$HF_MODELS/FLUX.1-dev}"
export CLIP_MODEL_DIR="${CLIP_MODEL_DIR:-$HF_MODELS/cache-dit-eval/CLIP-ViT-g-14-laion2B-s12B-b42K}"
export IMAGEREWARD_MODEL_DIR="${IMAGEREWARD_MODEL_DIR:-$HF_MODELS/cache-dit-eval/ImageReward}"

function run_flux_draw_bench_fast() {
  local test_num=200
  local save_dir="./tmp/DrawBench200_DBCache_Fast"
  local base_params="--test-num ${test_num} --save-dir ${save_dir} --flops"

  # rdt 0.32
  rdt=0.32
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  # baseline
  # python3 bench.py ${base_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 8 --max-warmup-steps 4 --rdt ${rdt} --mcc 10
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 8
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 8
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 8
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 8
}


function run_flux_draw_bench_with_taylorseer_fast() {
  local taylorseer_params="--taylorseer --order 1"
  local test_num=200
  local save_dir="./tmp/DrawBench200_DBCache_TaylorSeer_Fast"
  local base_params="--test-num ${test_num} --save-dir ${save_dir} --flops"

  # rdt 0.32
  rdt=0.32
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  # baseline
  # python3 bench.py ${base_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 8 --max-warmup-steps 4 --rdt ${rdt} --mcc 10 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 8 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 8 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 8 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 8 ${taylorseer_params}
}


bench_type=$1

if [[ "${bench_type}" == "taylorseer" ]]; then
  echo "bench_type: ${bench_type}, DBCache + TaylorSeer"
  run_flux_draw_bench_with_taylorseer
else 
  echo "bench_type: ${bench_type}, DBCache"
  run_flux_draw_bench
fi

# export CUDA_VISIBLE_DEVICES=6 && nohup bash bench_fast.sh default > log/cache_dit_bench_fast.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=7 && nohup bash bench_fast.sh taylorseer > log/cache_dit_bench_taylorseer_fast.log 2>&1 &
