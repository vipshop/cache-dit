export FLUX_DIR="$HF_MODELS/FLUX.1-dev"
export CLIP_MODEL_DIR="$HF_MODELS/cache-dit-eval/CLIP-ViT-g-14-laion2B-s12B-b42K"
export IMAGEREWARD_MODEL_DIR="$HF_MODELS/cache-dit-eval/ImageReward"

function run_flux_draw_bench() {
  local test_num=200
  local save_dir="./tmp/DrawBench200_DBCache"
  local base_params="--test-num ${test_num} --save-dir ${save_dir} --flops"

  # baseline
  python3 bench.py ${base_params}

  # rdt 0.08
  local rdt=0.08
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt}

  # rdt 0.12
  rdt=0.12
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2

  # rdt 0.16
  rdt=0.16
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2

  # rdt 0.20
  rdt=0.20
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2

  # rdt 0.24
  rdt=0.24
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 6
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2

  # rdt 0.32
  rdt=0.32
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 6
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6
  
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
}


function run_flux_draw_bench_with_taylorseer() {
  local taylorseer_params="--taylorseer --order 1"
  local test_num=200
  local save_dir="./tmp/DrawBench200_DBCache_TaylorSeer"
  local base_params="--test-num ${test_num} --save-dir ${save_dir} --flops"

  # baseline
  python3 bench.py ${base_params}

  # rdt 0.08
  local rdt=0.08
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} ${taylorseer_params}

  # rdt 0.12
  rdt=0.12
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}

  # rdt 0.16
  rdt=0.16
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}

  # rdt 0.20
  rdt=0.20
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}

  # rdt 0.24
  rdt=0.24
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 6 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}

  # rdt 0.32
  rdt=0.32
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 6 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 6 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}

  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py ${base_params} --cache --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
}


bench_type=$1

if [[ "${bench_type}" == "taylorseer" ]]; then
  echo "bench_type: ${bench_type}, DBCache + TaylorSeer"
  run_flux_draw_bench_with_taylorseer
else 
  echo "bench_type: ${bench_type}, DBCache"
  run_flux_draw_bench
fi

# export CUDA_VISIBLE_DEVICES=4 && nohup bash bench.sh default > log/cache_dit_bench_default.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=5 && nohup bash bench.sh taylorseer > log/cache_dit_bench_taylorseer.log 2>&1 &
