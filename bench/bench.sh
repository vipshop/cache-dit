export FLUX_DIR="$HF_MODELS/FLUX.1-dev"
export CLIP_MODEL_DIR="$HF_MODELS/cache-dit-eval/CLIP-ViT-g-14-laion2B-s12B-b42K"
export IMAGEREWARD_MODEL_DIR="$HF_MODELS/cache-dit-eval/ImageReward"

function run_flux_draw_bench() {
  local test_num=200

  # baseline
  python3 bench.py --test-num ${test_num}

  # rdt 0.08
  local rdt=0.08
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt}

  # rdt 0.12
  rdt=0.12
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2

  # rdt 0.16
  rdt=0.16
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2

  # rdt 0.20
  rdt=0.20
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2

  # rdt 0.24
  rdt=0.24
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2

  # rdt 0.32
  rdt=0.32
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2
}


function run_flux_draw_bench_with_taylorseer() {
  local taylorseer_params="--taylorseer --order 1"
  local test_num=200

  # baseline
  python3 bench.py --test-num ${test_num}
  
  # rdt 0.08
  local rdt=0.08
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} ${taylorseer_params}

  # rdt 0.12
  rdt=0.12
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}

  # rdt 0.16
  rdt=0.16
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}

  # rdt 0.20
  rdt=0.20
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}

  # rdt 0.24
  rdt=0.24
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}

  # rdt 0.32
  rdt=0.32
  echo "Running residual diff threshold: ${rdt}, test_num: ${test_num}"
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 4 ${taylorseer_params}

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 3 ${taylorseer_params}

  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
  python3 bench.py --cache --test-num ${test_num} --Fn 1 --Bn 0 --max-warmup-steps 4 --rdt ${rdt} --mcc 2 ${taylorseer_params}
}

run_flux_draw_bench
# run_flux_draw_bench_with_taylorseer
# case: run this bench.sh script with nohup

# nohup bash bench.sh > log/cache_dit_bench.log 2>&1 &
