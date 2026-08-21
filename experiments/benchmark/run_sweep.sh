#!/usr/bin/env bash
# Unified benchmark sweep driver.
#
#   ./run_sweep.sh <env> "<methods>" "<seeds>" [PARALLEL]
#
# Runs train.py for every (method x seed) at the env's registry-default timesteps,
# writing committable run records to results/<env>/<method>/seed<n>/.
#
# The 12 pretrained methods (<enc>_[align_]{frozen,finetune} for enc in st|ds|pn) each need
# a checkpoint for that (env, encoder, alignment arm). They are looked up by convention
# under $PRETRAIN_ROOT and SKIPPED when absent, so a partially-pretrained env just yields
# a ragged results matrix rather than an error -- e.g. car_flag's belief has only three
# reachable states, so it is never pretrained and those cells stay deliberately blank.
#
# Examples:
#   # Car-Flag: the analytic + from-scratch methods, full 10 seeds
#   ./run_sweep.sh car_flag "gaussian kmoments cgf deepset pointnet st_scratch" "0 1 2 3 4 5 6 7 8 9"
#   # Odd-Even: everything (pretrained cells appear once their checkpoints exist)
#   ./run_sweep.sh odd_even "$(python -c 'from set_transformer.rl.benchmark.registry import METHOD_ORDER; print(" ".join(METHOD_ORDER))')" "0 1"
set -u
cd "$(dirname "$0")/../.." || exit 1   # -> set_transformer/

ENV="${1:?usage: run_sweep.sh <env> \"<methods>\" \"<seeds>\" [PARALLEL]}"
METHODS="${2:?need methods}"
SEEDS="${3:?need seeds}"
PARALLEL="${4:-5}"

# Checkpoints written by experiments/benchmark/pretrain/3_pretrain_encoder.py:
#   $PRETRAIN_ROOT/<env>/<encoder>_<arm>/checkpoint_best.pt   arm in {plain, align}
PRETRAIN_ROOT="${PRETRAIN_ROOT:-experiments/benchmark/pretrained}"

# NOTE: there is deliberately no fallback to the old ant_tag v8 ST checkpoint
# (experiments/ant_tag_st_v8/chamfer_2026-08-12_19-12-43). It was pretrained at
# dim_hidden=128, whereas the capacity-matched ST encoder is dim_hidden=64, so its
# state_dict no longer loads. Every pretrained encoder must come from the pretrain/
# pipeline at the matched arch; a missing checkpoint skips the cell rather than
# silently mixing capacities.

# checkpoint_for <env> <method> -> path on stdout, empty if this method needs none or
# none exists yet.
checkpoint_for() {
  local env="$1" method="$2" enc arm
  case "$method" in
    st_*frozen|st_*finetune) enc="st" ;;
    ds_*frozen|ds_*finetune) enc="ds" ;;
    pn_*frozen|pn_*finetune) enc="pn" ;;
    *) return 0 ;;                      # analytic or from-scratch: no checkpoint
  esac
  [[ "$method" == *_align_* ]] && arm="align" || arm="plain"
  local path="$PRETRAIN_ROOT/$env/${enc}_${arm}/checkpoint_best.pt"
  [[ -f "$path" ]] && echo "$path"
  return 0
}

export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1
LOGDIR="logs/${ENV}_sweep"
mkdir -p "$LOGDIR"

run_one() {
  local env="$1" method="$2" seed="$3" ckpt="$4"
  local ckpt_arg=""
  if [[ "$method" == *_frozen || "$method" == *_finetune ]]; then
    if [[ -z "$ckpt" ]]; then
      echo "[skip] $env/$method seed$seed: no pretrained encoder for this (env, arm)"; return
    fi
    ckpt_arg="--pretrained_model_path $ckpt"
  fi
  local log="logs/${env}_sweep/${method}_seed${seed}.log"
  echo "[start] $env/$method seed$seed -> $log"
  conda run -n SetTransformer python experiments/benchmark/train.py \
    --env "$env" --method "$method" --seed "$seed" $ckpt_arg \
    > "$log" 2>&1
  echo "[done rc=$?] $env/$method seed$seed"
}
export -f run_one

for s in $SEEDS; do
  for m in $METHODS; do
    printf '%s\t%s\t%s\t%s\n' "$ENV" "$m" "$s" "$(checkpoint_for "$ENV" "$m")"
  done
done | xargs -P "$PARALLEL" -d '\n' -L 1 bash -c 'IFS=$'"'"'\t'"'"' read -r e m s c <<< "$1"; run_one "$e" "$m" "$s" "$c"' _
