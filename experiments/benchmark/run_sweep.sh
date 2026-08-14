#!/usr/bin/env bash
# Unified benchmark sweep driver.
#
#   ./run_sweep.sh <env> "<methods>" "<seeds>" [PARALLEL]
#
# Runs train.py for every (method x seed) at the env's registry-default timesteps,
# writing committable run records to results/<env>/<method>/seed<n>/.
# ST-pretrained methods (st_frozen/st_finetune) auto-attach the env's ST checkpoint;
# only ant_tag has a compatible (2-D) checkpoint today, so don't request them elsewhere.
#
# Examples:
#   # Ant-Tag pilot: 2 seeds, all 8 methods
#   ./run_sweep.sh ant_tag "gaussian kmoments cgf deepset pointnet st_scratch st_frozen st_finetune" "0 1"
#   # Car-Flag top-up: the two new pooling baselines, full 10 seeds
#   ./run_sweep.sh car_flag "deepset pointnet" "0 1 2 3 4 5 6 7 8 9"
set -u
cd "$(dirname "$0")/../.." || exit 1   # -> set_transformer/

ENV="${1:?usage: run_sweep.sh <env> \"<methods>\" \"<seeds>\" [PARALLEL]}"
METHODS="${2:?need methods}"
SEEDS="${3:?need seeds}"
PARALLEL="${4:-5}"

# Per-env ST checkpoint for st_frozen/st_finetune (empty => those methods unavailable here).
# ant_tag: v8, pretrained on data/ant_tag_pf_dataset_v8.npy — the first dataset collected
# with the corrected PF (env-derived kwargs + pre-step ant position in predict), so the
# pretraining belief distribution matches what the RL policy sees. Chamfer, not Sinkhorn:
# Sinkhorn (blur=0.5) never learned on ant-tag's 2-D particles.
case "$ENV" in
  ant_tag) ST_CKPT="experiments/ant_tag_st_v8/chamfer_2026-08-12_19-12-43/checkpoints/checkpoint_best.pt" ;;
  *)       ST_CKPT="" ;;
esac

export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1
LOGDIR="logs/${ENV}_sweep"
mkdir -p "$LOGDIR"

run_one() {
  local env="$1" method="$2" seed="$3" ckpt="$4"
  local ckpt_arg=""
  if [[ "$method" == st_frozen || "$method" == st_finetune ]]; then
    if [[ -z "$ckpt" ]]; then
      echo "[skip] $env/$method seed$seed: no ST checkpoint registered for this env"; return
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
    printf '%s\t%s\t%s\t%s\n' "$ENV" "$m" "$s" "$ST_CKPT"
  done
done | xargs -P "$PARALLEL" -d '\n' -L 1 bash -c 'IFS=$'"'"'\t'"'"' read -r e m s c <<< "$1"; run_one "$e" "$m" "$s" "$c"' _
