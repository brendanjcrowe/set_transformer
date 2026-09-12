#!/usr/bin/env bash
# CGF encoder arms on `smart` (pdomains-ant-tag-smart-v0), 2026-09-10.
#
# Four arms x 3 seeds, launched concurrently, GPUs alternated (6 per GPU), then
# a 100-episode true-reward eval of every FINAL agent (eval seed 42,
# deterministic -- the protocol behind every number in smart_ant_tag.md).
# Design and reasoning: change_mds/ant_tag_cgf_port_2026-09-10.md.
#
#   1 cgf_legacy            clamp 2.0, linspace_all_dims (0.1), K       -- the recorded baseline
#   2 cgf_polar_K           polar T=9 (registry), spread 0.25..7.2, K   -- new bound+init, same feature
#   3 cgf_polar_Kgrad       as 2 with K'                                -- feature changed
#   4 cgf_polar_Kgrad_109k  as 3 + readout MLP sized to 109,448 params  -- ST-sized encoder
#
# Common PPO recipe = the July winner in smart_ant_tag.md ("The recipe for
# future runs"); curriculum / evasion / reward come from the registry defaults.
# Usage:  SEEDS="0 1 2" bash st_pipeline_logs/run_cgf_arms_smart.sh
#   VARIANT=smart_mid_slow_v15 DEVICES="cpu" EVAL_SEEDS="7 99 2024 31337 5" \
#     bash st_pipeline_logs/run_cgf_arms_smart.sh        # the v15 ST-table protocol
# VARIANT: registry key (t_bound and schedules come from it). DEVICES: space-
# separated list cycled over the runs ("cuda:0 cuda:1" or "cpu"). EVAL_SEEDS:
# one eval of EVAL_EPISODES episodes per seed on the FINAL agent; the smart
# record uses "42", the v15 ST table used 5 seeds x 100. EXTRA: appended to
# every run (e.g. a --reward_schedule override).
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

SEEDS=${SEEDS:-"0 1 2"}
TOTAL=${TOTAL:-6000000}
VARIANT=${VARIANT:-smart}
DEVICES=${DEVICES:-"cuda:0 cuda:1"}
EVAL_EPISODES=${EVAL_EPISODES:-100}
EVAL_SEEDS=${EVAL_SEEDS:-"42"}
EXTRA=${EXTRA:-}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=st_pipeline_logs/${VARIANT}_cgf_arms_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] launching CGF arms on $VARIANT (devices: $DEVICES; extra: '$EXTRA'); log $LOG"

COMMON="--variant $VARIANT --total_timesteps $TOTAL --n_envs 4 --ppo_n_steps 4096 --batch_size 64 \
  --n_epochs 10 --learning_rate 3e-4 --lr_anneal --target_kl 0.03 \
  --eval_freq 40000 --n_eval_episodes 30 --save_freq 100000 \
  --target_speed_scale 0.0 --num_particles 100 $EXTRA"
POLAR="--t_param polar --t_init_mode spread --feature_norm none"

declare -A ARM_FLAGS=(
  [cgf_legacy]=""
  [cgf_polar_K]="$POLAR --feature_mode K"
  [cgf_polar_Kgrad]="$POLAR --feature_mode K_grad"
  [cgf_polar_Kgrad_109k]="$POLAR --feature_mode K_grad --match_params 109448"
)
# Arms defined at launch time, for flags only known then (e.g. a pretraining
# export path): ARM_DEFS="name1=<flags>;name2=<flags>". Unset -> the four
# recorded arms above, unchanged (2026-09-11).
if [ -n "${ARM_DEFS:-}" ]; then
  IFS=';' read -r -a _arm_defs <<< "$ARM_DEFS"
  for _def in "${_arm_defs[@]}"; do
    [ -n "$_def" ] && ARM_FLAGS[${_def%%=*}]=${_def#*=}
  done
fi
ARMS=${ARMS:-"cgf_legacy cgf_polar_K cgf_polar_Kgrad cgf_polar_Kgrad_109k"}

read -r -a DEVICE_LIST <<< "$DEVICES"
d=0
pids=()
for arm in $ARMS; do
  for seed in $SEEDS; do
    device=${DEVICE_LIST[$d]}
    echo "[$(date)] start $arm seed $seed on $device"
    # shellcheck disable=SC2086
    python3 4_train_rl_cgf.py $COMMON ${ARM_FLAGS[$arm]} --seed "$seed" --device "$device" \
      --run_tag "$arm" > "st_pipeline_logs/${VARIANT}_${arm}_seed${seed}_${STAMP}.launch.log" 2>&1 &
    pids+=($!)
    d=$(( (d + 1) % ${#DEVICE_LIST[@]} ))
    sleep 3
  done
done
echo "[$(date)] ${#pids[@]} runs launched: ${pids[*]}"

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "[$(date)] pid $pid exited non-zero"; fail=1; }
done
echo "[$(date)] all runs finished (fail=$fail); evaluating FINAL agents, $EVAL_EPISODES episodes x eval seeds [$EVAL_SEEDS]"

RUNS_DIR=runs/ant_tag_cgf_${VARIANT}
printf '\n%-24s %-5s %-9s %-10s %s\n' arm seed evalseed success run_dir
for arm in $ARMS; do
  for seed in $SEEDS; do
    dir=$(ls -d ${RUNS_DIR}/*_seed${seed}_${arm}/ 2>/dev/null | sort | tail -1)
    if [ -z "$dir" ] || [ ! -f "$dir/models/cgf_agent.zip" ]; then
      printf '%-24s %-5s %-9s %-10s %s\n' "$arm" "$seed" "-" "MISSING" "${dir:-?}"; continue
    fi
    for es in $EVAL_SEEDS; do
      out=$(python3 eval_scripts/eval_true_reward_cgf.py --variant "$VARIANT" \
        --model_path "$dir/models/cgf_agent.zip" --vecnormalize_path "$dir/models/vecnormalize.pkl" \
        --n_episodes "$EVAL_EPISODES" --seed "$es" 2>&1 | tee "$dir/eval_final_${EVAL_EPISODES}ep_seed${es}.log" \
        | grep "Success rate" | sed 's/.*: //')
      printf '%-24s %-5s %-9s %-10s %s\n' "$arm" "$seed" "$es" "${out:-ERR}" "$dir"
    done
  done
done
echo "[$(date)] done"
