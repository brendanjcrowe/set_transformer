#!/usr/bin/env bash
# Second-half reward ablation on smart_mid_slow_v15 (2026-09-08): does the residual
# mean-tracking (distance) term in the last 50% of training cost tag rate?
#
# 18 runs, all FORKED at 3,000,000 steps (progress 0.5, the schedule's switch point)
# from the main wave's checkpoints of end-to-end and plain finetune 0.1x, seeds 0 1 2,
# each trained to 6M under one of three second-half schedules:
#   ctrl       0.15 distance, 2 entropy, 50 tag   (the original entropy_flat second half;
#                                                    forked control = resume-fidelity check)
#   dist0      0    distance, 2 entropy, 50 tag   (mean-tracking removed)
#   dist0ent0  0    distance, 0 entropy, 50 tag   (step cost + tag bonus only)
# The first-half waypoints are irrelevant (the fork starts at 0.5) and kept identical
# so run_config.json reads as a full schedule. Everything else is the wave's RL_COMMON.
# 9 runs per GPU (alternating counter, 18 launches).
#
# 2026-09-09 reuse for the mixed-data encoders: ARMS takes any main-wave run-tag suffix
# (e.g. plain_finetune_mixed10k10k align_finetune_mixed10k10k), every *finetune* arm gets
# the 0.1x encoder LR, TAG_SUFFIX (e.g. _rep2) keeps a re-fork of an already-forked arm
# from overwriting the earlier logs, GPU_START picks the first GPU, DRY_RUN=1 prints the
# sources and exits. The eval stage covers exactly the runs this invocation launched.
#
#   tmux new-session -d -s st_v15_fork "bash st_pipeline_logs/run_fork3M_arms_smart_mid_slow_v15.sh 2>&1 | tee st_pipeline_logs/smart_mid_slow_v15_fork3M_pipeline.log"
set -euo pipefail
cd /home/himanshu/Documents/Research/rl_for_beliefmdps/set_transformer/experiments/ant_tag
LOGS=st_pipeline_logs
VARIANT=smart_mid_slow_v15
RUNS=runs/ant_tag_st_${VARIANT}
FORK_STEP="${FORK_STEP:-3000000}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-none_e2e plain_finetune}"          # main-wave run tags to fork from
SCHEDULES="${SCHEDULES:-ctrl dist0 dist0ent0}"
export WANDB_MODE=offline
stamp() { echo "=== [$(date '+%F %T')] $*"; }

declare -A SCHED=(
  [ctrl]="0:1:2:0,0.2:1:2:0,0.5:0.15:2:50,1:0.15:2:50"
  [dist0]="0:1:2:0,0.2:1:2:0,0.5:0:2:50,1:0:2:50"
  [dist0ent0]="0:1:2:0,0.2:1:2:0,0.5:0:0:50,1:0:0:50"
)

RL_COMMON=(
  --variant $VARIANT
  --total_timesteps 6000000
  --n_envs 4 --ppo_n_steps 4096 --batch_size 64 --n_epochs 10
  --learning_rate 3e-4 --lr_anneal --target_kl 0.03
  --num_particles 100
  --evasion_curriculum 0:0,0.2:0,0.5:1,1:1
  --mask_target_obs --target_speed_scale 0.0
  --eval_freq 40000 --save_freq 100000 --n_eval_episodes 30
  --num_encodings 8 --dim_encoder 8 --num_inds 16 --dim_hidden 64 --num_heads 4
  --ln --st_weight_channel
  --curriculum "0:100,0.2:100,0.4:1.5,1:1.5"
)

source_ckpt() {  # source_ckpt <seed> <arm>  -> the main-wave run's checkpoint zip at FORK_STEP
  local matches
  matches=$(ls -d ${RUNS}/*_seed$1_entropy_flat_$2 2>/dev/null)
  [[ $(echo "$matches" | wc -l) == 1 && -n "$matches" ]] || { echo "expected exactly one source run for seed $1 arm $2, got: $matches" >&2; exit 1; }
  echo "$matches/models/checkpoints/ant_tag_st_${FORK_STEP}_steps.zip"
}

rl_run() {  # rl_run <seed> <gpu> <arm> <sched>
  local seed=$1 gpu=$2 arm=$3 sched=$4
  local ck; ck=$(source_ckpt $seed $arm)
  [[ -f "$ck" ]] || { echo "missing checkpoint $ck" >&2; exit 1; }
  local extra=()
  [[ $arm == *finetune* ]] && extra+=(--st_encoder_lr_scale 0.1)
  local tag="entropy_flat_fork3M_${sched}_${arm}${TAG_SUFFIX:-}"
  local rc=0
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python3 4_train_rl_st.py "${RL_COMMON[@]}" --seed "$seed" --device "cuda:$gpu" \
    --reward_schedule "${SCHED[$sched]}" --resume_from "$ck" "${extra[@]}" --run_tag "$tag" \
    > "$LOGS/${VARIANT}_rl_${tag}_s${seed}.log" 2>&1 || rc=$?
  stamp "$tag seed $seed exited $rc (log: $LOGS/${VARIANT}_rl_${tag}_s${seed}.log)"
}

eval_run() {  # 5 eval seeds x 100 episodes, best + final (same as the driver)
  local r=$1
  for which in best final; do
    local mp; [[ $which == best ]] && mp="$r/models/best_model/best_model.zip" || mp="$r/models/st_agent.zip"
    [[ -f "$mp" ]] || { echo "no $mp"; continue; }
    for s in 7 99 2024 31337 5; do
      python3 eval_scripts/eval_true_reward_st.py --variant $VARIANT \
        --model_path "$mp" --vecnormalize_path "$r/models/vecnormalize.pkl" \
        --n_episodes 100 --seed "$s" 2>&1 | grep -E "Success rate|Median length" \
        | sed "s|^|$(basename "$r") $which seed$s: |"
    done
  done
}

stamp "START fork@${FORK_STEP} arms: schedules [$SCHEDULES] x arms [$ARMS] x seeds [$SEEDS]"
for sched in $SCHEDULES; do echo "  $sched: ${SCHED[$sched]}"; done
for seed in $SEEDS; do for arm in $ARMS; do echo "  source seed $seed $arm: $(source_ckpt $seed $arm)"; done; done
[[ "${DRY_RUN:-0}" == 1 ]] && { stamp "DRY_RUN: exiting before launch"; exit 0; }
before=$(date +%s)
gpu=${GPU_START:-0}
for sched in $SCHEDULES; do
  for seed in $SEEDS; do
    for arm in $ARMS; do
      rl_run $seed $gpu $arm $sched &
      gpu=$((1-gpu)); sleep 5
    done
  done
done
wait
stamp "all fork runs done; evaluating (true sparse reward, 5 seeds x 100 episodes, best + final)"
launched=()
for sched in $SCHEDULES; do for seed in $SEEDS; do for arm in $ARMS; do
  for r in $(find $RUNS -maxdepth 1 -mindepth 1 -type d -newermt "@$before" -name "*_seed${seed}_entropy_flat_fork3M_${sched}_${arm}${TAG_SUFFIX:-}"); do
    launched+=("$r"); eval_run "$r" > "$LOGS/${VARIANT}_eval_$(basename "$r").log" 2>&1 &
  done
done; done; done
wait
for r in "${launched[@]}"; do cat "$LOGS/${VARIANT}_eval_$(basename "$r").log"; done
stamp "FORK3M ARMS DONE"
