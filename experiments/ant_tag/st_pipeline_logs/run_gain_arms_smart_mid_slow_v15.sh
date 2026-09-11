#!/usr/bin/env bash
# Spread-gain shaping arms for smart_mid_slow_v15 (queued 2026-09-08 ~01:30, start when the
# main wave's driver prints PIPELINE DONE): end-to-end and plain 0.1x finetune, seeds 0 1 2,
# with the entropy_flat recipe PLUS the new spread-gain term at coefficient 10
# (frac:distance:entropy:tag_bonus:spread_gain). Everything else identical to the wave.
# Motivation: domain_mds/smart_mid_slow_v15_ant_tag.md 2026-09-08 -- the weight-entropy term
# is ~constant during a blind search; the plateau at ~40% is search failures; the gain term
# pays for shrinking the belief's spatial spread (sweeping mass-carrying regions).
#
#   tmux new-session -d -s st_v15_gain "bash st_pipeline_logs/run_gain_arms_smart_mid_slow_v15.sh 2>&1 | tee st_pipeline_logs/smart_mid_slow_v15_gain_pipeline.log"
set -euo pipefail
cd /home/himanshu/Documents/Research/rl_for_beliefmdps/set_transformer/experiments/ant_tag
LOGS=st_pipeline_logs
VARIANT=smart_mid_slow_v15
WAIT_FOR="${WAIT_FOR:-$LOGS/smart_mid_slow_v15_pipeline.log}"   # start once this log says PIPELINE DONE
GAIN="${GAIN:-10}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-none:e2e plain:finetune}"
ENCODER_LR_SCALE="${ENCODER_LR_SCALE:-0.1}"
CK_PLAIN=/home/himanshu/Documents/Research/rl_for_beliefmdps/set_transformer/experiments/ant_tag/experiments/st_pretrain_smart_mid_slow_v15_plain/sinkhorn_2026-09-07_15-53-26/checkpoints/checkpoint_best.pt
CK_ALIGN=/home/himanshu/Documents/Research/rl_for_beliefmdps/set_transformer/experiments/ant_tag/experiments/st_pretrain_smart_mid_slow_v15_align/sinkhorn_2026-09-07_16-40-10/checkpoints/checkpoint_best.pt
REWARD_SCHEDULE="0:1:2:0:${GAIN},0.2:1:2:0:${GAIN},0.5:0.15:2:50:${GAIN},1:0.15:2:50:${GAIN}"
export WANDB_MODE=offline
stamp() { echo "=== [$(date '+%F %T')] $*"; }

stamp "queued: waiting for 'PIPELINE DONE' in $WAIT_FOR"
until grep -q 'PIPELINE DONE' "$WAIT_FOR" 2>/dev/null; do sleep 120; done
stamp "main wave finished; START gain arms: $ARMS x seeds [$SEEDS], schedule $REWARD_SCHEDULE"

RL_COMMON=(
  --variant $VARIANT
  --total_timesteps 6000000
  --n_envs 4 --ppo_n_steps 4096 --batch_size 64 --n_epochs 10
  --learning_rate 3e-4 --lr_anneal --target_kl 0.03
  --num_particles 100
  --evasion_curriculum 0:0,0.2:0,0.5:1,1:1
  --reward_schedule "$REWARD_SCHEDULE"
  --mask_target_obs --target_speed_scale 0.0
  --eval_freq 40000 --save_freq 100000 --n_eval_episodes 30
  --num_encodings 8 --dim_encoder 8 --num_inds 16 --dim_hidden 64 --num_heads 4
  --ln --st_weight_channel
  --curriculum "0:100,0.2:100,0.4:1.5,1:1.5"
)

rl_run() {  # rl_run <seed> <gpu> <plain|align|none> <frozen|finetune|e2e>
  local seed=$1 gpu=$2 pre=$3 mode=$4
  local extra=()
  if [[ $mode != e2e ]]; then
    local ck; [[ $pre == plain ]] && ck=$CK_PLAIN || ck=$CK_ALIGN
    extra+=(--pretrained_st_model_path "$ck")
    [[ $mode == frozen ]] && extra+=(--st_frozen)
    [[ $mode == finetune && "$ENCODER_LR_SCALE" != "1.0" ]] && extra+=(--st_encoder_lr_scale "$ENCODER_LR_SCALE")
  fi
  local tag="entropy_flat_gain${GAIN}_${pre}_${mode}"
  local rc=0
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python3 4_train_rl_st.py "${RL_COMMON[@]}" --seed "$seed" --device "cuda:$gpu" \
    "${extra[@]}" --run_tag "$tag" \
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

before=$(date +%s)
gpu=0
for seed in $SEEDS; do
  for arm in $ARMS; do
    rl_run $seed $gpu "${arm%%:*}" "${arm##*:}" &
    gpu=$((1-gpu)); sleep 5
  done
done
wait
stamp "all gain runs done; evaluating (true sparse reward, 5 seeds x 100 episodes, best + final)"
for r in $(find runs/ant_tag_st_${VARIANT} -maxdepth 1 -mindepth 1 -type d -newermt "@$before" -name "*_gain${GAIN}_*"); do
  eval_run "$r" > "$LOGS/${VARIANT}_eval_$(basename "$r").log" 2>&1 &
done
wait
cat "$LOGS"/${VARIANT}_eval_*_gain${GAIN}_*.log
stamp "GAIN ARMS DONE"
