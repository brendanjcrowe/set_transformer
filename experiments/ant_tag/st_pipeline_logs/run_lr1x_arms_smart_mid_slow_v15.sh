#!/usr/bin/env bash
# Two extra arms for the 2026-09-07 smart_mid_slow_v15 ST wave, launched 2026-09-08 00:00
# while the main wave was at 4.2-5.2M: the pretrained encoders (aligned and plain)
# FINETUNED AT THE HEAD'S LEARNING RATE (no --st_encoder_lr_scale), seeds 0 1 2,
# so they pair seed-for-seed with the 0.1x finetune arms. Every other flag is the
# wave's RL_COMMON (run_st_pipeline_smart_hard.sh with VARIANT=smart_mid_slow_v15,
# RECIPE=entropy_flat, VIS_CURRICULUM 40%). Launched directly rather than through
# the driver because the driver script was still executing.
#
#   tmux new-session -d -s st_v15_lr1x "bash st_pipeline_logs/run_lr1x_arms_smart_mid_slow_v15.sh 2>&1 | tee st_pipeline_logs/smart_mid_slow_v15_lr1x_pipeline.log"
set -euo pipefail
cd /home/himanshu/Documents/Research/rl_for_beliefmdps/set_transformer/experiments/ant_tag
LOGS=st_pipeline_logs
VARIANT=smart_mid_slow_v15
CK_PLAIN=/home/himanshu/Documents/Research/rl_for_beliefmdps/set_transformer/experiments/ant_tag/experiments/st_pretrain_smart_mid_slow_v15_plain/sinkhorn_2026-09-07_15-53-26/checkpoints/checkpoint_best.pt
CK_ALIGN=/home/himanshu/Documents/Research/rl_for_beliefmdps/set_transformer/experiments/ant_tag/experiments/st_pretrain_smart_mid_slow_v15_align/sinkhorn_2026-09-07_16-40-10/checkpoints/checkpoint_best.pt
export WANDB_MODE=offline
stamp() { echo "=== [$(date '+%F %T')] $*"; }

RL_COMMON=(
  --variant $VARIANT
  --total_timesteps 6000000
  --n_envs 4 --ppo_n_steps 4096 --batch_size 64 --n_epochs 10
  --learning_rate 3e-4 --lr_anneal --target_kl 0.03
  --num_particles 100
  --evasion_curriculum 0:0,0.2:0,0.5:1,1:1
  --reward_schedule "0:1:2:0,0.2:1:2:0,0.5:0.15:2:50,1:0.15:2:50"
  --mask_target_obs --target_speed_scale 0.0
  --eval_freq 40000 --save_freq 100000 --n_eval_episodes 30
  --num_encodings 8 --dim_encoder 8 --num_inds 16 --dim_hidden 64 --num_heads 4
  --ln --st_weight_channel
  --curriculum "0:100,0.2:100,0.4:1.5,1:1.5"
)

rl_run() {  # rl_run <seed> <gpu> <plain|align>
  local seed=$1 gpu=$2 pre=$3
  local ck; [[ $pre == plain ]] && ck=$CK_PLAIN || ck=$CK_ALIGN
  local tag="entropy_flat_${pre}_finetune_lr1x"
  local rc=0
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python3 4_train_rl_st.py "${RL_COMMON[@]}" --seed "$seed" --device "cuda:$gpu" \
    --pretrained_st_model_path "$ck" --run_tag "$tag" \
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

stamp "START lr1x arms: {align,plain} finetune at the head LR, seeds 0 1 2 (6 runs, 3 per GPU)"
before=$(date +%s)
gpu=0
for seed in 0 1 2; do
  for pre in align plain; do
    rl_run $seed $gpu $pre &
    gpu=$((1-gpu)); sleep 5
  done
done
wait
stamp "all 6 runs done; evaluating (true sparse reward, 5 seeds x 100 episodes, best + final)"
for r in $(find runs/ant_tag_st_${VARIANT} -maxdepth 1 -mindepth 1 -type d -newermt "@$before" -name "*_lr1x"); do
  eval_run "$r" > "$LOGS/${VARIANT}_eval_$(basename "$r").log" 2>&1 &
done
wait
cat "$LOGS"/${VARIANT}_eval_*_lr1x.log
stamp "LR1X ARMS DONE"
