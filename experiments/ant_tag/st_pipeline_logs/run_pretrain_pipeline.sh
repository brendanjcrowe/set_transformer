#!/usr/bin/env bash
# Pipeline B: collect PF dataset -> pretrain ST -> launch frozen + finetune RL.
# Sequential by necessity (each stage consumes the previous stage's output).
set -euo pipefail
cd /home/himanshu/Documents/Research/rl_for_beliefmdps/set_transformer/experiments/ant_tag
# Output root (change 5.2 of the harness centralisation, 2026-09-12): every RL run,
# pretraining run and eval summary lands under $ROOT/ant_tag/<variant>/{rl,pretrain,eval}/;
# the wave logs go to $ROOT/ant_tag/waves/. Default <parent repo>/runs; the RL_BMDP_RUNS
# environment variable overrides it (set it for a smoke run).
ROOT=$(PYTHONPATH=../.. python3 -m set_transformer.rl.run_records)
PRETRAIN=$ROOT/ant_tag/cdens_terminal/pretrain
LOGS=$ROOT/ant_tag/waves; mkdir -p "$LOGS"

# ---- Stage 1: collect ------------------------------------------------------
# visibility_radius_min 1.0 matches the RL curriculum's FINAL radius, so the
# pretraining beliefs span the same visibility range the agent will see.
# evasion_scale 1.0 matches the final evasion_curriculum phase.
echo "=== [$(date)] STAGE 1: collect ==="
CUDA_VISIBLE_DEVICES=1 python3 2_collect_pf_dataset.py \
  --variant cdens_terminal \
  --num_trajectories 250 --timesteps 300 \
  --num_particles 100 \
  --pursuit_fraction 0.5 --fully_observed_fraction 0.2 \
  --visibility_radius_min 1.0 --visibility_radius_max 15.0 \
  --evasion_scale 1.0 \
  --locomotion_policy_path models/ant_locomotion_policy.zip \
  --locomotion_vecnorm_path models/locomotion_vecnorm.pkl \
  --max_snapshots 40000 \
  --seed 42 \
  --output_file data/cdens_terminal_pf_dataset.npz 2>&1 | tee "$LOGS/collect.log"

# ---- Stage 2: pretrain ----------------------------------------------------
# Geometry MUST match the RL arm: 8 x 8 = 64 features, num_inds 32,
# dim_hidden 128, num_heads 4, LayerNorm on, weight channel on.
# blur 0.01 is in NORMALIZED units (coords / 7.0) = 0.07 env units, well under
# the den radius the encoder must resolve.
echo "=== [$(date)] STAGE 2: pretrain ==="
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python3 3_train_st.py \
  --data_path data/cdens_terminal_pf_dataset.npz \
  --loss_type sinkhorn --sinkhorn_blur 0.01 --sinkhorn_scaling 0.5 \
  --num_encodings 8 --dim_encoder 8 \
  --num_inds 32 --dim_hidden 128 --num_heads 4 \
  --num_epochs 100 --batch_size 32 --learning_rate 1e-3 \
  --experiment_name st_pretrain_cdens_terminal 2>&1 | tee "$LOGS/pretrain.log"

# Newest run dir for this experiment; prefer best, fall back to latest.
RUN_DIR=$(ls -td "$PRETRAIN"/st_pretrain_cdens_terminal/*/ | head -1)
CKPT="${RUN_DIR}checkpoints/checkpoint_best.pt"
[ -f "$CKPT" ] || CKPT="${RUN_DIR}checkpoints/checkpoint_latest.pt"
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint under ${RUN_DIR}checkpoints/"; exit 1; }
CKPT=$(readlink -f "$CKPT")
echo "=== [$(date)] pretrained checkpoint: $CKPT ==="

# ---- Stage 3: two RL runs, identical except --st_frozen -------------------
# Every PPO/env flag below is copied from the CGF and Gaussian
# cdens_terminal run_config.json files.
common=(
  --variant cdens_terminal
  --seed 0
  --total_timesteps 6000000
  --n_envs 4 --ppo_n_steps 4096 --batch_size 64 --n_epochs 10
  --learning_rate 3e-4 --lr_anneal --target_kl 0.03
  --num_particles 100
  --curriculum 0:100,0.2:100,0.5:1.0,1:1.0
  --evasion_curriculum 0:0,0.2:0,0.5:1,1:1
  --reward_schedule 0:1:0:0,0.2:1:0:0,0.5:0:0:50,1:0:0:50
  --distance_coeff 1.0 --entropy_coeff 0.0
  --mask_target_obs --target_speed_scale 0.0
  --eval_freq 40000 --save_freq 100000 --n_eval_episodes 30
  --device cuda:1
  --num_encodings 8 --dim_encoder 8 --num_inds 32 --dim_hidden 128 --num_heads 4
  --ln --st_weight_channel
  --pretrained_st_model_path "$CKPT"
)

setsid nohup python3 4_train_rl_st.py "${common[@]}" --st_frozen \
  --run_tag terminal_v1_dist0_noent_frozen \
  > "$LOGS/rl_frozen.log" 2>&1 &
echo "launched frozen RL: pid $!"

setsid nohup python3 4_train_rl_st.py "${common[@]}" \
  --run_tag terminal_v1_dist0_noent_finetune \
  > "$LOGS/rl_finetune.log" 2>&1 &
echo "launched finetune RL: pid $!"

echo "=== [$(date)] pipeline done; both RL runs detached ==="
