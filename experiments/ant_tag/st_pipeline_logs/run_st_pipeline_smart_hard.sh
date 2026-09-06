#!/usr/bin/env bash
# ST pipeline for the smart_hard variant, queued to start at START_AT.
#
#   Stage 1  collect PF dataset          (CPU + one GPU for the locomotion policy)
#   Stage 2  EMD matrix (GPU 0)  ||  plain Sinkhorn pretraining (GPU 1)
#   Stage 3  aligned pretraining (GPU 0, needs the matrix)
#   Stage 4  RL: {plain, aligned} x {frozen, finetune} x seeds {0, 1}
#            = 8 runs, in two waves of 4 (two per GPU), each wave followed by
#            the 5-seed x 100-episode true-reward eval of best + final.
#
# Every PPO/env flag is the cdens_terminal / smart_hard CGF "terminal recipe"
# (arm A in domain_mds/smart_hard_ant_tag.md), so the ST arms compare against
# runs/ant_tag_cgf_smart_hard/20260904_021908_seed0_terminal_recipe_noent.
#
# Launched in tmux:  tmux new-session -d -s st_smart_hard "bash <this> 2>&1 | tee st_pipeline_logs/smart_hard_pipeline.log"
set -euo pipefail
cd /home/himanshu/Documents/Research/rl_for_beliefmdps/set_transformer/experiments/ant_tag
LOGS=st_pipeline_logs
VARIANT=smart_hard
# Every size below is overridable from the environment so the whole script can
# be exercised in miniature (SUFFIX=_SMOKE N_TRAJ=8 ... ) before the real run.
SUFFIX="${SUFFIX:-}"                  # appended to dataset / experiment / run names
START_AT="${START_AT:-09:00}"
N_TRAJ="${N_TRAJ:-400}"; T_STEPS="${T_STEPS:-400}"; MAX_SNAP="${MAX_SNAP:-100000}"
EMD_ROWS="${EMD_ROWS:-20000}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"; ST_EVAL_FREQ="${ST_EVAL_FREQ:-1000}"
ALIGN_WARMUP="${ALIGN_WARMUP:-3}"; ALIGN_RAMP="${ALIGN_RAMP:-5}"
RL_STEPS="${RL_STEPS:-6000000}"; RL_EVAL_FREQ="${RL_EVAL_FREQ:-40000}"; RL_EVAL_EPS="${RL_EVAL_EPS:-30}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"; EVAL_SEEDS="${EVAL_SEEDS:-7 99 2024 31337 5}"
SEEDS="${SEEDS:-0 1}"                 # RL seeds to run, one wave each (e.g. SEEDS=1 to redo a lost wave)
DATA=data/${VARIANT}${SUFFIX}_pf_dataset.npz
EMD=data/${VARIANT}${SUFFIX}_pf_dataset_emd.npy
export WANDB_MODE=offline

stamp() { echo "=== [$(date '+%F %T')] $*"; }

# ---- Wait until START_AT (today; if already past, start now) ---------------
target=$(date -d "$START_AT" +%s); now=$(date +%s)
if (( target > now )); then
  stamp "sleeping $(( (target - now) / 60 )) min until $START_AT"
  sleep $(( target - now ))
fi
stamp "START pipeline for $VARIANT"

# ---- Stage 1: collect -------------------------------------------------------
# visibility_radius_min 1.0 = the RL curriculum's FINAL radius; max 10.0 is
# larger than the 9x9 cage's diagonal, i.e. fully observed. evasion 1.0 =
# final evasion phase. Thresholds are ENV UNITS for THIS arena: a uniform
# belief on the 9x9 cage has per-coordinate std ~2.6, so the cdens default
# diffuse threshold of 4.0 would never fire; 2.5 marks near-uniform beliefs.
# --rebalance_no_upsample: downsample only, no duplicated rows (PITFALLS s.4);
# 100k raw snapshots leave ~55-60k after rebalancing.
if [[ -f "$DATA" ]]; then
  stamp "STAGE 1: $DATA exists, skipping collection"
else
  stamp "STAGE 1: collect"
  CUDA_VISIBLE_DEVICES=1 python3 2_collect_pf_dataset.py \
    --variant $VARIANT \
    --num_trajectories "$N_TRAJ" --timesteps "$T_STEPS" \
    --num_particles 100 \
    --pursuit_fraction 0.5 --fully_observed_fraction 0.2 \
    --visibility_radius_min 1.0 --visibility_radius_max 10.0 \
    --evasion_scale 1.0 --target_speed_scale 0.0 \
    --locomotion_policy_path models/ant_locomotion_policy.zip \
    --locomotion_vecnorm_path models/locomotion_vecnorm.pkl \
    --max_snapshots "$MAX_SNAP" \
    --collapsed_threshold 0.5 --diffuse_threshold 2.5 \
    --rebalance_no_upsample \
    --seed 42 \
    --output_file "$DATA" 2>&1 | grep -vE "it/s\]?$" | tee "$LOGS/${VARIANT}_collect.log"
fi

# ---- Stage 2: EMD matrix (GPU 0) || plain pretraining (GPU 1) --------------
# Same protocol as the cdens_terminal Stage A: first 20k rows, blur 0.01,
# scaling 0.5, 8x8 geometry, 30 epochs, seed 0. Both arms train on the SAME
# 20k rows (--max_samples) so plain vs aligned differs only in the loss.
ST_COMMON=(
  --data_path "$DATA"
  --loss_type sinkhorn --sinkhorn_blur 0.01 --sinkhorn_scaling 0.5
  --num_encodings 8 --dim_encoder 8 --num_inds 32 --dim_hidden 128 --num_heads 4
  --num_epochs "$PRETRAIN_EPOCHS" --batch_size 32 --learning_rate 1e-3
  --max_samples "$EMD_ROWS" --seed 0 --eval_freq "$ST_EVAL_FREQ"
)
have_best() { compgen -G "experiments/$1/*/checkpoints/checkpoint_best.pt" > /dev/null; }
stamp "STAGE 2: EMD matrix (cuda:0) and plain pretraining (cuda:1) in parallel"
if [[ -f "$EMD" ]]; then
  stamp "  $EMD exists, skipping"
else
  CUDA_VISIBLE_DEVICES=0 python3 2b_precompute_emd.py \
    --data_path "$DATA" --sinkhorn_blur 0.01 --sinkhorn_scaling 0.5 \
    --max_samples "$EMD_ROWS" 2>&1 | grep -vE "row block .* eta" > "$LOGS/${VARIANT}_emd.log" &
  EMD_PID=$!
fi
if have_best st_pretrain_${VARIANT}${SUFFIX}_plain; then
  stamp "  plain checkpoint exists, skipping plain pretraining"
else
  CUDA_VISIBLE_DEVICES=1 python3 3_train_st.py "${ST_COMMON[@]}" \
    --experiment_name st_pretrain_${VARIANT}${SUFFIX}_plain \
    2>&1 | grep -vE "it/s\]?$" > "$LOGS/${VARIANT}_pretrain_plain.log" &
  PLAIN_PID=$!
fi
[[ -n "${EMD_PID:-}" ]] && wait "$EMD_PID"
stamp "  EMD matrix done"
[[ -n "${PLAIN_PID:-}" ]] && wait "$PLAIN_PID"
stamp "  plain pretraining done"

# ---- Stage 3: aligned pretraining (GPU 0) -----------------------------------
# lambda 0.2, warmup 3, ramp 5 of 30 epochs (the cdens_terminal schedule).
# The lambda schedule is a per-domain hyperparameter; this is the starting
# point, not a tuned value. 3_train_st.py refuses a matrix whose sidecar
# disagrees on blur / scaling / weights / frame / dataset hash.
stamp "STAGE 3: aligned pretraining (cuda:0)"
if have_best st_pretrain_${VARIANT}${SUFFIX}_align; then
  stamp "  aligned checkpoint exists, skipping aligned pretraining"
else
  CUDA_VISIBLE_DEVICES=0 python3 3_train_st.py "${ST_COMMON[@]}" \
    --emd_matrix_path "$EMD" --align_lambda 0.2 \
    --align_warmup_epochs "$ALIGN_WARMUP" --align_ramp_epochs "$ALIGN_RAMP" \
    --experiment_name st_pretrain_${VARIANT}${SUFFIX}_align \
    2>&1 | grep -vE "it/s\]?$" | tee "$LOGS/${VARIANT}_pretrain_align.log"
fi

latest_best() {  # newest experiment dir under experiments/<name>/ -> its checkpoint_best.pt
  local d; d=$(ls -td "experiments/$1"/*/ | head -1)
  local ck="${d}checkpoints/checkpoint_best.pt"
  [[ -f "$ck" ]] || { echo "missing $ck" >&2; exit 1; }
  realpath "$ck"
}
CK_PLAIN=$(latest_best st_pretrain_${VARIANT}${SUFFIX}_plain)
CK_ALIGN=$(latest_best st_pretrain_${VARIANT}${SUFFIX}_align)
stamp "checkpoints: plain=$CK_PLAIN  align=$CK_ALIGN"

# ---- Stage 4: RL ------------------------------------------------------------
# Identical to the smart_hard CGF arm A (terminal recipe) except the extractor.
RL_COMMON=(
  --variant $VARIANT
  --total_timesteps "$RL_STEPS"
  --n_envs 4 --ppo_n_steps 4096 --batch_size 64 --n_epochs 10
  --learning_rate 3e-4 --lr_anneal --target_kl 0.03
  --num_particles 100
  --curriculum 0:100,0.2:100,0.5:1.0,1:1.0
  --evasion_curriculum 0:0,0.2:0,0.5:1,1:1
  --reward_schedule 0:1:0:0,0.2:1:0:0,0.5:0:0:50,1:0:0:50
  --mask_target_obs --target_speed_scale 0.0
  --eval_freq "$RL_EVAL_FREQ" --save_freq 100000 --n_eval_episodes "$RL_EVAL_EPS"
  --num_encodings 8 --dim_encoder 8 --num_inds 32 --dim_hidden 128 --num_heads 4
  --ln --st_weight_channel
)

rl_run() {  # rl_run <seed> <gpu> <plain|align> <frozen|finetune>
  local seed=$1 gpu=$2 pre=$3 mode=$4
  local ck; [[ $pre == plain ]] && ck=$CK_PLAIN || ck=$CK_ALIGN
  local extra=(); [[ $mode == frozen ]] && extra=(--st_frozen)
  local tag="terminal_recipe_noent_${pre}_${mode}${SUFFIX}"
  local rc=0
  python3 4_train_rl_st.py "${RL_COMMON[@]}" --seed "$seed" --device "cuda:$gpu" \
    --pretrained_st_model_path "$ck" "${extra[@]}" --run_tag "$tag" \
    > "$LOGS/${VARIANT}_rl_${tag}_s${seed}.log" 2>&1 || rc=$?
  stamp "$tag seed $seed exited $rc (log: $LOGS/${VARIANT}_rl_${tag}_s${seed}.log)"
}

eval_run() {  # eval_run <run_dir>  -> 5 eval seeds x 100 episodes, best + final
  local r=$1
  for which in best final; do
    local mp; [[ $which == best ]] && mp="$r/models/best_model/best_model.zip" || mp="$r/models/st_agent.zip"
    [[ -f "$mp" ]] || { echo "no $mp"; continue; }
    for s in $EVAL_SEEDS; do
      python3 eval_scripts/eval_true_reward_st.py --variant $VARIANT \
        --model_path "$mp" --vecnormalize_path "$r/models/vecnormalize.pkl" \
        --n_episodes "$EVAL_EPISODES" --seed "$s" 2>&1 | grep -E "Success rate|Median length" \
        | sed "s|^|$(basename "$r") $which seed$s: |"
    done
  done
}

for seed in $SEEDS; do
  stamp "STAGE 4: RL wave seed=$seed (4 runs, 2 per GPU)"
  before=$(date +%s)
  rl_run $seed 0 plain frozen   &
  sleep 5; rl_run $seed 1 plain finetune &
  sleep 5; rl_run $seed 0 align frozen   &
  sleep 5; rl_run $seed 1 align finetune &
  wait
  stamp "  wave seed=$seed done"
  stamp "  evaluating wave seed=$seed (true sparse reward, 5 seeds x 100 episodes)"
  for r in $(find runs/ant_tag_st_${VARIANT} -maxdepth 1 -mindepth 1 -type d -newermt "@$before" -name "*_seed${seed}_*${SUFFIX}"); do
    eval_run "$r" > "$LOGS/${VARIANT}_eval_$(basename "$r").log" 2>&1 &
  done
  wait
  cat "$LOGS"/${VARIANT}_eval_*_seed${seed}_*.log
done
stamp "PIPELINE DONE"
