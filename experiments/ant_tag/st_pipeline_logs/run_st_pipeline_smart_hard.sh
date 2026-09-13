#!/usr/bin/env bash
# ST pipeline for one Ant-Tag variant (VARIANT, default smart_hard), queued to start at START_AT.
#
#   Stage 1  collect PF dataset          (CPU + one GPU for the locomotion policy)
#   Stage 2  EMD matrix (GPU 0)  ||  plain Sinkhorn pretraining (GPU 1)
#   Stage 3  aligned pretraining (GPU 0, needs the matrix)
#   Stage 4  RL: ARMS x SEEDS, grouped into WAVES (default one wave per seed,
#            WAVES="0,1,2" runs every seed at once). Runs alternate GPUs with
#            a counter that is NOT reset between waves, so the split stays
#            even. Each wave is followed by the 5-seed x 100-episode
#            true-reward eval of best + final.
#
# Every PPO/env flag is the cdens_terminal / smart_hard CGF recipe (RECIPE
# picks the reward schedule; arm A / arm B in domain_mds/smart_hard_ant_tag.md).
#
# Launched in tmux:  tmux new-session -d -s st_smart_hard "bash <this> 2>&1 | tee <ROOT>/ant_tag/waves/smart_hard_pipeline.log"
set -euo pipefail
cd /home/himanshu/Documents/Research/rl_for_beliefmdps/set_transformer/experiments/ant_tag
VARIANT="${VARIANT:-smart_hard}"    # registry key; env id, filter, cap and curriculum come from variants.py
# Output root (change 5.2 of the harness centralisation, 2026-09-12): every RL run,
# pretraining run and eval summary lands under $ROOT/ant_tag/<variant>/{rl,pretrain,eval}/;
# the wave logs go to $ROOT/ant_tag/waves/. Default <parent repo>/runs; the RL_BMDP_RUNS
# environment variable overrides it (set it for a smoke run).
ROOT=$(PYTHONPATH=../.. python3 -m set_transformer.rl.run_records)
RL_RUNS=$ROOT/ant_tag/${VARIANT}/rl/st        # 4_train_rl_st.py's run folders for this variant
PRETRAIN=$ROOT/ant_tag/${VARIANT}/pretrain    # 3_train_st.py's experiment folders (its --base_dir default)
LOGS=$ROOT/ant_tag/waves; mkdir -p "$LOGS"
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
SEEDS="${SEEDS:-0 1}"                 # RL seeds to run (e.g. SEEDS=1 to redo a lost wave)
WAVES="${WAVES:-}"                    # optional grouping: space-separated waves of comma-separated seeds,
                                      # e.g. WAVES="0,1,2" = all three seeds in ONE wave. Default: one wave per seed.
# Visibility curriculum frac:radius. Empty = the registry default for VARIANT
# (reaches the env's own radius at 50%). E.g. "0:100,0.2:100,0.4:1.5,1:1.5"
# reaches the final radius at 40% instead, holding it longer before the end.
VIS_CURRICULUM="${VIS_CURRICULUM:-}"
RL_OMP_THREADS="${RL_OMP_THREADS:-4}"        # OMP/MKL threads per RL process (15 concurrent runs must not each grab 128 cores)
STOP_AFTER_PRETRAIN="${STOP_AFTER_PRETRAIN:-}" # set to exit after stage 3 (pretraining); rerun later for RL
# Reward schedule frac:distance:entropy:tag_bonus. Default = the cdens_terminal
# "arm A" recipe; RECIPE=entropy_flat gives the smart-winning flat PF-entropy
# shaping (0:1:2:0,0.2:1:2:0,0.5:0.15:2:50,1:0.15:2:50).
RECIPE="${RECIPE:-terminal_noent}"
case "$RECIPE" in
  terminal_noent) REWARD_SCHEDULE="0:1:0:0,0.2:1:0:0,0.5:0:0:50,1:0:0:50" ;;
  entropy_flat)   REWARD_SCHEDULE="0:1:2:0,0.2:1:2:0,0.5:0.15:2:50,1:0.15:2:50" ;;
  # entropy_flat with the mean-tracking term at 0 from progress 0.5 -- the second
  # half the 2026-09-08 fork ablation recommends, run UNFORKED (2026-09-10).
  entropy_flat_dist0) REWARD_SCHEDULE="0:1:2:0,0.2:1:2:0,0.5:0:2:50,1:0:2:50" ;;
  *) echo "unknown RECIPE=$RECIPE" >&2; exit 2 ;;
esac
ARMS="${ARMS:-plain:frozen plain:finetune align:frozen align:finetune}"   # <pretrain>:<mode>, pretrain in {plain,align,none}, mode in {frozen,finetune,e2e}
ENCODER_LR_SCALE="${ENCODER_LR_SCALE:-1.0}"   # applied to finetune arms only (--st_encoder_lr_scale)
# ST geometry, shared by pretraining and RL (4_train_rl_st.py refuses a checkpoint
# that disagrees). 8x8 = 64 features, 4 heads, 2 post-PMA SAB blocks throughout.
# Default since 2026-09-07: the SMALL encoder (16 inducing points, hidden 64,
# 109,640 params). The cdens_terminal / smart_hard ST runs used 32 / 128
# (428,168 params); pass NUM_INDS=32 DIM_HIDDEN=128 to reproduce those.
NUM_INDS="${NUM_INDS:-16}"; DIM_HIDDEN="${DIM_HIDDEN:-64}"
# Dataset location (decision 1 of plan section 7, 2026-09-13): new datasets live under the run
# root, $ROOT/ant_tag/<variant>/data/; a recorded dataset still in data/ beside the scripts is
# used when it exists. DATA= overrides both; the EMD matrix always sits beside the dataset.
DATA="${DATA:-}"
if [[ -z "$DATA" ]]; then
  DATA=$ROOT/ant_tag/${VARIANT}/data/${VARIANT}${SUFFIX}_pf_dataset.npz
  [[ -f "data/${VARIANT}${SUFFIX}_pf_dataset.npz" ]] && DATA=data/${VARIANT}${SUFFIX}_pf_dataset.npz
fi
EMD=${DATA%.npz}_emd.npy
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
# visibility_radius_min = the env's own visible radius (= the RL curriculum's
# FINAL radius), read off the registry; max 10.0 is larger than the 9x9 cage's
# diagonal, i.e. fully observed. evasion 1.0 =
# final evasion phase. Thresholds are ENV UNITS for THIS arena: a uniform
# belief on the 9x9 cage has per-coordinate std ~2.6, so the cdens default
# diffuse threshold of 4.0 would never fire; 2.5 marks near-uniform beliefs.
# --rebalance_no_upsample: downsample only, no duplicated rows (PITFALLS s.4);
# 100k raw snapshots leave ~55-60k after rebalancing.
ENV_VIS=$(python3 -c "import gymnasium as gym, variants; e=gym.make(variants.resolve('$VARIANT').env_id, rendering=False); print(e.unwrapped.visible_radius)")
ENV_ID=$(python3 -c "import variants; print(variants.resolve('$VARIANT').env_id)")
stamp "variant $VARIANT -> $ENV_ID, visible_radius $ENV_VIS, recipe $RECIPE ($REWARD_SCHEDULE), arms: $ARMS, seeds: $SEEDS, waves: ${WAVES:-one per seed}, vis curriculum: ${VIS_CURRICULUM:-registry default}, encoder lr scale (finetune): $ENCODER_LR_SCALE, ST geometry 8x8 inds=$NUM_INDS hidden=$DIM_HIDDEN"
if [[ -f "$DATA" ]]; then
  stamp "STAGE 1: $DATA exists, skipping collection"
else
  stamp "STAGE 1: collect"
  CUDA_VISIBLE_DEVICES=1 python3 2_collect_pf_dataset.py \
    --variant $VARIANT \
    --num_trajectories "$N_TRAJ" --timesteps "$T_STEPS" \
    --num_particles 100 \
    --pursuit_fraction 0.5 --fully_observed_fraction 0.2 \
    --visibility_radius_min "$ENV_VIS" --visibility_radius_max 10.0 \
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
  --num_encodings 8 --dim_encoder 8 --num_inds "$NUM_INDS" --dim_hidden "$DIM_HIDDEN" --num_heads 4
  --num_epochs "$PRETRAIN_EPOCHS" --batch_size 32 --learning_rate 1e-3
  --max_samples "$EMD_ROWS" --seed 0 --eval_freq "$ST_EVAL_FREQ"
)
have_best() { compgen -G "$PRETRAIN/$1/*/checkpoints/checkpoint_best.pt" > /dev/null; }
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

latest_best() {  # newest experiment dir under $PRETRAIN/<name>/ -> its checkpoint_best.pt
  local d; d=$(ls -td "$PRETRAIN/$1"/*/ | head -1)
  local ck="${d}checkpoints/checkpoint_best.pt"
  [[ -f "$ck" ]] || { echo "missing $ck" >&2; exit 1; }
  realpath "$ck"
}
CK_PLAIN=$(latest_best st_pretrain_${VARIANT}${SUFFIX}_plain)
CK_ALIGN=$(latest_best st_pretrain_${VARIANT}${SUFFIX}_align)
stamp "checkpoints: plain=$CK_PLAIN align=$CK_ALIGN"
if [[ -n "${STOP_AFTER_PRETRAIN:-}" ]]; then
  # Stages 1-3 only (2026-09-08: pretraining on the mixed dataset while the
  # GPUs are full of RL runs); re-invoke without this to run stage 4, the
  # existing outputs are skipped.
  stamp "STOP_AFTER_PRETRAIN set; not starting RL. PRETRAIN DONE"
  exit 0
fi

# ---- Stage 4: RL ------------------------------------------------------------
# Identical to the CGF arm under the same recipe except the extractor. The
# visibility curriculum is the registry default for the variant (it ends at
# the env's own visible radius) unless VIS_CURRICULUM overrides it.
RL_COMMON=(
  --variant $VARIANT
  --total_timesteps "$RL_STEPS"
  --n_envs 4 --ppo_n_steps 4096 --batch_size 64 --n_epochs 10
  --learning_rate 3e-4 --lr_anneal --target_kl 0.03
  --num_particles 100
  --evasion_curriculum 0:0,0.2:0,0.5:1,1:1
  --reward_schedule "$REWARD_SCHEDULE"
  --mask_target_obs --target_speed_scale 0.0
  --eval_freq "$RL_EVAL_FREQ" --save_freq 100000 --n_eval_episodes "$RL_EVAL_EPS"
  --num_encodings 8 --dim_encoder 8 --num_inds "$NUM_INDS" --dim_hidden "$DIM_HIDDEN" --num_heads 4
  --ln --st_weight_channel
)
[[ -n "$VIS_CURRICULUM" ]] && RL_COMMON+=(--curriculum "$VIS_CURRICULUM")

rl_run() {  # rl_run <seed> <gpu> <plain|align|none> <frozen|finetune|e2e>
  local seed=$1 gpu=$2 pre=$3 mode=$4
  local extra=()
  if [[ $mode == e2e ]]; then
    :  # no checkpoint: encoder trained from scratch under PPO
  else
    local ck; [[ $pre == plain ]] && ck=$CK_PLAIN || ck=$CK_ALIGN
    extra+=(--pretrained_st_model_path "$ck")
    [[ $mode == frozen ]] && extra+=(--st_frozen)
    [[ $mode == finetune && "$ENCODER_LR_SCALE" != "1.0" ]] && extra+=(--st_encoder_lr_scale "$ENCODER_LR_SCALE")
  fi
  local tag="${RECIPE}_${pre}_${mode}${SUFFIX}"
  local rc=0
  OMP_NUM_THREADS=$RL_OMP_THREADS MKL_NUM_THREADS=$RL_OMP_THREADS \
  python3 4_train_rl_st.py "${RL_COMMON[@]}" --seed "$seed" --device "cuda:$gpu" \
    "${extra[@]}" --run_tag "$tag" \
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

[[ -z "$WAVES" ]] && WAVES="${SEEDS// / }"   # default: each seed is its own wave
gpu=0   # global alternation: not reset per wave, so the GPU split stays even
for wave in $WAVES; do
  wave_seeds="${wave//,/ }"
  n_runs=$(( $(wc -w <<< "$wave_seeds") * $(wc -w <<< "$ARMS") ))
  stamp "STAGE 4: RL wave seeds=[$wave_seeds] ($n_runs runs, alternating GPUs from cuda:$gpu)"
  before=$(date +%s)
  for seed in $wave_seeds; do
    for arm in $ARMS; do
      rl_run $seed $gpu "${arm%%:*}" "${arm##*:}" &
      gpu=$((1-gpu)); sleep 5
    done
  done
  wait
  stamp "  wave seeds=[$wave_seeds] done"
  stamp "  evaluating wave seeds=[$wave_seeds] (true sparse reward, 5 seeds x 100 episodes)"
  for seed in $wave_seeds; do
    for r in $(find "$RL_RUNS" -maxdepth 1 -mindepth 1 -type d -newermt "@$before" -name "*_seed${seed}_*${SUFFIX}"); do
      eval_run "$r" > "$LOGS/${VARIANT}_eval_$(basename "$r").log" 2>&1 &
    done
  done
  wait
  for seed in $wave_seeds; do cat "$LOGS"/${VARIANT}_eval_*_seed${seed}_*.log; done
done
stamp "PIPELINE DONE"
