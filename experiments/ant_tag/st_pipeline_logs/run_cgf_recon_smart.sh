#!/usr/bin/env bash
# Reconstruction-pretrained CGF on `smart` (2026-09-11): does pretraining the arm's
# CGF block by weighted Sinkhorn reconstruction help, frozen or finetuned, against the
# same block trained from scratch under PPO (arm 3 `cgf_polar_Kgrad`, 77.3 % on
# 2026-09-10, same recipe, same machine)?
#
#   1. 3_train_st.py --encoder cgf on data/smart_pf_dataset.npz (GPU 0): polar 9.0,
#      spread, K', no norm, no readout -- arm 3's geometry exactly -- with the ST
#      pretraining's loss settings on this variant (sinkhorn blur 0.01, scaling 0.5,
#      30 epochs, batch 32, lr 1e-3, seed 0), full dataset. Exports
#      checkpoints/checkpoint_best_cgf_arm.pt.
#   2. run_cgf_arms_smart.sh (CPU, the July recipe, 6M steps, 3 seeds) with two arms
#      defined here through ARM_DEFS:
#        cgf_recon_frozen   -- --pretrained_cgf_model_path <export> --cgf_frozen
#        cgf_recon_finetune -- --pretrained_cgf_model_path <export>  (t trains at the
#                              shared LR; the CGF arm has no encoder_lr_scale flag)
#      then the driver's 100-episode deterministic eval of every FINAL agent at seed 42.
#
# Launch:  tmux new-session -d -s cgf_recon_smart "bash st_pipeline_logs/run_cgf_recon_smart.sh"
# Record:  domain_mds/smart_ant_tag.md, 2026-09-11 entry.
set -euo pipefail
cd "$(dirname "$0")/.."
# Output root (change 5.2 of the harness centralisation, 2026-09-12): every RL run,
# pretraining run and eval summary lands under $ROOT/ant_tag/<variant>/{rl,pretrain,eval}/;
# the wave logs go to $ROOT/ant_tag/waves/. Default <parent repo>/runs; the RL_BMDP_RUNS
# environment variable overrides it (set it for a smoke run).
ROOT=$(PYTHONPATH=../.. python3 -m set_transformer.rl.run_records)
PRETRAIN=$ROOT/ant_tag/smart/pretrain       # 3_train_st.py's default --base_dir for the smart dataset
LOGS=$ROOT/ant_tag/waves; mkdir -p "$LOGS"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=$LOGS/smart_cgf_recon_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] CGF reconstruction pretraining + RL arms on smart; log $LOG"

DATA=${DATA:-data/smart_pf_dataset.npz}
PRETRAIN_GPU=${PRETRAIN_GPU:-0}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-30}
EXP_NAME=${EXP_NAME:-smart_cgf_recon}
GEOM="--t_param polar --t_bound 9.0 --t_init_mode spread --feature_mode K_grad --feature_norm none --num_cgf_features 64"

# START_AT=2 (or EXPORT=<path>) skips the pretraining and reuses the newest export
# under $PRETRAIN/$EXP_NAME -- used 2026-09-11 after the first launch died between
# the two steps (the export pickled a dataclass; fixed in models/cgf_arm_ae.py).
START_AT=${START_AT:-1}
EXPORT=${EXPORT:-}
if [ "$START_AT" -le 1 ] && [ -z "$EXPORT" ]; then
echo "[$(date)] step 1: pretraining ($DATA, GPU $PRETRAIN_GPU, $PRETRAIN_EPOCHS epochs)"
CUDA_VISIBLE_DEVICES=$PRETRAIN_GPU WANDB_MODE=offline python3 3_train_st.py \
  --data_path "$DATA" --encoder cgf $GEOM \
  --loss_type sinkhorn --sinkhorn_blur 0.01 --sinkhorn_scaling 0.5 \
  --num_epochs "$PRETRAIN_EPOCHS" --batch_size 32 --learning_rate 1e-3 \
  --num_encodings 8 --dim_hidden 128 --seed 0 --eval_freq 1000 --save_freq 5000 \
  --experiment_name "$EXP_NAME" 2>&1 | tee "$LOGS/smart_cgf_recon_pretrain_${STAMP}.log"
else
echo "[$(date)] step 1 skipped (START_AT=$START_AT, EXPORT=${EXPORT:-<newest>})"
fi

[ -n "$EXPORT" ] || EXPORT=$(ls -t "$PRETRAIN/${EXP_NAME}"/*/checkpoints/checkpoint_best_cgf_arm.pt | head -1)
[ -f "$EXPORT" ] || { echo "no export found under $PRETRAIN/${EXP_NAME}"; exit 2; }
EXPORT=$(readlink -f "$EXPORT")
echo "[$(date)] step 1 done; export: $EXPORT"
# the scripts bootstrap sys.path themselves; this snippet must too (the editable
# install is not importable from a bare interpreter on this machine)
python3 - "$EXPORT" <<'PY'
import sys, torch
sys.path.insert(0, "../..")
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("  export config:", {k: c["config"][k] for k in ("t_param","t_bound","t_init_max","feature_mode","feature_norm","arena_scale","weighted_pretraining","objective")})
print("  epoch", c["epoch"], "best val", c["best_val_loss"])
PY

echo "[$(date)] step 2: RL arms (CPU, July recipe, 6M, seeds 0 1 2) via run_cgf_arms_smart.sh"
POLAR="--t_param polar --t_init_mode spread --feature_norm none --feature_mode K_grad"
ARM_DEFS="cgf_recon_frozen=$POLAR --pretrained_cgf_model_path $EXPORT --cgf_frozen;cgf_recon_finetune=$POLAR --pretrained_cgf_model_path $EXPORT" \
ARMS="cgf_recon_frozen cgf_recon_finetune" VARIANT=smart DEVICES=cpu EVAL_SEEDS=42 SEEDS="0 1 2" \
  bash st_pipeline_logs/run_cgf_arms_smart.sh
echo "[$(date)] all done"
