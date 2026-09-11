#!/usr/bin/env bash
# Queue (2026-09-08 20:30): once the 18 fork@3M runs have finished (FORK3M ARMS DONE) and
# the mixed-dataset pretraining has produced both checkpoints (PRETRAIN DONE), run the
# driver's RL stage for the re-pretrained encoders: {plain, align} x {frozen, finetune 0.1x}
# x seeds 0 1 2 = 12 runs at 6M, entropy_flat recipe, 40% curriculum -- the main wave's
# arms 1-4 with the new checkpoints, so every number pairs with an existing one.
# Stages 1-3 are skipped by the driver because their outputs exist.
#   tmux new-session -d -s st_v15_mixed_rl "bash st_pipeline_logs/run_mixed_rl_after_forks_smart_mid_slow_v15.sh 2>&1 | tee st_pipeline_logs/smart_mid_slow_v15_mixed10k10k_rl_pipeline.log"
set -euo pipefail
cd /home/himanshu/Documents/Research/rl_for_beliefmdps/set_transformer/experiments/ant_tag
LOGS=st_pipeline_logs
stamp() { echo "=== [$(date '+%F %T')] $*"; }
stamp "queued: waiting for FORK3M ARMS DONE and PRETRAIN DONE"
until grep -q "FORK3M ARMS DONE" $LOGS/smart_mid_slow_v15_fork3M_pipeline.log 2>/dev/null \
   && grep -q "PRETRAIN DONE" $LOGS/smart_mid_slow_v15_mixed10k10k_pretrain_pipeline.log 2>/dev/null; do sleep 120; done
stamp "both done; START RL on the mixed-dataset encoders"
VARIANT=smart_mid_slow_v15 SUFFIX=_mixed10k10k START_AT=now RECIPE=entropy_flat \
  ARMS="plain:frozen plain:finetune align:frozen align:finetune" SEEDS="0 1 2" WAVES="0,1,2" \
  ENCODER_LR_SCALE=0.1 VIS_CURRICULUM="0:100,0.2:100,0.4:1.5,1:1.5" \
  bash $LOGS/run_st_pipeline_smart_hard.sh
stamp "MIXED RL DONE"
