# set_transformer

PyTorch implementation of the
[Set Transformer](http://proceedings.mlr.press/v97/lee19d.html)
(Lee et al., 2019), plus an optional RL/particle-filter subpackage that
uses Set Transformer encoders to compress particle-based POMDP belief
representations into fixed-size features for downstream policy learning.

The core library can be used standalone (no gym / SB3 dependencies). The
RL infrastructure is pulled in via the `[rl]` install extra.

## Installation

```bash
# core ST library only (no gym / SB3 / filterpy)
pip install -e .

# with RL / particle-filter infra
pip install -e .[rl]

# with dev tools (pytest / flake8 / black)
pip install -e .[rl,dev]
```

Python >= 3.11.

## Repository layout

```
set_transformer/
├── set_transformer/                  # the only Python package
│   ├── modules.py                    # MAB / SAB / ISAB / PMA / PFDecoder
│   ├── models/                       # model classes (extensible — drop new
│   │   ├── set_transformer.py        #   variants like vae.py / vqvae.py here)
│   │   ├── deep_set.py
│   │   └── pf_set_transformer.py
│   ├── loss.py                       # Chamfer / Sinkhorn / EMD / Hausdorff
│   ├── plots.py
│   ├── data/                         # POMDPDataset + raw_to_numpy
│   ├── training/                     # config, trainer, main entry point
│   └── rl/                           # OPTIONAL — needs the [rl] extra
│       ├── train.py                  #   the ONE RL trainer (PPO/SAC over a Domain x Encoder pair)
│       ├── eval_true_reward.py       #   the ONE evaluation script (+ JSON summary)
│       ├── domains/                  #   one module per problem: registry, env factory,
│       │   ├── base.py               #     curriculum schedules, flags, eval protocol -> Domain record
│       │   ├── ant_tag.py            #     ANT_TAG
│       │   └── odd_even.py           #     ODD_EVEN
│       ├── encoders.py               #   Encoder table: cgf, st, gaussian, deepset, pointnet, kmoments
│       ├── curriculum.py             #   generic schedules (interpolate -> apply to env)
│       ├── run_records.py            #   output root / run dirs / run_config.json / run_status.json
│       ├── pretrained_encoder.py     #   reload-after-PPO + verification
│       ├── encoder_finetune.py       #   encoder LR scaling / unfreeze
│       ├── particle_filters/         #   per-domain PF implementations
│       ├── wrappers/                 #   gym observation wrappers
│       ├── feature_extractors/       #   SB3 BaseFeaturesExtractor adapters
│       └── benchmark/                #   Brendan's encoder benchmark (separate harness)
│
├── experiments/                      # per-domain pipeline scripts (entry points of rl/)
│   ├── ant_tag/                      #   numbered to match the 4-step pipeline
│   └── odd_even/
│
└── tests/                            # pytest suite (core library, rl/, experiment scripts)
```

## Set Transformer architecture

`set_transformer.modules` contains the building blocks:
- **MAB** — Multihead Attention Block
- **SAB** — Set Attention Block
- **ISAB** — Induced Set Attention Block (inducing points → O(n) complexity)
- **PMA** — Pooling by Multihead Attention
- **PFDecoder** — particle-set decoder used by `PFSetTransformer`

`set_transformer.models` exposes the full models:
- `SetTransformer` — the architecture from Lee et al.
- `DeepSet` — Zaheer et al. baseline
- `PFSetTransformer` — encoder/decoder autoencoder for particle reconstruction

```python
from set_transformer.models import SetTransformer, DeepSet, PFSetTransformer
from set_transformer.loss import ChamferDistanceLoss, SinkhornLoss
```

## ST autoencoder pretraining (model-agnostic)

The pretraining command is `python3 -m set_transformer.rl.pretrain --domain <d> --variant <v>
--encoder st|cgf [--objective reconstruction | belief_kl ...]` (2026-09-13; `set_transformer/rl/pretrain.py`);
`experiments/ant_tag/3_train_st.py` is its entry point with the objective fixed to
`reconstruction`, and `experiments/odd_even/3_pretrain_st_belief.py` the entry point for the
Odd-Even exact-posterior objectives. Reconstruction is env-generic: it reads a `.npz` written by
the collector (particles, PF weights, coordinate frame, git provenance) and never touches an env,
so the same objective pretrains the Odd-Even and Ant-Tag encoders. Set size, coordinate dimension,
weightedness and frame all come from the dataset; a contradicting flag is an error. With
`--variant` given, `--data_path` defaults to the variant's dataset under the run root. Runs land
in `<root>/runs/<domain>/<variant>/pretrain/<encoder>/<objective>/<timestamp>_seed<n>/` with
`checkpoints/`, `run_config.json` and `run_status.json` (the entry points keep their historical
folders); `run_records.latest_pretrain_checkpoint(...)` finds the newest RL-loadable file.

```bash
cd experiments/ant_tag
WANDB_MODE=offline python3 3_train_st.py \
    --data_path data/<domain>_pf_dataset.npz \
    --num_encodings 8 --dim_encoder 8 \
    --sinkhorn_blur 0.02 --seed 0
```

Trainable losses: `sinkhorn` (default; the only one that accepts weighted
measures) and `chamfer` (unweighted sets only). `emd` is the eval metric and
`hausdorff` is broken upstream in geomloss. `--align_lambda` adds the latent
metric-alignment term (see `2b_precompute_emd.py`).

`set_transformer/training/main.py` was retired on 2026-09-06 (on a weighted
dataset it scored the reconstruction against the weighted measure while
feeding the encoder bare coordinates, and it took the set geometry from CLI
defaults) and deleted on 2026-09-12; git history before commit `b6d020b` has it.

## RL pipeline (per domain)

Each domain in `experiments/<domain>/` follows the same numbered pipeline:

1. **(MuJoCo only) Locomotion warm-start** — `1_train_locomotion.py`: a baseline policy
   that can move toward goals, so the data-collection rollouts cover the state space.
2. **Particle-filter dataset collection** — `python3 -m set_transformer.rl.collect --domain <d>
   --variant <v>` (entry point `2_collect_pf_dataset.py --variant <v>`): roll out the SAME
   belief env the RL scripts use, snapshot particles AND PF weights to an `.npz` with metadata
   (env id, filter, variant, coordinate frame, CLI args, command, threads, git provenance) under
   `<root>/runs/<domain>/<variant>/data/`. Step 2b, `python3 -m set_transformer.rl.precompute_emd`
   (entry point `2b_precompute_emd.py`), builds the EMD matrix for the aligned arm beside it.
3. **Encoder pretraining** — `python3 -m set_transformer.rl.pretrain` (see above): `reconstruction`
   (weighted Sinkhorn, ST or the CGF block; env-generic) for every domain, plus the objectives a
   domain declares (Odd-Even: `belief_kl` / `mode_ce` / `state_ce`, supervised on exact posteriors).
4. **RL** — `4_train_rl_<encoder>.py --variant <v>`: PPO with the encoder inside the SB3
   policy as a features extractor over the `{"obs", "particles", "weights"}` observation.

**One harness (2026-09-12).** Every `4_train_rl_<encoder>.py` is a short entry point of
`set_transformer/rl/train.py`, which reads one `Domain` record (`rl/domains/<domain>.py`)
and one `Encoder` record (`rl/encoders.py`) and runs the same loop for every pair:

```bash
# any domain x encoder, no script needed
python3 -m set_transformer.rl.train --domain ant_tag --encoder st --variant smart --seed 0
python3 -m set_transformer.rl.train --list_encoders
python3 -m set_transformer.rl.train --domain odd_even --encoder cgf --list_variants

# start modes (every arm's historical flag spelling still works too)
python3 -m set_transformer.rl.train --domain ant_tag --encoder st --variant smart      # end to end
    --pretrained_path <ckpt> --frozen                                                  # frozen encoder
    --pretrained_path <ckpt> --encoder_lr_scale 0.1 --unfreeze_at 1000000              # finetune
```

Runs go to `<root>/runs/<domain>/<variant>/rl/<encoder>/<timestamp>_seed<seed>[_<tag>]/`
(`models/<encoder>_agent.zip`, `models/vecnormalize.pkl`, `run_config.json`,
`run_status.json`, `logs/`); pretraining to `<variant>/pretrain/<experiment_name>/`, eval
summaries to `<variant>/eval/`. The root is `--output_root` > `$RL_BMDP_RUNS` > the parent
repo's `runs/` when this checkout is a git submodule, else this checkout's `runs/`; never the
current directory (`rl/run_records.py`).

Evaluation is one script as well:

```bash
python3 -m set_transformer.rl.eval_true_reward --domain ant_tag --variant smart \
    --model_path <run>/models/st_agent.zip --vecnormalize_path <run>/models/vecnormalize.pkl \
    --n_episodes 100 --seed 7
```

`experiments/<domain>/eval_scripts/eval_true_reward_*.py` are its entry points (they import
the arm's training script first, so SB3 can unpickle the extractor class a recorded zip
names). The env is the domain's eval env, the cap comes from the gym registration, the
particle count off the checkpoint, `--seed` is re-applied after `PPO.load`, and the report
is printed and written as JSON.

Adding a problem: `rl/domains/<name>.py` ending in `<NAME> = Domain(...)` (see `base.py`)
plus one line in `rl/domains/__init__.py`; nothing in it may refer to another domain.
Adding an encoder: one `Encoder(...)` entry in `rl/encoders.py`.

### Ant-Tag

POMDP environment from
[`pomdp-domains`](https://github.com/brendanjcrowe/pomdp-domains)
(`pdomains-ant-tag-*-v0`; the registry in `rl/domains/ant_tag.py` maps `--variant` to env id,
particle filter, episode cap and default curricula). 31-D obs (qpos 15 + qvel 14 + target xy
2); the agent only sees the target when the ant is within the env's visible radius.

```bash
cd experiments/ant_tag
# 1. Pre-train locomotion policy (dense reward wrapper on AntTag)
python3 1_train_locomotion.py --total_timesteps 1000000
# 2. Collect PF dataset (mix of random + pursuit with the locomotion policy) -> <root>/runs/ant_tag/smart/data/
python3 2_collect_pf_dataset.py --variant smart --locomotion_policy_path models/ant_locomotion_policy.zip
# 3. Pretrain the ST autoencoder (output under <root>/runs/ant_tag/smart/pretrain/; --data_path
#    defaults to the dataset step 2 wrote when --variant is given)
WANDB_MODE=offline python3 3_train_st.py --variant smart --num_encodings 8 --dim_encoder 8
# 4. RL: the ST arm, encoder trained under PPO / frozen pretrained / finetuned pretrained
python3 4_train_rl_st.py --variant smart --seed 0
python3 4_train_rl_st.py --variant smart --pretrained_st_model_path <ckpt> --st_frozen
python3 4_train_rl_st.py --variant smart --pretrained_st_model_path <ckpt> --st_encoder_lr_scale 0.1
# the other arms: 4_train_rl_{cgf,gaussian,deepset,pointnet,kmoments}.py, same flags
# 5. Evaluate on the true sparse tag reward
python3 eval_scripts/eval_true_reward_st.py --variant smart --model_path <run>/models/st_agent.zip \
    --vecnormalize_path <run>/models/vecnormalize.pkl --n_episodes 100
```

`st_pipeline_logs/run_st_pipeline_smart_hard.sh` runs collect -> EMD matrix -> pretraining ->
RL arms -> evals for one variant (environment-variable parametrised; `SUFFIX=_SMOKE` with
`RL_BMDP_RUNS=<scratch>` for a smoke run). `diagnostics/` holds the offline probes and
gates; `4_train_rl_frozen.py` is a forwarding file for the shared wrappers, not an arm;
`4_train_rl_finetune.py` is the pre-registry finetune script, not comparable with the arms.

### Odd-Even BeliefMDP

Discrete POMDP test bed (also from `pomdp-domains`; four gym ids,
`pdomains-odd-even-{10,50,50-long,50-short}-v0`, registry keys `oe10`, `oe50`, `oe50_long`,
`oe50_short`). The pipeline lives in `experiments/odd_even/`: `2_collect_pf_dataset.py`, the
shared `3_train_st.py` above for Sinkhorn pretraining, `3_pretrain_st_belief.py` for
supervised pretraining on exact posteriors, `4_train_rl_{cgf,st,gaussian}.py`, and
`eval_scripts/eval_true_reward_odd_even.py` (transient / steady split against the Bayes
oracle; no "success" on this domain). Everything the scripts share is in
`rl/domains/odd_even.py`. Results and run directories are recorded in the parent repo's
`domain_mds/oddeven.md`.

```bash
cd experiments/odd_even
python3 4_train_rl_st.py --variant oe50_short --total_timesteps 3000000 \
    --target_kl 0.03 --lr_anneal --pretrained_st_model_path <ckpt> --st_frozen
```

## Tests

```bash
pytest tests/
```

The suite covers the core library (modules, models, loss, plots, dataset,
raw_to_numpy), the `rl/` subpackage (trainer, domains, encoders, eval, run
records, curriculum, particle filters, wrappers, extractors) and the experiment
scripts (loaded by path; `tests/test_harness_behaviour_inventory.py` pins the
command-line behaviour of every arm). `tests/tools/rl_parity.py` runs one short
training command on two checkouts and compares the outputs bit for bit. Five
tests of the benchmark harness fail since PR #6 (`test_benchmark_results.py`,
`test_configs_current.py`, `test_probe_env.py`); see the parent repo's
`domain_mds/PITFALLS.md` section 11.

## Reference

```bibtex
@InProceedings{lee2019set,
    title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    pages={3744--3753},
    year={2019}
}
```
