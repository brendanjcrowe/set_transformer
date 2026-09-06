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
│       ├── particle_filters/         #   per-domain PF implementations
│       ├── wrappers/                 #   gym observation wrappers
│       ├── feature_extractors/       #   SB3 BaseFeaturesExtractor adapters
│       └── evaluate.py
│
├── experiments/                      # per-domain pipeline scripts
│   ├── ant_tag/                      #   numbered to match the 4-step pipeline
│   └── odd_even/
│
└── tests/                            # pytest suite for the core library
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

The training entry point is `experiments/ant_tag/3_train_st.py`. Despite its
location it is env-generic: it reads a `.npz` written by a domain's
`2_collect_pf_dataset.py` (particles, PF weights, coordinate frame, git
provenance) and never touches an env, so the same script pretrains the
Odd-Even and Ant-Tag encoders. Set size, coordinate dimension, weightedness
and frame all come from the dataset; a contradicting flag is an error.

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

`set_transformer/training/main.py` is RETIRED (2026-09-06): on a weighted
dataset it scored the reconstruction against the weighted measure while
feeding the encoder bare coordinates, and it took the set geometry from CLI
defaults. Its banner has the details.

## RL pipeline (per domain)

Each domain in `experiments/<domain>/` follows the same 4-step pipeline:

1. **(MuJoCo only) Locomotion warm-start** — train a baseline policy that
   can actually move toward goals so the data-collection rollouts cover
   the state space.
2. **Particle-filter dataset collection** — roll out a mix of random and
   goal-directed policies, snapshot PF particle sets at each step.
3. **Set Transformer pretraining** — train `PFSetTransformer` to reconstruct
   the collected particle sets; the encoder bottleneck is the belief
   feature.
4. **RL training with the pretrained ST** — PPO / SAC over an augmented
   observation `[base_obs ‖ st_features]`. ST weights can be either frozen
   (`*_frozen.py`) or fine-tuned end-to-end (`*_finetune.py`).

### Ant-Tag

POMDP environment from
[`pomdp-domains`](https://github.com/brendanjcrowe/pomdp-domains)
(`pdomains-ant-tag-v0`). 31-D obs (qpos 15 + qvel 14 + target xy 2);
the agent only sees the target when the ant is within `vis_radius`
of it.

```bash
# 1. Pre-train locomotion policy (dense reward wrapper on AntTag)
python experiments/ant_tag/1_train_locomotion.py --total_timesteps 1000000

# 2. Collect PF dataset (mix of random + pursuit with locomotion policy)
python experiments/ant_tag/2_collect_pf_dataset.py \
    --locomotion_policy_path models/ant_locomotion_policy.zip

# 3. Pretrain ST autoencoder
python experiments/ant_tag/3_train_st.py \
    --data_path data/ant_tag_pf_dataset.npy

# 4a. RL with frozen ST features
python experiments/ant_tag/4_train_rl_frozen.py \
    --pretrained_st_model_path models/ant_tag_st_pretrained.pt

# 4b. RL with fine-tunable ST inside the policy
python experiments/ant_tag/4_train_rl_finetune.py \
    --pretrained_st_model_path models/ant_tag_st_pretrained.pt
```

Eval / rendering helpers in the same directory:
- `eval_true_reward.py` — eval on the real sparse tag reward, no shaping.
- `sample_and_render.py` — render trajectory snapshots with PF clouds.
- `visualize_st_reconstruction.py` — original vs reconstructed particle sets.
- `sanity_check_fully_observed.py` — fully-observed baseline.

### Odd-Even BeliefMDP

Discrete POMDP test bed (also from `pomdp-domains`; four gym ids,
`pdomains-odd-even-{10,50,50-long,50-short}-v0`). The pipeline lives in
`experiments/odd_even/`: `2_collect_pf_dataset.py`, the shared `3_train_st.py`
above for Sinkhorn pretraining, `3_pretrain_st_belief.py` for supervised
pretraining on exact posteriors, `4_train_rl_{cgf,st,gaussian}.py`, and
`eval_scripts/eval_true_reward_odd_even.py`. Results and run directories are
recorded in the parent repo's `domain_mds/oddeven.md`.
`experiments/odd_even/train_rl_pretrained.py` is retired (banner in file).

```bash
cd experiments/odd_even
python3 4_train_rl_st.py --variant oe50_short --total_timesteps 3000000 \
    --target_kl 0.03 --lr_anneal --pretrained_st_model_path <ckpt> --st_frozen
```

## Tests

```bash
pytest tests/
```

The suite covers the core library (modules, models, loss, plots,
dataset, raw_to_numpy). The RL subpackage and experiment scripts are
not currently exercised by automated tests.

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
