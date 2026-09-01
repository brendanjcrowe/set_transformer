# PF-belief encoder benchmark

Compares learned belief encoders (Set Transformer, CGF, DeepSet, PointNet) against
analytic summaries of the same particle set (Gaussian, k-moments), as feature extractors
for an RL policy on POMDPs — plus, since 2026-08, **latent metric-alignment** arms that
test whether pretraining an encoder so its latent geometry mirrors EMD geometry helps the
downstream policy.

**Read [`ENV_VIABILITY.md`](ENV_VIABILITY.md) first if you are adding an environment.**
Three of four environments examined had a reward under which a cheap statistic was
provably sufficient; that document has the diagnostic and the checklist.

## Method interface

Every method is an SB3 `BaseFeaturesExtractor` over the Dict observation
`{"obs", "particles": (N, d)}` from `PFDictObservationWrapper`, all subclassing
`_BasePFStatExtractor`, so the obs-MLP + concat + projection head is **identical across
methods** and only the particle statistic differs. Policy is always `MultiInputPolicy`.

Extractors use `observations["weights"]` when the wrapper provides it and fall back to
uniform otherwise — so weight support is opt-in per environment without changing any
extractor. (Uniform weights are exactly equivalent to unweighted; proven in
`test_cgf_weighted_matches_unweighted_under_uniform_weights`.)

### Capacity fairness (locked 2026-08-20)

Every *learned* encoder shares a bottleneck of **stat_dim = 16** and lands within 15% on
parameter count. Gaussian and k-moments stay analytic at 0 learned parameters — being
parameter-free sufficient statistics is their purpose, not a confound to correct.

| method | stat_dim | encoder params (d=2) |
|---|---|---|
| gaussian | 5 | 0 |
| kmoments (k=4) | 8 | 0 |
| cgf | 16 | 1,168 |
| deepset / pointnet | 16 | 101,520 |
| st_* | 16 | 111,106 |

Enforced by tests (`test_every_learned_encoder_shares_the_matched_bottleneck`,
`test_learned_encoder_parameter_counts_are_within_tolerance`) so it cannot drift, and
recorded per run in `meta.json` as `extractor_params_encoder` — the encoder-only count,
since the pretrained autoencoders keep an unused decoder attached and it would otherwise
overstate capacity by ~15k.

The knobs: ST reaches parity at `dim_hidden=64`, the pooling encoders at
`dim_hidden=128, dim_encoder=2`. `POOLING_ARCH` pins only `dim_hidden` — the bottleneck
comes from the shared CLI arch so every learned method moves together if it changes.

### CGF

Ported from the collaborator's `WeightedCGFFeaturesExtractor`: `t_init_mode`
(`spread` default), `t_clamp`, `exp_arg_clamp`, `t_frozen`, weighted MGF. Two deliberate
divergences, both to keep the fairness contract:

- **Matched bottleneck.** `num_t=64` CGF evaluations are projected to `stat_dim=16`, so
  sampling resolution and bottleneck width are independent knobs where the original
  conflated them. The full 8-directions × 8-log-spaced-norms `spread` init runs at its
  intended resolution while the policy still sees 16 features.
- **Shared head**, rather than concatenating the CGF raw to the observation.

`particle_scale` is per-env (`EnvSpec.particle_scale`: ant_tag 4.5, odd_even 10.0,
car_flag 1.0) because `exp(t·x)` is scale-sensitive and the original hardcoded Ant-Tag's
arena. `t_value_norms()` reports **post-clamp** norms — what `forward` actually evaluates.

## The 18 methods

`gaussian`, `kmoments`, `cgf` (analytic) · `deepset`, `pointnet`, `st_scratch`
(learned, trained with the policy) · and for each encoder in `{st, ds, pn}` and arm in
`{unaligned, aligned}`, both `frozen` and `finetune`:

```
ds_frozen  ds_finetune  ds_align_frozen  ds_align_finetune
pn_frozen  pn_finetune  pn_align_frozen  pn_align_finetune
st_frozen  st_finetune  st_align_frozen  st_align_finetune
```

`run_sweep.sh` resolves each pretrained method's checkpoint by convention and **skips the
cell when it is absent**, so a ragged results matrix is the normal outcome — see
Car-Flag in `ENV_VIABILITY.md` for a case where blanks are the correct answer.

## Pipeline

```bash
# 0. Gate the env BEFORE sweeping (see ENV_VIABILITY.md)
python experiments/benchmark/probe_env.py --env odd_even --n_episodes 400

# 1-3. Pretrained arms: collect beliefs -> EMD matrix -> 6 encoders (3 x {plain, align})
python experiments/benchmark/pretrain/1_collect_pf_dataset.py --env odd_even --episodes 200
python experiments/benchmark/pretrain/2_precompute_emd.py --env odd_even     # aligned arm only
python experiments/benchmark/pretrain/3_pretrain_encoder.py --env odd_even --all

# 4. Sweep
./experiments/benchmark/run_sweep.sh odd_even "gaussian kmoments cgf ..." "0 1 2 3 4 5 6 7 8 9" 5

# 5. Aggregate + plot (ragged matrices handled)
python experiments/benchmark/aggregate.py results --out_dir results/aggregated
python experiments/benchmark/plot.py results --fig_dir results/figures
```

Collection steps the **same wrapper stack the trainer uses** and records
`obs["particles"]`, so the pretraining belief distribution is the RL-time distribution by
construction — the mismatch that silently staled the old Ant-Tag checkpoint.

### Latent alignment

`--align_lambda 0.2` (warmup 15, ramp 15, cosine) adds the Pearson latent↔EMD term from
`set_transformer.latent_alignment`. On the MoG study this took the Set Transformer's
held-out correlation from 0.674 to 0.993 and more than doubled kNN overlap **at zero
reconstruction cost**; whether that transfers to policy performance is what these arms
test.

One hazard, guarded: model selection is by val EMD, which is blind to alignment. If the
best epoch lands before the λ ramp completes, the "aligned" checkpoint is not actually
aligned — a silent mislabel that would void the comparison it exists to support. The
trainer warns, records `best_epoch`, and reports `selected_val_r` (the correlation at the
*shipped* checkpoint, not wherever training stopped).

## Conventions

- **10 seeds** (0-9) per (env, method); bootstrap 95% CIs.
- Training may be shaped; **evaluation is always on the true, unshaped reward**.
- `--no_shaping` disables an env's potential. On a hidden-state env a true-state potential
  is not predictable from the agent's observation, so it can act as advantage noise rather
  than guidance — Ng et al.'s guarantee is an MDP result.
- Run records: `results/<env>/<method>/seed<n>/{meta.json,evaluations.npz}`. `meta.json`
  records the *resolved* extractor config (`extractor_config`), not just CLI overrides.
- `--init_policy` warm-starts a whole policy (distinct from `--pretrained_model_path`,
  which loads only the particle encoder).

## Gotchas

- `conda run` buffers stdout — read the log files, not the console.
- `set_transformer` is a namespace package and the editable install may point at a stale
  checkout; `train.py` inserts the repo root, other scripts need `PYTHONPATH`.
- SAC leaves `policy.features_extractor` as `None` and builds separate actor/critic
  extractors; use `find_features_extractor`. (This crashed every SAC run *after* training
  was paid for, until 2026-08-24.)
- Test suite: 217 pass, plus 2 pre-existing unrelated failures (`test_hausdorff_loss`,
  `test_visualize_particle_filter_reconstruction`). Run headless.
