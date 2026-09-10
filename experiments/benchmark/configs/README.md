# Benchmark configurations

Generated from the live registry by `experiments/benchmark/generate_configs.py`.
Do not edit by hand — change the registry or `train.py` and regenerate:

```bash
python experiments/benchmark/generate_configs.py          # rewrite
python experiments/benchmark/generate_configs.py --check  # fail if stale
```

## Layout

| Path | Contents |
|---|---|
| `_shared.yaml` | Protocol every run obeys: shared head, seeds, eval, capacity fairness |
| `algorithms/` | PPO and SAC hyperparameters |
| `envs/` | One per registered environment: belief, reward, defaults |
| `methods/` | One per method — the encoder configuration and its parameter counts |
| `pretraining/` | One per encoder family that can be pretrained |

## Methods

**Analytic (2)** — no learned particle-side parameters: `gaussian`, `kmoments`

**Learned (20)** — matched at stat_dim 16: `cgf`, `deepset`, `pointnet`, `st_scratch`, `ds_frozen`, `ds_finetune`, `ds_align_frozen`, `ds_align_finetune`, `pn_frozen`, `pn_finetune`, `pn_align_frozen`, `pn_align_finetune`, `st_frozen`, `st_finetune`, `st_align_frozen`, `st_align_finetune`, `cgf_frozen`, `cgf_finetune`, `cgf_align_frozen`, `cgf_align_finetune`

## Reading a method file

`resolved_kwargs` is what `build_extractor_kwargs` actually passes to the
extractor — registry pins merged over the CLI arch, so it is the effective
configuration rather than the defaults in isolation. `params.trainable_in_rl`
is the head alone for frozen arms and everything otherwise;
`params.encoder_only` is the number the capacity claim rests on, since the
pretrained decoder stays attached for checkpoint round-tripping but never runs
in the policy.

## Conventions worth knowing before a merge

- Training may be shaped; **evaluation is always the true, unshaped reward**.
- Reward wrappers sit below every method, so all methods see one task and the
  `pomdp-domains` submodule stays untouched.
- `train.py` puts no algorithm in the results path. Sweeping two algorithms into
  one `--results_dir` makes the second overwrite the first.
