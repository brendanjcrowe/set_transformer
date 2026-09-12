# to_be_deleted/

Files moved here on 2026-09-11 because nothing runs them: no pipeline script, no test, not
the benchmark harness. They are parked rather than deleted so the move can be reviewed as
one change; delete the folder once that review is done. Nothing imports from this folder
and it is not a Python package.

| file here | was | what it did | replaced by |
|---|---|---|---|
| `feature_extractors_pretrained.py` | `set_transformer/rl/feature_extractors/pretrained.py` | `PretrainedSetTransformerProcessor`: ran a pretrained Set Transformer as a numpy processor *inside the env wrapper*, so the encoder could never be trained or finetuned; unweighted, no arena scale | `rl/feature_extractors/st.py::SetTransformerFeaturesExtractor` with `pretrained_st_model_path` / `st_frozen`, reloaded after PPO construction by `rl/pretrained_encoder.py` |
| `feature_extractors_e2e.py` | `set_transformer/rl/feature_extractors/e2e.py` | `CustomSetTransformerExtractor`: the May-2026 end-to-end ST extractor; unweighted, no scale, no geometry check | the same `SetTransformerFeaturesExtractor` with no pretrained path (PPO trains it); his benchmark's `SetTransformerExtractor` for `st_scratch` |
| `rl_evaluate.py` | `set_transformer/rl/evaluate.py` | eval loop built on the two classes above and on `PFPlusFeaturesObservationWrapper`; zero importers | `experiments/<env>/eval_scripts/eval_true_reward_*.py`; the benchmark evaluates inline in `experiments/benchmark/train.py` |
| `odd_even_train_rl_pretrained.py` | `experiments/odd_even/train_rl_pretrained.py` | retired 2026-09-03 (Gap 11) with a banner; used the processor above with hardcoded Odd-Even assumptions | `experiments/odd_even/4_train_rl_st.py --pretrained_st_model_path` |
| `ant_tag_4_train_rl_frozen_legacy_main.py` | the runnable half of `experiments/ant_tag/4_train_rl_frozen.py` (`AntTagPretrainedProcessor`, `make_ant_tag_pretrained_env`, `train_ant_tag_pretrained`, `__main__`) | the May-2026 Ant-Tag frozen-ST arm: same processor-inside-the-env design as the two files above; the only remaining user of `PFPlusFeaturesObservationWrapper` | `experiments/ant_tag/4_train_rl_st.py --pretrained_st_model_path <ckpt> --st_frozen`. The file itself stays as a LIBRARY of the shared Ant-Tag wrappers (shaping, curriculum, PF glue) that `4_train_rl_cgf.py` imports; its `__main__` now exits with that pointer |
| `training_main.py` | `set_transformer/training/main.py` | retired 2026-09-06 with a banner: on a weighted dataset it scored against the weighted measure while feeding the encoder bare coordinates | `experiments/ant_tag/3_train_st.py` (env-generic) |

Removed in the same change, not parked (they were glue for the files above):

- `PFPlusFeaturesObservationWrapper` in `rl/wrappers/particle_filter.py` -- the wrapper that
  ran the processor inside the env and emitted `obs ++ features` as one flat vector. Every
  live script uses `PFDictWithWeightsObservationWrapper` (`{"obs", "particles", "weights"}`)
  and puts the encoder inside the SB3 policy instead.
- the re-exports of `PretrainedSetTransformerProcessor`, `CustomSetTransformerExtractor` and
  `PFPlusFeaturesObservationWrapper` from the two `__init__.py` files.

Kept, with a status banner: `experiments/ant_tag/4_train_rl_frozen.py`, no runnable arm. On
2026-09-11 it held 561 lines of shared wrappers (was 1,057); on 2026-09-12 those moved into the
package as `set_transformer/rl/domains/ant_tag.py` and the file became a forwarding file that
re-exports them under its historical name (importers use the flat module name).
