# Synthetic sets — VAE / VQ-VAE experiment results

**Question.** On a mix of 10 synthetic 2D point-set distributions, do a
variational (VAE) or vector-quantized (VQ-VAE) bottleneck improve
reconstruction over a plain Set Transformer autoencoder? And is the
Set Transformer encoder itself buying anything over a mean-pool DeepSet?

**Answer.** No on both counts of the VAE/VQ-VAE question; yes on DeepSet
vs. Set Transformer. Vanilla Set Transformer wins reconstruction on
every family. Attention adds ~25% EMD improvement over mean-pool.
Regularized bottlenecks (Gaussian or discrete) trade fidelity for
representation properties we did not need.

## Setup

- 20,000 train / 2,000 eval point sets, 100 particles each, 2D.
- 10 distribution families sampled with uniform mixture weights:
  `iso_gaussian`, `aniso_gaussian`, `mixture_2g`, `mixture_3g`,
  `uniform_square`, `uniform_disk`, `ring`, `two_moons`, `swiss_roll_2d`,
  `spiral`.
- All models share the same encoder shape (`num_encodings=8`,
  `dim_encoder=16`), same `PFDecoder`, same Chamfer training loss, same
  50 epochs / Adam 1e-3 / cosine schedule.
- Only the bottleneck (and encoder, for the DeepSet arm) differs across
  runs.

## Headline numbers (mean EMD across the eval set)

Vanilla / baseline is the target to beat.

| Model         | Mean EMD | Notes                                       |
| ------------- | -------: | ------------------------------------------- |
| **ST-Vanilla** | **0.200** | best on every family                        |
| ST-VAE (`kl_weight=1e-3`) | 0.247 | KL tax ≈ +23% EMD                  |
| DS-Vanilla    | 0.250 | attention buys ~25% EMD over mean-pool       |
| DS-VAE (`kl_weight=1e-3`)  | 0.257 | KL tax on top of the DS penalty      |
| ST-VQ-VAE (`K=64`) | 0.366 | codebook only ~50% used                     |
| DS-VQ-VAE (`K=64`) | 0.445 | worst; DS latents quantize badly            |

See `figures/F2_all_metrics.png` for per-family bars with 95% bootstrap CIs
and `figures/F1_all_reconstruction.png` for one held-out example per family.

## Sweeps

### VAE — `kl_weight ∈ {1e-4, 1e-3, 1e-2, 1e-1}`

`figures/F5_kl_sweep.png` and `.csv`.

| kl_weight | EMD    | Mean KL |
| --------: | -----: | ------: |
| 1e-4      | 0.2406 | 44.7    |
| 1e-3      | 0.2416 | 9.0     |
| 1e-2      | 0.2627 | 1.2     |
| 1e-1      | 0.3178 | 0.39    |

- `1e-4` and `1e-3` tie on reconstruction; `1e-3` posterior is 5× tighter.
  → `1e-3` is the operating point.
- `1e-1` is nearly collapsed to the prior; recon +50%.

### VQ-VAE — `codebook_size ∈ {16, 64, 256}`

`figures/F6_codebook_sweep.png` and `.csv`.

| K   | EMD    | Perplexity | Codes actually used |
| --: | -----: | ---------: | ------------------: |
| 16  | 0.890  | 11.5       | 100%                |
| 64  | 0.852  | 11.2       | 47%                 |
| 256 | 0.364  | 28.2       | 13%                 |

- Bigger K helps despite dead-code collapse — the ~28 effective codes at
  K=256 outperform the 11-code plateau of smaller K.
- Classic VQ-VAE collapse; a codebook-reset heuristic would likely help
  but was not implemented (out of scope for this thread).

## Interpretation

1. **The Set Transformer is already a competent set-distribution learner
   at this scale.** Its deterministic bottleneck reconstructs the mix of
   distributions well enough that regularization only costs fidelity.
2. **Attention isn't a wash.** ST-Vanilla beats DS-Vanilla by 25% EMD
   averaged over families, and the gap is largest on structured
   distributions (`spiral`, `two_moons`, `mixture_3g`).
3. **VAE tax is uniform (~+20% EMD).** No family is helped by the
   Gaussian bottleneck; the smoothing hurts spiky ones (`spiral`,
   `mixture_3g`) most.
4. **VQ-VAE collapses.** Even at K=256 only 13% of codes see use. This
   is the expected VQ-VAE failure mode without codebook reset; not a
   surprise, just a confirmation.

## What we did NOT test (and why)

- **Generative capabilities** (sampling new sets from the prior;
  interpretability of VQ codes). User is not interested in the generative
  angle for this task; reconstruction was the criterion.
- **Codebook-reset VQ-VAE, hierarchical SetVAE (Kim et al.), PointFlow,
  diffusion-over-point-clouds.** All bigger implementation lifts. Not
  justified by the reconstruction findings above.

## Reproducing the numbers

```bash
# from set_transformer/ (the submodule workdir):
python experiments/synthetic_sets/1_generate_data.py
python experiments/synthetic_sets/2_train_baseline.py
python experiments/synthetic_sets/2_train_vae.py --kl_weight 1e-3
python experiments/synthetic_sets/2_train_vqvae.py --codebook_size 64
python experiments/synthetic_sets/2_train_ds_baseline.py
python experiments/synthetic_sets/2_train_ds_vae.py --kl_weight 1e-3
python experiments/synthetic_sets/2_train_ds_vqvae.py --codebook_size 64
python experiments/synthetic_sets/7_compare_all_encoders.py --models "ST-Vanilla:<ckpt>,..."
```

Sweep scripts: `5_vae_kl_sweep_analysis.py`, `6_vqvae_codebook_sweep_analysis.py`.
