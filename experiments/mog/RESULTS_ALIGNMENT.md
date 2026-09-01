# Latent metric alignment — results

**Question.** A set autoencoder's reconstruction loss constrains each code individually
and says nothing about the *relation* between codes. If we add a term that forces latent
pairwise geometry to mirror the pairwise EMD geometry of the inputs — similar point clouds
get similar codes — what does it cost, and what does it buy?

**Answer.** It costs nothing measurable and buys a great deal. On the Set Transformer,
held-out latent↔EMD correlation goes **0.674 → 0.993** and kNN overlap **0.259 → 0.580**,
while reconstruction is unchanged to four decimal places. Retrieval error over the oracle
drops **80%**. The benefit survives out-of-distribution decoding.

## Setup

- Variable-separability MoG point sets (`data_varsep/`), 100 points each, 20k train /
  2k eval. Sinkhorn reconstruction loss, 60 epochs, 5 seeds.
- Arms: `st_ae` and `ds_ae`, each unaligned (`runs_varsep_sinkhorn/`) and aligned
  (`runs_align_sinkhorn/`). Identical architecture, loss, and schedule; only the alignment
  term differs.
- Alignment: Pearson correlation between cosine latent distances and precomputed debiased
  Sinkhorn distances, `λ=0.2`, warmup 15 epochs then a 15-epoch linear ramp.
- Target geometry: full 20000² pairwise matrix (`emd_train.npy`, ~1 h on an RTX 4070 Ti).
  Verified against `geomloss(debias=True)` to <1e-3; distribution well spread (mean 3.48,
  std 2.51), so the correlation target carries real signal.

## Headline

Mean [95% bootstrap CI] over 5 seeds, scored on 2,000 held-out clouds:

| method | arm | Pearson r | Spearman ρ | kNN@10 | val EMD |
|---|---|---|---|---|---|
| ST-AE | unaligned | 0.674 [0.614, 0.746] | 0.702 | 0.259 | 0.3193 [0.3151, 0.3235] |
| ST-AE | **aligned** | **0.993** [0.9928, 0.9934] | 0.986 | **0.580** | 0.3194 [0.3151, 0.3238] |
| DS-AE | unaligned | 0.904 [0.897, 0.911] | 0.895 | 0.452 | 0.4698 [0.4646, 0.4744] |
| DS-AE | **aligned** | 0.988 | 0.981 | 0.580 | 0.4708 [0.4651, 0.4771] |

**Reconstruction is unchanged** — ST 0.3193 → 0.3194, DS 0.4698 → 0.4708, CIs on top of
each other. The deterministic bottleneck evidently had spare capacity that reconstruction
alone was not using.

Two further observations. **The unaligned baseline is itself informative:** DeepSet's
latent is already well aligned (r ≈ 0.90) while the Set Transformer's is not (r ≈ 0.67),
even though ST reconstructs far better — mean-pooling yields something close to a moment
embedding whose geometry tracks W2, while attention buys reconstruction and tangles the
latent geometry. And both methods converge to the same kNN@10 (0.580), suggesting a
ceiling set by the data rather than the encoder.

## Retrieval — the most legible statement

Mean true EMD of the 3 latent-nearest neighbours over all 2,000 held-out clouds; the
oracle (the 3 EMD-nearest) is the floor at 0.325.

| method | unaligned | aligned | excess over oracle |
|---|---|---|---|
| ST-AE | 0.694 | **0.400** | 0.369 → 0.075, **−80%** |
| DS-AE | 0.446 | **0.390** | 0.121 → 0.065, −46% |

Used as a retrieval index, the unaligned ST encoder returns neighbours more than twice as
far away as it should; alignment closes 80% of that gap for free.

## Reconstruction training actively degrades ST latent geometry

From `figures_align/alignment_curves.png`: during the λ=0 warmup, ST-AE's held-out r
*declines* from 0.90 to 0.83 as reconstruction improves, and unaligned runs end at 0.674
by epoch 60. So alignment is not adding a property that was slowly emerging on its own —
it is **reversing an active drift**. The moment λ engages at epoch 15, r climbs to ~0.99
within ten epochs, before the ramp even reaches its target. DeepSet starts near 0.89 and
stays flat through warmup.

## Out-of-distribution probe

Four decoded codes per model as two pairs: anchors are the most distant pair in a
uniform-per-coordinate sample over each latent's observed range (deliberately
off-manifold — measured at ×7 to ×12 the real codes' own nearest-neighbour distance), with
partners *constructed* at the real-code NN distance, since uniform draws in 128-d are
never near each other.

| model | within-pair EMD | across-pair EMD | separation |
|---|---|---|---|
| ST-AE unaligned | 1.16 | 1.63 | 1.4× |
| ST-AE **aligned** | 1.04 | 2.30 | **2.2×** |
| DS-AE unaligned | 0.55 | 1.42 | 2.6× |
| DS-AE **aligned** | 0.46 | 2.30 | **5.0×** |

The benefit survives off-manifold, and is largest exactly where it was largest
in-distribution. Caveat: ST's within-pair EMD stays ~1.0 even for near-identical codes —
3× its in-distribution reconstruction EMD — so the decoder is far more sensitive off the
manifold than on it; alignment improves the *ratio*, not this absolute sensitivity.

## Phase 2 was not needed

None of the spec's fallback triggers fired: the scatter is linear rather than
curved-but-monotone, training was stable, and Pearson and Spearman agree closely. Soft-rank
Spearman, CKA and stress/MDS remain unbuilt, as the spec directs.

The cosine ceiling I flagged during design never bound either — aligned latent distances
spread to fill the available range (max ~1.45 of a possible 2.0) against EMD running to
30. No reason to switch to Euclidean.

## Reproduce

```bash
python experiments/mog/8_precompute_emd_matrix.py --data_dir experiments/mog/data_varsep
python experiments/mog/2_train_sweep.py --data_dir experiments/mog/data_varsep \
    --out_dir experiments/mog/runs_align_sinkhorn --methods st_ae ds_ae \
    --seeds 0 1 2 3 4 --num_epochs 60 --loss sinkhorn \
    --align_lambda 0.2 --align_warmup 15 --align_ramp 15
python experiments/mog/9_eval_alignment.py       # metrics, scatter, summary, curves
python experiments/mog/10_inference_alignment.py # reconstruction grids + retrieval
python experiments/mog/11_latent_sampling.py     # OOD decode probe
```

Figures and CSVs land in `experiments/mog/figures_align/`.

**One hazard, guarded.** Model selection is by val EMD, which is blind to alignment. If
the best epoch lands before the λ ramp completes, the "aligned" checkpoint is not actually
aligned — a silent mislabel that would void the comparison. All 10 runs here are safe
(best epoch 50-59 of 60, λ at full 0.20), and the trainer now warns, records `best_epoch`,
and reports `selected_val_r` — the correlation at the *shipped* checkpoint.
