# Set Transformer VAE / VQ-VAE on synthetic 2D distributions

Compares three Set Transformer autoencoder variants — vanilla `PFSetTransformer`,
`SetVAE`, `SetVQVAE` — on a mix of synthetic 2D point-set distributions
(Gaussians, mixtures, uniforms, ring, two moons, swiss roll, spiral).

## Pipeline

```bash
# All commands run from the set_transformer/ submodule.
# 1. Generate train + eval splits + a distribution gallery figure
python experiments/synthetic_sets/1_generate_data.py

# 2. Train each variant (same hparams aside from variant-specific ones)
python experiments/synthetic_sets/2_train_baseline.py
python experiments/synthetic_sets/2_train_vae.py
python experiments/synthetic_sets/2_train_vqvae.py

# 3. Reconstruction grid (F1) + per-family metrics chart (F2)
python experiments/synthetic_sets/3_compare_reconstruction.py \
    --baseline_ckpt experiments/synthetic_sets/runs/baseline/checkpoints/checkpoint_best.pt \
    --vae_ckpt      experiments/synthetic_sets/runs/vae/checkpoints/checkpoint_best.pt \
    --vqvae_ckpt    experiments/synthetic_sets/runs/vqvae/checkpoints/checkpoint_best.pt

# 4. VQ-VAE codebook probes (F7)
python experiments/synthetic_sets/4_probe_vqvae.py \
    --vqvae_ckpt experiments/synthetic_sets/runs/vqvae/checkpoints/checkpoint_best.pt
```

Outputs land in `experiments/synthetic_sets/figures/`.
