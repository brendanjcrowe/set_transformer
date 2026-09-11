"""Decoder-collapse diagnostics for Sinkhorn-pretrained ST checkpoints (oddeven.md 2026-09-06).

Usage: python3 diagnostics/measure_st_decoder_collapse.py <checkpoint.pt> [...]

For each checkpoint: held-out rows 20000:22000 of oe50_short_pf_dataset.npz
(never seen by any run, which trained on the first 20000 rows), in the RL
frame ((x - 25.5) / 24.5), weight channel scaled by N exactly as the trainer.
Reports
  out_spread   mean over clouds of std over the 50 decoded particles
               (0 = the decoder emits ONE point)
  centroid_std std over clouds of the decoded centroid (0 = same point for
               every input, i.e. still on the plateau)
  lat_abs_std  mean over the 64 latent dims of the across-cloud std
  own_loss     weighted Sinkhorn(recon uniform vs target weighted) at the
               run's own blur
  common_loss  the same at blur 0.02 so runs are comparable
  point_mass   weighted Sinkhorn at blur 0.02 of a single point at the
               weighted mean vs the target -- the "one point" floor
  r_mean       |pearson r| between decoded centroid and target weighted mean
"""
import sys, glob, json
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from set_transformer.models.pf_set_transformer import PFSetTransformer
from set_transformer.loss import SinkhornLoss

d = np.load(ROOT / "experiments/odd_even/data/oe50_short_pf_dataset.npz", allow_pickle=True)
P = torch.from_numpy(d["particles"][20000:22000]).float()
W = torch.from_numpy(d["weights"][20000:22000]).float()
centre, scale = float(d["particle_centre"]), float(d["particle_scale"])
X = (P - centre) / scale
Wc = torch.clamp(torch.nan_to_num(W), min=0.0)
Wc = Wc / (Wc.sum(-1, keepdim=True) + 1e-8)
inp = torch.cat([X, (Wc * Wc.shape[-1]).unsqueeze(-1)], dim=-1)
target_mean = (X.squeeze(-1) * Wc).sum(-1)

dev = "cuda" if torch.cuda.is_available() else "cpu"
common = SinkhornLoss(blur=0.02, scaling=0.5).to(dev)
with torch.no_grad():
    pm = target_mean[:, None, None].expand(-1, 50, 1).contiguous().to(dev)
    point_mass = float(common(pm, X.to(dev), None, Wc.to(dev)))

rows = []
paths = sys.argv[1:]
for ck_path in paths:
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = PFSetTransformer(
        num_particles=cfg.num_particles, dim_particles=cfg.dim_particles + 1,
        num_encodings=cfg.num_encodings, dim_encoder=cfg.dim_encoder,
        num_inds=cfg.num_inds, dim_hidden=cfg.dim_hidden, num_heads=cfg.num_heads,
        ln=cfg.use_layer_norm, dim_output_particles=cfg.dim_particles,
    ).to(dev).eval()
    model.load_state_dict(ck["model_state_dict"])
    own = SinkhornLoss(blur=cfg.sinkhorn_blur, scaling=cfg.sinkhorn_scaling).to(dev)
    with torch.no_grad():
        lat = model.encode(inp.to(dev))
        rec = model.decoder(lat)
        out_spread = float(rec.std(dim=1).mean())
        cent = rec.mean(dim=1).squeeze(-1)
        centroid_std = float(cent.std())
        lat_abs_std = float(lat.flatten(1).std(dim=0).mean())
        own_loss = float(own(rec, X.to(dev), None, Wc.to(dev)))
        common_loss = float(common(rec, X.to(dev), None, Wc.to(dev)))
        r = float(np.corrcoef(cent.cpu().numpy(), target_mean.numpy())[0, 1]) if centroid_std > 0 else float("nan")
    rows.append(dict(blur=cfg.sinkhorn_blur, epoch=ck["epoch"], step=ck["global_step"],
                     out_spread=out_spread, centroid_std=centroid_std, lat_abs_std=lat_abs_std,
                     own_loss=own_loss, common_loss=common_loss, r_mean=abs(r), path=ck_path))

rows.sort(key=lambda r: r["blur"])
print(f"point-mass-at-weighted-mean floor (blur 0.02): {point_mass:.4f}")
print(f"{'blur':>6} {'ep':>3} {'out_spread':>10} {'centroid_std':>12} {'lat_abs_std':>11} {'own_loss':>9} {'common@.02':>10} {'|r| cent,mean':>13}")
for r in rows:
    print(f"{r['blur']:>6} {r['epoch']:>3} {r['out_spread']:>10.5f} {r['centroid_std']:>12.4f} {r['lat_abs_std']:>11.2e} {r['own_loss']:>9.4f} {r['common_loss']:>10.4f} {r['r_mean']:>13.4f}")
json.dump(rows, open("st_decoder_collapse_results.json", "w"), indent=1)
