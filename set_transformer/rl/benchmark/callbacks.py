"""Training callbacks for the belief-encoder benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CGFTNormCallback(BaseCallback):
    """Log the distribution of the learned CGF sampling-point norms ``||t_m||``.

    The CGF extractor learns where in ``t``-space to evaluate the log-MGF. The norms of
    those points reveal what the method is actually doing: norms near 0 mean it has
    collapsed into (a reparameterization of) the k-moments baseline, while larger norms
    mean it is exploiting the non-local / support-function (max-pooling) regime that
    finite moments cannot represent — the empirical argument that CGF is a distinct
    method rather than approximate k-moments.

    No-op unless the policy's feature extractor exposes ``t_value_norms`` (i.e. the ``cgf``
    method). Records mean / min / max / median to the SB3 logger (→ TensorBoard) every
    ``log_freq`` env steps, and writes a final ``cgf_t_norms.npz`` snapshot (per-point
    norms + the raw learned ``t`` vectors) into ``save_dir`` for offline paper analysis.
    """

    def __init__(self, save_dir: str | Path, log_freq: int = 20_000, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.log_freq = log_freq

    def _extractor(self):
        fe = self.model.policy.features_extractor
        return fe if hasattr(fe, "t_value_norms") else None

    def _record(self) -> None:
        fe = self._extractor()
        if fe is None:
            return
        norms = fe.t_value_norms().cpu().numpy()
        self.logger.record("cgf/t_norm_mean", float(norms.mean()))
        self.logger.record("cgf/t_norm_min", float(norms.min()))
        self.logger.record("cgf/t_norm_max", float(norms.max()))
        self.logger.record("cgf/t_norm_median", float(np.median(norms)))

    def _on_step(self) -> bool:
        n_envs = getattr(self.training_env, "num_envs", 1)
        if self.n_calls % max(self.log_freq // n_envs, 1) == 0:
            self._record()
        return True

    def _on_training_end(self) -> None:
        fe = self._extractor()
        if fe is None:
            return
        norms = fe.t_value_norms().cpu().numpy()
        t_values = fe.t_values.detach().cpu().numpy()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self.save_dir / "cgf_t_norms.npz", t_norms=norms, t_values=t_values)
        if self.verbose:
            print(
                f"[cgf] learned ||t|| mean={norms.mean():.3f} "
                f"min={norms.min():.3f} max={norms.max():.3f} "
                f"-> {self.save_dir/'cgf_t_norms.npz'}"
            )
