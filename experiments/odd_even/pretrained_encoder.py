"""Reload a pretrained belief encoder AFTER PPO construction, and prove it.

Shared by the ST and CGF arms. PITFALLS.md section 1: SB3's
ActorCriticPolicy._build ends with ``self.apply(init_weights)``, which walks
the whole policy INCLUDING the features extractor and re-initialises every
Linear. An encoder loaded in the extractor's __init__ is therefore overwritten
a moment later, while the "loaded encoder" line has already been printed. Two
6M-step runs were lost to that. So every pretrained arm does three things
after ``PPO(...)`` returns: reload, re-freeze, and ASSERT max|delta| == 0
against the checkpoint. The assertion is the only positive evidence in the
logs that the reload landed.

`4_train_rl_st.py` keeps its own ``reload_pretrained_encoder`` (its reference
and live dicts need the ``set_transformer.`` / ``features_extractor.encoder.``
prefix handling) and calls :func:`verify_matches_checkpoint` here for the
comparison; the CGF arm uses :func:`reload_pretrained_cgf` end to end.
"""

from __future__ import annotations

import torch


def max_abs_delta(reference: dict, live: dict) -> tuple[float, int]:
    """Largest |reference - live| over the shared keys, and how many there are.

    Raises if NO key is shared: that means the reload cannot have touched
    the live module, which is the failure this exists to catch.
    """
    shared = [key for key in reference if key in live]
    if not shared:
        raise AssertionError(
            "the checkpoint shares no tensor names with the live encoder; the "
            "reload cannot have done anything")
    worst = 0.0
    for key in shared:
        ref = reference[key].detach().cpu().to(torch.float64)
        cur = live[key].detach().cpu().to(torch.float64)
        if ref.shape != cur.shape:
            raise AssertionError(f"{key}: checkpoint shape {tuple(ref.shape)} vs "
                                 f"live {tuple(cur.shape)}")
        worst = max(worst, float((ref - cur).abs().max()))
    return worst, len(shared)


def verify_matches_checkpoint(reference: dict, live: dict, path: str,
                              label: str = "encoder") -> None:
    worst, n_shared = max_abs_delta(reference, live)
    if worst != 0.0:
        raise AssertionError(
            f"{label} does not match {path}: max|delta| = {worst:.3e} over "
            f"{n_shared} tensors. SB3's init_weights overwrote the reload, or "
            "the wrong checkpoint was loaded.")
    print(f"Verified: {label} matches {path} exactly "
          f"({n_shared} tensors, max|delta| = 0.0)")


def _cgf_reference_state(path: str) -> dict:
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    return loaded["model_state_dict"] if "model_state_dict" in loaded else loaded


def reload_pretrained_cgf(model, path: str, frozen: bool, verify: bool = True) -> None:
    """The CGF arm's post-construction reload. Mirrors the ST arm's steps.

    The extractor IS the encoder (t, norm statistics, readout), so the
    reference is the checkpoint's whole ``model_state_dict`` and the live
    side is the extractor's whole ``state_dict``. ``frozen`` calls
    ``freeze_encoder()``, which also pins the module in eval mode so the
    running norm stops updating.
    """
    extractors = [model.policy.features_extractor]
    for attr in ("actor", "critic", "critic_target"):
        module = getattr(model.policy, attr, None)
        other = getattr(module, "features_extractor", None)
        if other is not None and other is not extractors[0]:
            extractors.append(other)
    for extractor in extractors:
        extractor._load_pretrained_encoder(path)
        if frozen:
            extractor.freeze_encoder()
    print("WeightedCGFFeaturesExtractor: encoder RE-loaded after PPO construction "
          "(SB3 init_weights would otherwise overwrite it)"
          + (" and re-frozen" if frozen else ""))
    if verify:
        verify_matches_checkpoint(_cgf_reference_state(path),
                                  model.policy.features_extractor.state_dict(),
                                  path, label="CGF encoder")
    # Do not bake an absolute pretraining path into the saved policy
    # (PITFALLS.md section 7): SB3 re-runs the constructor on load.
    kwargs = model.policy_kwargs.get("features_extractor_kwargs", {})
    if "pretrained_cgf_model_path" in kwargs:
        kwargs["pretrained_cgf_model_path"] = None
