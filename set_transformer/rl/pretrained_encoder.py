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

Domain-independent, so it lives in the package (moved here from
``experiments/odd_even/pretrained_encoder.py`` on 2026-09-10, which now
re-exports these names).

Since change 2 of the harness centralisation (2026-09-12) there is ONE reload,
:func:`reload_pretrained`, written against the shared encoder interface every learned
extractor exposes (``load_pretrained``, ``freeze``, ``reference_state``,
``encoder_state_dict``, ``PRETRAINED_PATH_KWARG``; see ``feature_extractors/st.py``).
:func:`reload_pretrained_cgf`, ``feature_extractors.pooled.reload_pretrained_pooled`` and
both ST scripts' ``reload_pretrained_encoder`` forward to it. Before that the step was
written four times, and the Ant-Tag ST copy never verified.
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


def policy_extractors(model) -> list:
    """Every features extractor on the policy, each once: the shared one first, then the
    policy's own ``pi_features_extractor`` / ``vf_features_extractor`` when they are distinct
    objects (PPO with ``share_features_extractor=False``, the trainer's ``--separate_extractors``;
    batch 9.1, 2026-09-13), then any an actor / critic / critic_target holds of its own
    (SAC-style policies build those)."""
    extractors = [model.policy.features_extractor]
    seen = {id(extractors[0])}
    candidates = [getattr(model.policy, attr, None)
                  for attr in ("pi_features_extractor", "vf_features_extractor")]
    for attr in ("actor", "critic", "critic_target"):
        module = getattr(model.policy, attr, None)
        candidates.append(getattr(module, "features_extractor", None))
    for other in candidates:
        if other is not None and id(other) not in seen:
            extractors.append(other)
            seen.add(id(other))
    return extractors


def reload_pretrained(model, path: str, frozen: bool, verify: bool = True) -> None:
    """The one post-construction reload, for every learned extractor (change 2, 2026-09-12).

    Four steps, all required (PITFALLS.md sections 1 and 7): reload every extractor on the
    policy from ``path``; re-freeze if ``frozen``; ASSERT the live encoder equals the
    checkpoint (max|delta| == 0 -- the only positive evidence in the logs that the reload
    landed) -- for EVERY extractor the policy holds, so a separate value-network extractor
    (``--separate_extractors``) is verified too, not only the shared / actor one; blank the
    checkpoint path in ``policy_kwargs`` so no absolute path is baked into the saved zip. Uses
    only the shared encoder interface, so the ST, CGF and pooled extractors go through exactly
    the same code.
    """
    extractors = policy_extractors(model)
    for extractor in extractors:
        extractor.load_pretrained(path)
        if frozen:
            extractor.freeze()
    name = type(extractors[0]).__name__
    print(f"{name}: encoder RE-loaded after PPO construction "
          "(SB3 init_weights would otherwise overwrite it)"
          + (" and re-frozen" if frozen else "")
          + (f" -- {len(extractors)} extractors (separate actor / critic)"
             if len(extractors) > 1 else ""))
    if verify:
        reference = extractors[0].reference_state(path)
        for index, extractor in enumerate(extractors):
            label = (f"{name} encoder" if len(extractors) == 1
                     else f"{name} encoder [{index + 1}/{len(extractors)}]")
            verify_matches_checkpoint(reference, extractor.encoder_state_dict(), path,
                                      label=label)
    kwargs = model.policy_kwargs.get("features_extractor_kwargs", {})
    key = getattr(extractors[0], "PRETRAINED_PATH_KWARG", None)
    if key is not None and key in kwargs:
        kwargs[key] = None


def reload_pretrained_cgf(model, path: str, frozen: bool, verify: bool = True) -> None:
    """The CGF arm's post-construction reload. Since change 2 a forwarder to
    :func:`reload_pretrained`; kept under this name for both CGF scripts, the Odd-Even
    ``pretrained_encoder.py`` shim and the tests.
    """
    reload_pretrained(model, path, frozen, verify=verify)
