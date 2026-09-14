"""Turn a ``Trainer`` checkpoint of an arm autoencoder into the file the RL side loads.

Batch 10.11 of the harness refactor (``refactor_plans.md`` section 10, 2026-09-14). A ``Trainer``
checkpoint of :class:`~set_transformer.models.cgf_arm_ae.CGFArmAutoencoder` or
:class:`~set_transformer.models.pooled_arm_ae.PooledArmAutoencoder` holds the whole autoencoder
under ``extractor.`` / ``decoder.`` prefixes. The RL extractor's loader wants its own tensors
under its own prefix and its geometry record: :func:`export_arm_checkpoint` copies the extractor,
loads the checkpoint's ``extractor.*`` tensors into the copy (strict, so a checkpoint of another
geometry is refused) and hands the copy to
:func:`~set_transformer.rl.pretrained_encoder.encoder_checkpoint`, the one writer every learned
extractor's ``checkpoint_state()`` / ``checkpoint_config()`` feed (batch 10.4). The CGF arm's
export (``models/cgf_arm_ae.py::export_arm_checkpoint``, 2026-09-11) was the same steps spelled
for the CGF extractor alone; it now calls this function and writes the same content.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import torch

from set_transformer.rl.pretrained_encoder import encoder_checkpoint

#: Prefix ``Trainer`` checkpoints carry on the arm extractor's tensors (the autoencoders name the
#: wrapped extractor ``self.extractor``).
EXTRACTOR_PREFIX = "extractor."


def plain(obj):
    """Dataclass / namespace -> dict so the export payload is pure Python + tensors (the export
    must unpickle without the set_transformer package importable: driver snippets, other envs)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, (dict, list, tuple, str, int, float)):
        return dict(vars(obj))
    return obj


def extractor_state_from_trainer_checkpoint(loaded: dict, extractor) -> dict:
    """The ``extractor.*`` tensors of a ``Trainer`` checkpoint, unprefixed, checked against the
    extractor's own key set (missing / unexpected keys are named)."""
    full = loaded["model_state_dict"]
    state = {k[len(EXTRACTOR_PREFIX):]: v.detach().cpu() for k, v in full.items()
             if k.startswith(EXTRACTOR_PREFIX)}
    expected = set(extractor.state_dict().keys())
    if set(state) != expected:
        raise RuntimeError(
            "checkpoint's extractor keys do not match the extractor built from the "
            f"CLI: missing {sorted(expected - set(state))}, unexpected "
            f"{sorted(set(state) - expected)}")
    return state


def export_arm_checkpoint(trainer_checkpoint: Path, out_path: Path, extractor, *,
                          encoder_name: str, particle_centre: float, objective: str,
                          data_path: str, extra_config: dict | None = None,
                          pretraining: str = "3_train_st.py") -> Path:
    """Write the RL-loadable file for one ``Trainer`` checkpoint of an arm autoencoder.

    ``model_state_dict`` / ``config`` come from the extractor's own ``checkpoint_state()`` /
    ``checkpoint_config()`` (through ``encoder_checkpoint``), evaluated on a COPY of the extractor
    that holds the checkpoint's tensors -- the live extractor keeps its last-epoch weights, and
    ``best`` and ``latest`` can be exported in any order. ``config`` adds the provenance the CGF
    export always wrote (encoder, objective, pretraining script, dataset, particle_centre,
    encoder_params, the objective's extras); the top level adds the Trainer's epoch / step /
    best loss / alignment record, its config flattened to a plain dict, and the source path.
    """
    loaded = torch.load(trainer_checkpoint, map_location="cpu", weights_only=False)
    state = extractor_state_from_trainer_checkpoint(loaded, extractor)
    snapshot = copy.deepcopy(extractor).cpu()
    snapshot.load_state_dict(state, strict=True)
    config = {
        "encoder": encoder_name,
        "objective": objective,
        "pretraining": pretraining,
        "dataset": str(data_path),
        "particle_centre": float(particle_centre),
        "encoder_params": int(snapshot.encoder_parameter_count()),
        **(extra_config or {}),
    }
    payload = encoder_checkpoint(
        snapshot, config=config,
        epoch=loaded.get("epoch"),
        global_step=loaded.get("global_step"),
        best_val_loss=loaded.get("best_val_loss"),
        particle_centre=float(particle_centre),
        alignment=loaded.get("alignment"),
        trainer_config=plain(loaded.get("config")),
        source_checkpoint=str(trainer_checkpoint),
    )
    out_path = Path(out_path)
    torch.save(payload, out_path)
    return out_path
