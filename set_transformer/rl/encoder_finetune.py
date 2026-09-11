"""Finetuning a pretrained Set Transformer encoder under PPO without collapsing it.

Two fixes, each fighting an SB3 behaviour that would silently undo it
(oddeven.md 2026-09-05/06; ported to Ant-Tag 2026-09-05 -- the smart_hard
finetune arms flipped between 0% and 10% by seed at the shared rate):

* :func:`scale_encoder_learning_rate` (fix 1) puts the encoder in its own
  optimizer param group at ``scale`` x the head rate. SB3 writes the scheduled
  rate into EVERY param group before each ``train()``, so a plain second
  group would be back at the head rate after the first update; the
  :class:`_ScaledLRParamGroup` dict subclass keeps the scale through every
  write. ``PPO.load`` rebuilds a one-group optimizer and refuses a two-group
  state dict, so the optimizer's ``state_dict`` is collapsed to one fresh
  group on save (:func:`_group_collapsing`): every saved agent stays
  loadable, at the cost that resuming TRAINING restarts the Adam moments.
* :class:`UnfreezeEncoderCallback` (fix 2) keeps the reloaded encoder frozen
  for ``unfreeze_at`` env steps, then releases it, so the heads learn to read
  the pretrained code before the encoder sees their gradients.

Both scripts (``experiments/odd_even/4_train_rl_st.py``,
``experiments/ant_tag/4_train_rl_st.py``) import from here; the regression
tests are ``tests/test_st_finetune_fixes.py`` (Odd-Even) and
``tests/test_ant_tag_st_finetune_lr.py`` (Ant-Tag).
"""
from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class _ScaledLRParamGroup(dict):
    """An optimizer param group whose ``lr`` is always ``lr_scale`` x the value written.

    SB3's ``utils.update_learning_rate`` does ``param_group["lr"] = lr`` on every
    group at the start of each ``train()``, which would wipe out a plain second
    group's smaller rate. torch keeps the very dict object it was given in
    ``optimizer.param_groups``, so a dict subclass intercepting ``__setitem__``
    for ``"lr"`` makes the scale stick through every schedule update, with no
    change to the shared training loop and nothing extra for ``model.save`` to
    pickle (``state_dict()`` copies groups into plain dicts).
    """

    def __setitem__(self, key, value):
        if key == "lr":
            value = value * dict.get(self, "lr_scale", 1.0)
        super().__setitem__(key, value)


class EncoderLRLoggingCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        groups = self.model.policy.optimizer.param_groups
        self.logger.record("train/encoder_learning_rate", float(groups[0]["lr"]))


def _group_collapsing(optimizer_class):
    """Optimizer subclass whose saved ``state_dict`` looks like a FRESH
    single-group optimizer over all parameters.

    ``PPO.load`` rebuilds a plain policy (one param group) and then calls
    ``optimizer.load_state_dict`` with ``exact_match=True``; a two-group state
    dict is refused ("different number of parameter groups") and every saved
    agent -- final, best_model, checkpoints -- becomes unloadable. Evaluation
    never needs the Adam moments, so the state is dropped and the groups
    merged. Resuming TRAINING from such a save restarts the moments; that is
    the documented cost of fix 1.
    """
    class GroupCollapsing(optimizer_class):
        def state_dict(self):
            sd = super().state_dict()
            groups = sd["param_groups"]
            n = sum(len(g["params"]) for g in groups)
            plain = {k: v for k, v in groups[-1].items() if k not in ("params", "lr_scale")}
            return {"state": {}, "param_groups": [{**plain, "params": list(range(n))}]}
    GroupCollapsing.__name__ = f"{optimizer_class.__name__}GroupCollapsing"
    return GroupCollapsing


def scale_encoder_learning_rate(model, scale: float) -> None:
    """Finetune-collapse fix 1: give the encoder its own, smaller learning rate.

    Rebuilds the policy optimizer with two param groups: the encoder in a
    :class:`_ScaledLRParamGroup` carrying ``lr_scale``, everything else plain.
    ``--lr_anneal`` still applies to both (the scaled group tracks the schedule
    at ``scale`` x the rate).
    """
    policy = model.policy
    encoder_params = list(policy.features_extractor.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [p for p in policy.parameters() if id(p) not in encoder_ids]
    base_lr = model.lr_schedule(1.0)
    encoder_group = _ScaledLRParamGroup(params=encoder_params, lr_scale=scale)
    encoder_group["lr"] = base_lr          # -> base_lr * scale via __setitem__
    policy.optimizer = _group_collapsing(policy.optimizer_class)(
        [encoder_group, {"params": other_params, "lr": base_lr}],
        lr=base_lr, **policy.optimizer_kwargs)
    assert policy.optimizer.param_groups[0] is encoder_group
    print(f"ST encoder learning rate scaled by {scale} "
          f"({len(encoder_params)} encoder tensors at {encoder_group['lr']:.2e}, "
          f"{len(other_params)} head tensors at {base_lr:.2e}; anneal applies to both)")


class UnfreezeEncoderCallback(BaseCallback):
    """Finetune-collapse fix 2: keep the encoder frozen for the first
    ``unfreeze_at`` environment steps, then release it.

    The heads first learn to read the pretrained code while it cannot move;
    only then does the encoder see gradients, which by that point are
    informative rather than the noise a random head emits. The encoder's
    parameters were in the optimizer all along (SB3 builds it over
    ``policy.parameters()`` regardless of requires_grad), so flipping the flag
    is sufficient. Logs ``st/encoder_trainable`` so the switch is visible in
    TensorBoard, and prints once.
    """

    def __init__(self, unfreeze_at: int):
        super().__init__()
        self.unfreeze_at = int(unfreeze_at)
        self.done = False

    def _encoders(self):
        found = [self.model.policy.features_extractor]
        for attr in ("actor", "critic", "critic_target"):
            module = getattr(self.model.policy, attr, None)
            other = getattr(module, "features_extractor", None)
            if other is not None and other is not found[0]:
                found.append(other)
        return found

    def _on_step(self) -> bool:
        if not self.done and self.num_timesteps >= self.unfreeze_at:
            n = 0
            for extractor in self._encoders():
                # The extractor's own flag wraps forward() in torch.no_grad()
                # when set (st.py); requires_grad alone would leave the
                # encoder trainable in name only.
                extractor.st_frozen = False
                extractor.encoder.train()
                for param in extractor.encoder.parameters():
                    param.requires_grad_(True)
                    n += 1
            self.done = True
            print(f"UnfreezeEncoderCallback: encoder UNFROZEN at step "
                  f"{self.num_timesteps:,} ({n} tensors now trainable)", flush=True)
        self.logger.record("st/encoder_trainable", float(self.done))
        return True
