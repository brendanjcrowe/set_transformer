"""Behaviour-clone a locomotion gait into a benchmark policy (Ant-Tag rescue probe b).

Ant-Tag's failure is motor learning, not task difficulty: the pretrained locomotion
policy tags 4/20 episodes and moves the ant 4.9 units, where random actions tag 0/20 and
move 2.2. So an agent that already walks can be judged on how well it *uses its belief*,
which is the only thing the benchmark is trying to measure.

The locomotion policy cannot be loaded directly: it is a flat 31-dim ``ActorCriticPolicy``
whose first layer is ``(64, 31)``, while a benchmark policy consumes a Dict observation
through a per-method feature extractor and enters its MLP at ``(64, 128)``. Transferring
the later layers alone would feed them a representation they were never trained on.

Cloning sidesteps that. The teacher's actions are recorded against the *student's* Dict
observations, then every method learns to reproduce them through its own extractor. Each
method therefore starts from the same behaviour rather than the same weights -- which is
the fair comparison, since the weights are not commensurable across methods to begin with.

    python experiments/benchmark/pretrain/warmstart_locomotion.py --env ant_tag \
        --teacher models/ant_locomotion_policy.zip \
        --vecnorm models/locomotion_vecnorm.pkl --method gaussian

Writes ``<out_dir>/<env>/<method>_warmstart.zip``, loadable by ``train.py --init_policy``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from set_transformer.rl.benchmark.registry import (
    build_extractor_kwargs,
    get_env_spec,
    get_method_spec,
)

train_mod = __import__("train")

DEFAULT_OUT = Path("experiments/benchmark/pretrain/warmstart")


def load_teacher(path: str, vecnorm: str | None):
    """The teacher plus the obs normalization it was trained under."""
    policy = PPO.load(path)
    normalize = (lambda o: o)
    if vecnorm:
        with open(vecnorm, "rb") as fh:
            vn = pickle.load(fh)
        def normalize(o, vn=vn):  # noqa: E306
            return np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + 1e-8),
                           -10.0, 10.0).astype(np.float32)
    return policy, normalize


def collect_demonstrations(env_spec, teacher, normalize, episodes: int, seed: int):
    """Roll the teacher; record the STUDENT's Dict observations against its actions.

    The teacher reads the raw environment observation (which still carries the target
    position); the student's observation is the masked Dict. Pairing them is the point --
    the student learns the gait from what it can actually see.
    """
    env = train_mod.make_env_thunk(env_spec, seed, for_eval=True)()
    obs_buf, part_buf, act_buf, tags = [], [], [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        base = env.unwrapped._get_obs(True) if hasattr(env.unwrapped, "_get_obs") else None
        for _ in range(env_spec.max_ep_steps or 400):
            raw = np.asarray(base if base is not None else obs["obs"], dtype=np.float32)
            action, _ = teacher.predict(normalize(raw), deterministic=False)
            obs_buf.append(np.asarray(obs["obs"], dtype=np.float32))
            part_buf.append(np.asarray(obs["particles"], dtype=np.float32))
            act_buf.append(np.asarray(action, dtype=np.float32))
            obs, _, terminated, truncated, _ = env.step(action)
            base = env.unwrapped._get_obs(True) if hasattr(env.unwrapped, "_get_obs") else None
            if terminated or truncated:
                tags += int(terminated)
                break
        if (ep + 1) % max(1, episodes // 5) == 0:
            print(f"  episode {ep + 1}/{episodes}: {len(act_buf)} transitions, "
                  f"{tags} tags", flush=True)
    env.close()
    return (np.asarray(obs_buf), np.asarray(part_buf), np.asarray(act_buf), tags)


def _deterministic_action(policy, batch) -> torch.Tensor:
    """The policy's mean action for a Dict batch, differentiably.

    ``policy.forward`` samples and returns detached actions, so BC has to go through the
    layers directly: features -> actor MLP -> action head. ``extract_features`` returns a
    (pi, vf) pair when the actor and critic do not share an extractor.
    """
    features = policy.extract_features(batch)
    if isinstance(features, tuple):
        features = features[0]
    latent_pi = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent_pi)


def clone(env_spec, method: str, demos, args, teacher_log_std=None) -> PPO:
    """Fit a fresh benchmark policy to the demonstrations by regression on actions."""
    method_spec = get_method_spec(method)
    extractor_kwargs = build_extractor_kwargs(
        method_spec, args.features_dim, args.obs_mlp_hidden_dims,
        encoder_arch=dict(num_encodings=8, dim_encoder=2, num_inds=32, dim_hidden=64,
                          num_heads=4, ln=True),
        particle_scale=env_spec.particle_scale,
    )
    vec = DummyVecEnv([train_mod.make_env_thunk(env_spec, args.seed, for_eval=True)])
    model = PPO(
        "MultiInputPolicy", vec, seed=args.seed, device=args.device,
        policy_kwargs=dict(features_extractor_class=method_spec.extractor_class,
                           features_extractor_kwargs=extractor_kwargs),
        verbose=0,
    )

    # Copy the teacher's exploration noise. BC fits only the action MEAN, so log_std
    # keeps PPO's init of 0 (std = 1.0) -- and PPO collects rollouts *stochastically*, so
    # the cloned gait is swamped by noise the moment training starts and the first
    # gradients erase it. Measured: the same clone tags 2/12 acting deterministically and
    # 0/12 stochastically at std 1.0. The teacher's own log_std is the right level by
    # construction: it is the noise the gait was trained under and survived.
    if args.init_log_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(float(args.init_log_std))
    elif teacher_log_std is not None:
        with torch.no_grad():
            model.policy.log_std.copy_(torch.as_tensor(
                teacher_log_std, dtype=model.policy.log_std.dtype,
                device=model.policy.log_std.device))
    print(f"  policy log_std set to "
          f"{model.policy.log_std.detach().cpu().numpy().round(3)}", flush=True)

    obs_np, part_np, act_np = demos[0], demos[1], demos[2]
    device = model.device
    obs_t = torch.as_tensor(obs_np, device=device)
    part_t = torch.as_tensor(part_np, device=device)
    act_t = torch.as_tensor(act_np, device=device)

    params = [p for p in model.policy.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    n = len(act_t)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            batch = {"obs": obs_t[idx], "particles": part_t[idx]}
            pred = _deterministic_action(model.policy, batch)
            loss = torch.nn.functional.mse_loss(pred, act_t[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            total += float(loss) * len(idx)
        print(f"  epoch {epoch + 1}/{args.epochs}: bc mse {total / n:.4f}", flush=True)
    vec.close()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="ant_tag")
    ap.add_argument("--method", default="gaussian")
    ap.add_argument("--teacher", default="models/ant_locomotion_policy.zip")
    ap.add_argument("--vecnorm", default="models/locomotion_vecnorm.pkl")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--features_dim", type=int, default=128)
    ap.add_argument("--obs_mlp_hidden_dims", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_log_std", type=float, default=None,
                    help="override the cloned policy's log_std (default: copy the "
                         "teacher's, which is the noise its gait survived)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    env_spec = get_env_spec(args.env)
    teacher, normalize = load_teacher(args.teacher, args.vecnorm)
    print(f"collecting demonstrations from {args.teacher} ...", flush=True)
    demos = collect_demonstrations(env_spec, teacher, normalize, args.episodes, args.seed)
    print(f"{len(demos[2])} transitions, {demos[3]} teacher tags", flush=True)

    teacher_log_std = None
    if hasattr(teacher.policy, "log_std"):
        teacher_log_std = teacher.policy.log_std.detach().cpu().numpy()
    model = clone(env_spec, args.method, demos, args, teacher_log_std)
    out = args.out_dir / args.env
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.method}_warmstart.zip"
    model.save(str(path))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
