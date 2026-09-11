"""Collect particle-filter snapshots from any registered benchmark env.

Generalizes the Ant-Tag-only ``experiments/ant_tag/2_collect_pf_dataset.py``. Rather than
driving a bare env and a hand-built filter, this steps the *same*
``PFDictObservationWrapper`` stack the RL trainer uses (via ``train.py``'s own
``make_env_thunk``) and records ``obs["particles"]`` — so the belief distribution an
encoder is pretrained on is by construction the one the policy will see at RL time. That
match is exactly what went stale for the old ant_tag checkpoint after the PF was fixed.

Actions come from a random policy by default. ``--policy_path`` loads an SB3 policy
instead, which matters on envs where random actions never reach the informative part of
the state space (Ant-Tag: the ant must actually move for the target to be observed).

    python experiments/benchmark/pretrain/1_collect_pf_dataset.py --env odd_even
    python experiments/benchmark/pretrain/1_collect_pf_dataset.py --env ant_tag \
        --episodes 200 --policy_path models/ant_locomotion_policy.zip

Writes ``<out_dir>/<env>/{train,eval}.points.npy`` of shape
``(num_samples, num_particles, particle_dim)``, loadable by ``POMDPDataset``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from set_transformer.rl.benchmark.registry import get_env_spec

train_mod = __import__("train")

DEFAULT_OUT = Path("experiments/benchmark/pretrain/data")


def collect(env_spec, episodes: int, max_steps: int, seed: int, policy_path: str | None,
            deterministic: bool) -> np.ndarray:
    """Step the wrapped env and record one particle snapshot per timestep."""
    # for_eval=True: no reward shaping, which does not affect the belief and keeps the
    # collection env identical to the one the trainer evaluates on.
    env = train_mod.make_env_thunk(env_spec, seed, for_eval=True)()

    policy = None
    if policy_path:
        from stable_baselines3 import PPO
        policy = PPO.load(policy_path)
        print(f"loaded action policy: {policy_path}", flush=True)

    snapshots, lengths = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        for t in range(max_steps):
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=deterministic)
            else:
                action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            snapshots.append(np.asarray(obs["particles"], dtype=np.float32))
            if terminated or truncated:
                break
        lengths.append(t + 1)
        if (ep + 1) % max(1, episodes // 10) == 0:
            print(f"  episode {ep + 1}/{episodes}: {len(snapshots)} snapshots", flush=True)
    env.close()

    data = np.asarray(snapshots, dtype=np.float32)
    print(f"collected {data.shape} | mean episode length {np.mean(lengths):.1f}", flush=True)
    return data


def report_spread(data: np.ndarray) -> dict:
    """Belief-spread summary. A dataset whose spread never varies means the filter is not
    producing diverse beliefs, and pretraining on it will teach the encoder very little —
    worth seeing before spending an hour on the EMD matrix."""
    spread = data.std(axis=1).mean(axis=1)
    stats = {
        "num_samples": int(len(data)),
        "num_particles": int(data.shape[1]),
        "particle_dim": int(data.shape[2]),
        "spread_mean": float(spread.mean()),
        "spread_std": float(spread.std()),
        "spread_min": float(spread.min()),
        "spread_max": float(spread.max()),
        "unique_spread_values": int(len(np.unique(np.round(spread, 6)))),
    }
    print(json.dumps(stats, indent=2), flush=True)
    if stats["unique_spread_values"] < 10:
        print("WARNING: the belief takes very few distinct shapes — this env may not "
              "support a meaningful pretraining/alignment study.", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--eval_episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=0,
                    help="cap per episode; 0 = the env's registry max_ep_steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy_path", default=None,
                    help="SB3 policy for action selection (default: random actions)")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    env_spec = get_env_spec(args.env)
    max_steps = args.max_steps or env_spec.max_ep_steps or 200
    out_dir = args.out_dir / args.env
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"env": args.env, "episodes": args.episodes, "max_steps": max_steps,
            "seed": args.seed, "policy_path": args.policy_path,
            "particle_filter": env_spec.particle_filter_class.__name__,
            "num_particles": env_spec.num_particles}

    for split, episodes, seed in (("train", args.episodes, args.seed),
                                  ("eval", args.eval_episodes, args.seed + 100_000)):
        print(f"\n=== {args.env} / {split} ===", flush=True)
        data = collect(env_spec, episodes, max_steps, seed, args.policy_path,
                       args.deterministic)
        np.save(out_dir / f"{split}.points.npy", data)
        meta[split] = report_spread(data)
        print(f"wrote {out_dir / f'{split}.points.npy'}", flush=True)

    (out_dir / "collection_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
