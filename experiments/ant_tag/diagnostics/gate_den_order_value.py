"""
Order-value gate for Twin-Den Ant-Tag: is guessing the wrong den actually
expensive under the real ant body and the 200-step cap?

Three scripted drivers, no particle filter anywhere, each driving the ant
with the Phase-2 locomotion policy via waypoint injection into obs[-2:]
(same harness as gate_dumb_patrol_baseline.py):

  oracle-order  first waypoint = the LOOSE den center (reads ground truth
                deliberately -- this is the upper bound on what knowing the
                tight/loose assignment buys you)
  anti-order    first waypoint = the TIGHT den center (the systematically
                wrong order)
  random-order  coin flip per episode (what an uninformed policy can do)

In all three: once the ant has been within --switch_radius of its first den
center for --switch_steps consecutive steps WITHOUT the true target being
within visible_radius, it gives up and heads for the other den. Whenever the
true target IS within visible_radius, the waypoint becomes the true target
position -- that is what any policy would see and chase.

The gap oracle-order minus anti-order is the maximum achievable value of the
ordering bit. The plan pre-registers >= 15 points before any RL money is
spent.

Usage:
    python3 diagnostics/gate_den_order_value.py --env_id pdomains-ant-tag-dens-v0 \
        --max_steps 200 --n_episodes 100
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdomains  # noqa: F401 - registers pdomains-ant-tag-dens-v0

MODES = ("oracle-order", "anti-order", "random-order")


def run_mode(mode, env, raw, loco_vec, model, args, rng):
    tagged = 0
    lengths = []
    first_was_loose = 0

    for ep in range(args.n_episodes):
        env.reset(seed=args.env_seed + ep)
        tight = int(raw.tight_den)
        loose = 1 - tight
        if mode == "oracle-order":
            first = loose
        elif mode == "anti-order":
            first = tight
        else:
            first = int(rng.integers(2))
        first_was_loose += int(first == loose)

        order = [first, 1 - first]
        leg = 0
        stalled = 0
        ep_len = 0

        for t in range(args.max_steps):
            ant = np.asarray(raw.data.qpos[:2], dtype=np.float64)
            true_target = np.asarray(raw.get_target_pos(), dtype=np.float64)
            visible = np.linalg.norm(ant - true_target) < raw.visible_radius

            den_wp = np.asarray(raw.den_positions[order[leg]], dtype=np.float64)
            if visible:
                waypoint = true_target
                stalled = 0
            else:
                waypoint = den_wp
                # Searched this den long enough without seeing anything? Move on.
                if leg == 0 and np.linalg.norm(ant - den_wp) < args.switch_radius:
                    stalled += 1
                    if stalled >= args.switch_steps:
                        leg = 1
                        stalled = 0

            raw_obs = np.concatenate(
                [raw.data.qpos, raw.data.qvel, waypoint]).astype(np.float32)
            action, _ = model.predict(
                loco_vec.normalize_obs(raw_obs[None, :]), deterministic=True)
            _, _, terminated, truncated, _ = env.step(action[0])
            ep_len += 1
            if terminated or truncated:
                break

        if ep_len < args.max_steps:
            tagged += 1
        lengths.append(ep_len)

    lengths = np.array(lengths)
    return {
        "mode": mode,
        "success": 100.0 * tagged / args.n_episodes,
        "tagged": tagged,
        "mean_len": float(lengths.mean()),
        "median_len": float(np.median(lengths)),
        "tag_len": (float(lengths[lengths < args.max_steps].mean())
                    if tagged else float("nan")),
        "first_loose_frac": first_was_loose / args.n_episodes,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env_id", type=str, default="pdomains-ant-tag-dens-v0")
    p.add_argument("--locomotion_model", type=str,
                   default="models/ant_locomotion_policy.zip")
    p.add_argument("--locomotion_vecnorm", type=str,
                   default="models/locomotion_vecnorm.pkl")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--switch_radius", type=float, default=1.6)
    p.add_argument("--switch_steps", type=int, default=3)
    p.add_argument("--env_seed", type=int, default=5000,
                   help="Episode e uses seed env_seed+e, identical across the "
                        "three drivers, so the comparison is paired.")
    p.add_argument("--driver_seed", type=int, default=7)
    p.add_argument("--gate", type=float, default=15.0,
                   help="Pre-registered oracle-minus-anti gap, in points.")
    args = p.parse_args()

    env = gym.make(args.env_id, rendering=False)
    raw = env.unwrapped
    if not hasattr(raw, "den_positions"):
        raise SystemExit(f"{args.env_id} has no den_positions; this gate only "
                         "applies to TwinDenAntTagEnv.")

    loco_vec = DummyVecEnv([lambda: gym.make(args.env_id, rendering=False)])
    loco_vec = VecNormalize.load(args.locomotion_vecnorm, loco_vec)
    loco_vec.training = False
    loco_vec.norm_reward = False
    model = PPO.load(args.locomotion_model, device="cpu")

    print(f"Env {args.env_id}: dens at {np.round(raw.den_positions, 2).tolist()}, "
          f"r_tight={raw.den_radius_tight}, r_loose={raw.den_radius_loose}, "
          f"cap={args.max_steps}, n_episodes={args.n_episodes}")

    results = {}
    for mode in MODES:
        rng = np.random.default_rng(args.driver_seed)
        r = run_mode(mode, env, raw, loco_vec, model, args, rng)
        results[mode] = r
        print(f"  {mode:<13} success {r['tagged']:>3}/{args.n_episodes} "
              f"({r['success']:5.1f}%)  mean_len={r['mean_len']:6.1f}  "
              f"median={r['median_len']:5.1f}  when_tagged={r['tag_len']:6.1f}  "
              f"[first den was the loose one in "
              f"{100*r['first_loose_frac']:.0f}% of episodes]")

    gap = results["oracle-order"]["success"] - results["anti-order"]["success"]
    rand = results["random-order"]["success"]
    midway = (results["oracle-order"]["success"]
              + results["anti-order"]["success"]) / 2.0
    print(f"\n=== Order-value gate ===")
    print(f"oracle - anti      : {gap:+.1f} points   (pre-registered gate: "
          f">= {args.gate:.0f})  -> {'PASS' if gap >= args.gate else 'FAIL'}")
    print(f"random-order       : {rand:.1f}%  (expected roughly midway: "
          f"{midway:.1f}%)")
    env.close()
    loco_vec.close()


if __name__ == "__main__":
    main()
