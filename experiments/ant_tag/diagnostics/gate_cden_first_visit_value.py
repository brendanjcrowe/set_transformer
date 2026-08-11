"""
First-visit-value gate for Counterweighted-Den Ant-Tag: is committing to the
HEAVY den first actually worth points?

This replaces Twin-Den's order-value gate, which failed (6 +- 5 points vs a
15-point gate) because there the decision bit was independent of occupancy,
so the cost of a wrong first commitment structurally cancelled. Here the
mirror bit IS the occupancy prior: the heavy-near den is occupied with
probability w = f/(h+f) ~ 0.74 and the light-far den with 1-w ~ 0.26, so
"which side is heavy" is decision-identical to "go there first".

Three scripted drivers, no particle filter anywhere, each driving the ant with
the Phase-2 locomotion policy via waypoint injection into obs[-2:] (the
gate_den_order_value.py harness):

  oracle  first waypoint = env.unwrapped.cden_heavy_pos (ground truth,
          deliberately -- the upper bound on what knowing the mirror bit buys)
  anti    first waypoint = cden_light_pos (the systematically wrong choice)
  coin    uniform coin flip per episode (what an uninformed policy can do)

In all three: the driver switches to the OTHER den center when the spook
alarm fires (info["cden_spooked"] flips), or after --switch_steps consecutive
steps within --switch_radius of the first center with no visual contact.
Whenever the true target is within visible_radius, the waypoint becomes the
true target -- that is what any policy would see and chase.

Also reports mean per-step displacement toward the current waypoint, which
doubles as the locomotion-transfer check on the enlarged arena.

Pre-registered gate: oracle - anti >= 25 points AND oracle - coin >= 12
points, on at least one of the two registered env ids.

Usage:
    python3 diagnostics/gate_cden_first_visit_value.py --env_id pdomains-ant-tag-cdens-v0 \
        --max_steps 300 --n_episodes 100
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

import pdomains  # noqa: F401 - registers pdomains-ant-tag-cdens-v0

MODES = ("oracle", "anti", "coin")


def run_mode(mode, env, raw, loco_vec, model, args, rng):
    tagged = 0
    lengths = []
    first_was_heavy = 0
    first_was_occupied = 0
    tagged_by_first_choice = {True: [0, 0], False: [0, 0]}  # correct -> [tag, n]
    progress = []
    n_spooked = 0

    for ep in range(args.n_episodes):
        env.reset(seed=args.env_seed + ep)
        heavy = np.asarray(raw.cden_heavy_pos, dtype=np.float64)
        light = np.asarray(raw.cden_light_pos, dtype=np.float64)
        occupied_heavy = bool(raw._occupied_is_heavy)

        if mode == "oracle":
            go_heavy_first = True
        elif mode == "anti":
            go_heavy_first = False
        else:
            go_heavy_first = bool(rng.integers(2))
        first_was_heavy += int(go_heavy_first)
        first_correct = (go_heavy_first == occupied_heavy)
        first_was_occupied += int(first_correct)

        order = [heavy, light] if go_heavy_first else [light, heavy]
        leg = 0
        stalled = 0
        ep_len = 0
        was_spooked = False

        for t in range(args.max_steps):
            ant = np.asarray(raw.data.qpos[:2], dtype=np.float64).copy()
            true_target = np.asarray(raw.get_target_pos(), dtype=np.float64)
            visible = np.linalg.norm(ant - true_target) < raw.visible_radius

            den_wp = order[leg]
            if visible:
                waypoint = true_target
                stalled = 0
            else:
                waypoint = den_wp
                # Searched this den long enough with nothing to show for it?
                if leg == 0 and np.linalg.norm(ant - den_wp) < args.switch_radius:
                    stalled += 1
                    if stalled >= args.switch_steps:
                        leg = 1
                        stalled = 0

            d0 = np.linalg.norm(ant - waypoint)
            raw_obs = np.concatenate(
                [raw.data.qpos, raw.data.qvel, waypoint]).astype(np.float32)
            action, _ = model.predict(
                loco_vec.normalize_obs(raw_obs[None, :]), deterministic=True)
            _, _, terminated, truncated, info = env.step(action[0])
            ep_len += 1
            ant2 = np.asarray(raw.data.qpos[:2], dtype=np.float64)
            progress.append(d0 - np.linalg.norm(ant2 - waypoint))

            # The alarm is the definitive "this den is empty" signal: commit
            # to the other one immediately.
            if info.get("cden_spooked", False) and not was_spooked:
                was_spooked = True
                leg = 1
                stalled = 0

            if terminated or truncated:
                break

        n_spooked += int(was_spooked)
        if ep_len < args.max_steps:
            tagged += 1
            tagged_by_first_choice[first_correct][0] += 1
        tagged_by_first_choice[first_correct][1] += 1
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
        "first_heavy_frac": first_was_heavy / args.n_episodes,
        "first_correct_frac": first_was_occupied / args.n_episodes,
        "succ_if_first_correct": (
            100.0 * tagged_by_first_choice[True][0]
            / max(tagged_by_first_choice[True][1], 1)),
        "n_first_correct": tagged_by_first_choice[True][1],
        "succ_if_first_wrong": (
            100.0 * tagged_by_first_choice[False][0]
            / max(tagged_by_first_choice[False][1], 1)),
        "n_first_wrong": tagged_by_first_choice[False][1],
        "progress": float(np.mean(progress)),
        "spook_frac": n_spooked / args.n_episodes,
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env_id", type=str, default="pdomains-ant-tag-cdens-v0")
    p.add_argument("--locomotion_model", type=str,
                   default="models/ant_locomotion_policy.zip")
    p.add_argument("--locomotion_vecnorm", type=str,
                   default="models/locomotion_vecnorm.pkl")
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--switch_radius", type=float, default=1.6)
    p.add_argument("--switch_steps", type=int, default=3)
    p.add_argument("--env_seed", type=int, default=5000,
                   help="Episode e uses seed env_seed+e, identical across the "
                        "three drivers, so the comparison is paired.")
    p.add_argument("--driver_seed", type=int, default=7)
    p.add_argument("--gate_anti", type=float, default=25.0,
                   help="Pre-registered oracle-minus-anti gap, in points.")
    p.add_argument("--gate_coin", type=float, default=12.0,
                   help="Pre-registered oracle-minus-coin gap, in points.")
    p.add_argument("--loco_gate", type=float, default=0.12,
                   help="Locomotion-transfer threshold, u/step toward the "
                        "current waypoint.")
    args = p.parse_args()

    env = gym.make(args.env_id, rendering=False)
    raw = env.unwrapped
    if not hasattr(raw, "cden_heavy_pos"):
        raise SystemExit(f"{args.env_id} has no cden_heavy_pos; this gate only "
                         "applies to CounterweightedDenAntTagEnv.")

    loco_vec = DummyVecEnv([lambda: gym.make(args.env_id, rendering=False)])
    loco_vec = VecNormalize.load(args.locomotion_vecnorm, loco_vec)
    loco_vec.training = False
    loco_vec.norm_reward = False
    model = PPO.load(args.locomotion_model, device="cpu")

    print(f"Env {args.env_id}: h={raw.cden_h}, f={raw.cden_f}, r={raw.cden_r}, "
          f"w_heavy={raw.cden_w_heavy:.4f}, vis={raw.visible_radius}, "
          f"spook={'on' if raw.cden_spook_enabled else 'OFF'} "
          f"(radius {raw.cden_spook_radius}), cap={args.max_steps}, "
          f"n_episodes={args.n_episodes}")

    results = {}
    for mode in MODES:
        rng = np.random.default_rng(args.driver_seed)
        r = run_mode(mode, env, raw, loco_vec, model, args, rng)
        results[mode] = r
        print(f"  {mode:<7} success {r['tagged']:>3}/{args.n_episodes} "
              f"({r['success']:5.1f}%)  mean_len={r['mean_len']:6.1f}  "
              f"median={r['median_len']:5.1f}  when_tagged={r['tag_len']:6.1f}")
        print(f"          first=heavy in {100*r['first_heavy_frac']:.0f}% of eps; "
              f"first den was the OCCUPIED one in "
              f"{100*r['first_correct_frac']:.0f}%; "
              f"success | first correct = {r['succ_if_first_correct']:.1f}% "
              f"(n={r['n_first_correct']}), | first wrong = "
              f"{r['succ_if_first_wrong']:.1f}% (n={r['n_first_wrong']}); "
              f"alarm fired in {100*r['spook_frac']:.0f}%")

    gap_anti = results["oracle"]["success"] - results["anti"]["success"]
    gap_coin = results["oracle"]["success"] - results["coin"]["success"]
    midway = (results["oracle"]["success"] + results["anti"]["success"]) / 2.0
    loco = float(np.mean([results[m]["progress"] for m in MODES]))

    print("\n=== First-visit-value gate (plan sec. 3 Step 7 / sec. 6) ===")
    ok_anti = gap_anti >= args.gate_anti
    ok_coin = gap_coin >= args.gate_coin
    print(f"oracle - anti  : {gap_anti:+.1f} points   (gate >= "
          f"{args.gate_anti:.0f})  -> {'PASS' if ok_anti else 'FAIL'}")
    print(f"oracle - coin  : {gap_coin:+.1f} points   (gate >= "
          f"{args.gate_coin:.0f})  -> {'PASS' if ok_coin else 'FAIL'}")
    print(f"coin           : {results['coin']['success']:.1f}%  "
          f"(expected roughly midway: {midway:.1f}%)")
    print(f"OVERALL        : {'PASS' if (ok_anti and ok_coin) else 'FAIL'}")
    print(f"\nlocomotion transfer (mean per-step progress toward waypoint): "
          f"{loco:.4f} u/step   (threshold >= {args.loco_gate})  -> "
          f"{'PASS' if loco >= args.loco_gate else 'FAIL'}")
    env.close()
    loco_vec.close()


if __name__ == "__main__":
    main()
