"""Pre-flight check: does using the hidden state actually pay in this environment?

Run this **before** sweeping any new env. A POMDP only discriminates belief encoders if
the return-maximizing policy has to *use* the latent. If information costs more than it is
worth, RL converges to a latent-ignoring policy and every encoder scores the same — the
benchmark cell is vacuous even though nothing errors.

This is not hypothetical: stock ``pdomains-car-flag-v0`` pays -1/step while the priest
detour costs ~27 steps and the information is worth only 3, so PPO converged to a coin-flip
gambler (success 0.55). Note that **potential-based shaping cannot repair this** — PBRS
preserves the optimal policy by construction — so the task reward itself must change.

The probe compares hand-coded reference policies on the true reward:

    oracle    — knows the latent for free (upper bound; not achievable by any agent)
    informed  — gathers information, then acts on it (what we *want* RL to learn)
    gambler   — ignores the latent and commits immediately
    staller   — does nothing (times out)

Verdicts:
    informed > gambler  => information pays; the env can discriminate belief encoders
    gambler  > staller  => committing beats stalling; no degenerate do-nothing optimum

Examples
--------
    python experiments/benchmark/probe_env.py --env car_flag
    python experiments/benchmark/probe_env.py --env car_flag --n_episodes 500
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# `set_transformer` is a namespace package and the editable install may point at a
# different checkout; ensure THIS repo's package root is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import numpy as np

from set_transformer.rl.benchmark.registry import get_env_spec
from set_transformer.rl.benchmark.results import bootstrap_mean_ci

#: A probe policy maps ``(observation, context)`` to an action. ``context`` persists for
#: the episode and carries ``"latent"`` (for oracle policies) plus whatever the policy
#: chooses to remember (e.g. a revealed direction).
ProbePolicy = Callable[[np.ndarray, dict], np.ndarray]


@dataclass
class ProbeSpec:
    """Hand-coded reference policies for one environment."""

    policies: dict[str, ProbePolicy] = field(default_factory=dict)
    #: Reads the true latent from a freshly-reset env, for the oracle policy only.
    latent_fn: Optional[Callable[[gym.Env], object]] = None
    notes: str = ""


# --- Car-Flag probe ------------------------------------------------------------
# Latent: heaven side in {+1, -1}. It is revealed by obs[2] ("direction") only inside the
# priest region, so "informed" must detour there first.

def _car_flag_gambler(obs, ctx):
    """Ignore the priest and drive straight at the +1 flag (heaven 50% of the time)."""
    return np.array([1.0])


def _car_flag_informed(obs, ctx):
    """Drive toward the priest; once direction is read, commit to the true heaven."""
    direction = float(obs[2])
    if direction != 0.0:
        ctx["revealed"] = direction
    revealed = ctx.get("revealed", 0.0)
    if revealed != 0.0:
        return np.array([float(np.sign(revealed))])
    return np.array([1.0])  # still seeking the priest (it lies to the right of start)


def _car_flag_oracle(obs, ctx):
    """Upper bound: drive at the true heaven from step 0, paying no detour."""
    return np.array([float(np.sign(ctx["latent"]))])


def _car_flag_staller(obs, ctx):
    """Do nothing; the episode times out."""
    return np.array([0.0])


PROBE_REGISTRY: dict[str, ProbeSpec] = {
    "car_flag": ProbeSpec(
        policies={
            "oracle": _car_flag_oracle,
            "informed": _car_flag_informed,
            "gambler": _car_flag_gambler,
            "staller": _car_flag_staller,
        },
        latent_fn=lambda env: float(env.unwrapped.heaven_position),
        notes="Latent is a single bit (heaven side), revealed by obs[2] at the priest.",
    ),
}


def run_policy(env_spec, probe: ProbeSpec, policy: ProbePolicy,
               n_episodes: int, seed: int):
    """Roll out one hand-coded policy; return per-episode returns, lengths, successes."""
    env = env_spec.make_base_env(seed=seed)
    returns, lengths, successes = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        # The latent is redrawn on reset, so read it after resetting.
        ctx = {"latent": probe.latent_fn(env) if probe.latent_fn else None}
        total, steps = 0.0, 0
        while True:
            obs, reward, terminated, truncated, _ = env.step(policy(obs, ctx))
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        returns.append(total)
        lengths.append(steps)
        if env_spec.success_fn is not None:
            successes.append(bool(env_spec.success_fn(total, steps)))
    env.close()
    return np.array(returns), np.array(lengths), np.array(successes, dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, help=f"env with a probe: {sorted(PROBE_REGISTRY)}")
    p.add_argument("--n_episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.env not in PROBE_REGISTRY:
        raise SystemExit(
            f"No probe registered for '{args.env}'. Registered: {sorted(PROBE_REGISTRY)}.\n"
            "Add a ProbeSpec (oracle / informed / gambler / staller) for it before sweeping."
        )
    env_spec = get_env_spec(args.env)
    probe = PROBE_REGISTRY[args.env]

    print(f"\nProbe: {args.env}  ({args.n_episodes} episodes/policy, true reward)")
    if probe.notes:
        print(f"  {probe.notes}")
    print(f"\n{'policy':10s} {'return':>22s} {'ep_len':>8s} {'success':>9s}")

    stats = {}
    for name, policy in probe.policies.items():
        R, L, S = run_policy(env_spec, probe, policy, args.n_episodes, args.seed)
        mean, lo, hi = bootstrap_mean_ci(R)
        stats[name] = (mean, lo, hi)
        success = f"{S.mean():9.2f}" if S.size else f"{'n/a':>9s}"
        print(f"{name:10s} {mean:8.2f} [{lo:7.2f},{hi:7.2f}] {L.mean():8.1f} {success}")

    print()
    ok = True
    if "informed" in stats and "gambler" in stats:
        inf_m, inf_lo, _ = stats["informed"]
        gam_m, _, gam_hi = stats["gambler"]
        gap = inf_m - gam_m
        # Require CI separation, not just a higher mean.
        if inf_lo > gam_hi:
            print(f"  PASS  information pays: informed beats gambler by {gap:+.2f} (CIs disjoint)")
        else:
            ok = False
            print(f"  FAIL  information does NOT pay: informed - gambler = {gap:+.2f} "
                  "(CIs overlap or negative).\n"
                  "        The optimal policy can ignore the latent, so every belief encoder\n"
                  "        will score alike. Fix the TASK REWARD -- potential-based shaping\n"
                  "        cannot help, it preserves the optimal policy by construction.")
    if "gambler" in stats and "staller" in stats:
        if stats["gambler"][0] > stats["staller"][0]:
            print(f"  PASS  committing beats stalling by "
                  f"{stats['gambler'][0] - stats['staller'][0]:+.2f}")
        else:
            ok = False
            print("  FAIL  stalling beats committing: the agent is rewarded for timing out.")
    if "oracle" in stats and "informed" in stats:
        print(f"  INFO  cost of gathering information (oracle - informed) = "
              f"{stats['oracle'][0] - stats['informed'][0]:+.2f}")

    print(f"\n  => {args.env} is {'READY to sweep' if ok else 'NOT ready -- fix the reward first'}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
