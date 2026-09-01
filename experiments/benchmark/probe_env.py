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


# --- Odd-Even probe ------------------------------------------------------------
# Latent: an integer true_state in [1, n], fixed per episode. Every observation is an
# integer of true_state's own parity, so parity is knowable exactly from the first sample
# while the value stays uncertain for ~10 steps. Under the parity-gated reward a
# wrong-parity guess scores -(n-1)^2, which is what separates "informed" from "gambler":
# the gambler tracks the belief MEAN, and the mean of a comb sits between its teeth, on a
# state of the opposite parity.

_ODD_EVEN_N = 10
_ODD_EVEN_STD = 2.0


def _odd_even_grid():
    return np.arange(1, _ODD_EVEN_N + 1)


def _odd_even_log_lik(sample: int) -> np.ndarray:
    """Log P(sample | candidate) for every candidate state; -inf on parity mismatch."""
    grid = _odd_even_grid()
    out = np.full(len(grid), -np.inf)
    for i, c in enumerate(grid):
        if (int(sample) % 2) != (int(c) % 2):
            continue
        same = grid[grid % 2 == c % 2]
        dens = np.exp(-0.5 * ((same - c) / _ODD_EVEN_STD) ** 2)
        out[i] = np.log(dens[same == int(sample)][0] / dens.sum())
    return out


def _odd_even_posterior(ctx) -> np.ndarray:
    lp = ctx.get("logpost")
    if lp is None:
        return np.ones(_ODD_EVEN_N) / _ODD_EVEN_N
    post = np.exp(lp - lp.max())
    return post / post.sum()


def _odd_even_observe(obs, ctx) -> None:
    """Accumulate the exact log-posterior from this step's observation sample(s)."""
    lp = ctx.get("logpost")
    if lp is None:
        lp = np.zeros(_ODD_EVEN_N)
    for sample in np.atleast_1d(np.asarray(obs)).ravel():
        lp = lp + _odd_even_log_lik(int(round(float(sample))))
    ctx["logpost"] = lp


def _odd_even_informed(obs, ctx):
    """Track the full posterior and play its mode — always a possible state, so never
    the wrong parity. This is what a belief encoder that represents the comb enables."""
    _odd_even_observe(obs, ctx)
    return int(np.argmax(_odd_even_posterior(ctx)))


def _odd_even_gambler(obs, ctx):
    """Track only the posterior MEAN and round it — the Gaussian baseline's own decision
    rule. On a comb the mean falls between the teeth, so this lands on the impossible
    parity roughly half the time and eats the floor."""
    _odd_even_observe(obs, ctx)
    mean = float((_odd_even_grid() * _odd_even_posterior(ctx)).sum())
    return int(np.clip(round(mean), 1, _ODD_EVEN_N) - 1)


def _odd_even_oracle(obs, ctx):
    """Upper bound: predict the true state every step (reward 0)."""
    return int(ctx["latent"]) - 1


def _odd_even_staller(obs, ctx):
    """Ignore every observation and always guess the middle of the range."""
    return int(_ODD_EVEN_N // 2) - 1


# --- Multimodal Search probe ---------------------------------------------------
# Latent: a static target hidden in one of K random Gaussian modes. The belief mean is
# pinned at the origin by construction, so `gambler` -- which steers by the belief mean,
# the Gaussian baseline's own summary -- is steering by a constant that says nothing
# about the target. `informed` uses the mode LOCATIONS, which is exactly the information
# a mean cannot carry.

_MSEARCH_BASE = 7          # observation entries before the packed mode parameters


def _msearch_step_toward(obs, goal, speed_eps=1e-9):
    delta = np.asarray(goal, dtype=np.float64) - np.asarray(obs[0:2], dtype=np.float64)
    n = float(np.linalg.norm(delta))
    return (delta / n if n > speed_eps else np.zeros(2)).astype(np.float32)


def _msearch_modes(obs, ctx):
    """Mode centres from the episode's own prior, cached per episode.

    The probe runs on the BASE env, so the observation still carries the packed mode
    parameters that the benchmark masks from a learning agent.
    """
    if "modes" not in ctx:
        from set_transformer.rl.envs.multimodal_search import unpack_modes
        means, _ = unpack_modes(np.asarray(obs), ctx.get("k_max", 10))
        ctx["modes"] = [m for m in means]
    return ctx["modes"]


def _msearch_lawnmower(obs, ctx, arena=14.0, swath=3.0, tol=3.0):
    """Boustrophedon sweep of the whole arena — the best a belief-blind agent can do."""
    if "lanes" not in ctx:
        xs = np.arange(-arena + swath / 2, arena, swath)
        pts = []
        for i, x in enumerate(xs):
            ys = [-arena, arena] if i % 2 == 0 else [arena, -arena]
            pts += [np.array([x, ys[0]]), np.array([x, ys[1]])]
        ctx["lanes"], ctx["lane_i"] = pts, 0
    pts, i = ctx["lanes"], ctx["lane_i"]
    if i >= len(pts):
        return np.zeros(2, dtype=np.float32)
    if np.linalg.norm(np.asarray(obs[0:2]) - pts[i]) < tol:
        ctx["lane_i"] = i = min(i + 1, len(pts))
        if i >= len(pts):
            return np.zeros(2, dtype=np.float32)
    return _msearch_step_toward(obs, pts[i])


def _msearch_informed(obs, ctx):
    """Visit the nearest unvisited mode, then the next — needs mode LOCATIONS."""
    modes = _msearch_modes(obs, ctx)
    visited = ctx.setdefault("visited", set())
    pos = np.asarray(obs[0:2], dtype=np.float64)
    remaining = [i for i in range(len(modes)) if i not in visited]
    if not remaining:
        return _msearch_lawnmower(obs, ctx)      # prior exhausted; fall back to sweeping
    j = min(remaining, key=lambda i: np.linalg.norm(modes[i] - pos))
    if np.linalg.norm(modes[j] - pos) < 3.0:   # arrival tolerance ~ one step
        visited.add(j)
    return _msearch_step_toward(obs, modes[j])


def _msearch_gambler(obs, ctx):
    """Steer by the belief MEAN, then sweep — the Gaussian baseline's decision rule.

    The mean is the origin at t=0 by construction, and after eliminations it is the
    centroid of the surviving modes: empty space between them. So this reaches an
    uninformative point and is left with an arena-wide sweep it cannot finish in time.
    """
    if not ctx.get("mean_done"):
        modes = _msearch_modes(obs, ctx)
        centroid = np.mean(modes, axis=0) if len(modes) else np.zeros(2)
        if np.linalg.norm(np.asarray(obs[0:2]) - centroid) < 3.0:
            ctx["mean_done"] = True
        else:
            return _msearch_step_toward(obs, centroid)
    return _msearch_lawnmower(obs, ctx)


def _msearch_oracle(obs, ctx):
    """Upper bound: walk straight at the true target."""
    return _msearch_step_toward(obs, ctx["latent"])


def _msearch_staller(obs, ctx):
    """Do nothing; the episode times out."""
    return np.zeros(2, dtype=np.float32)


PROBE_REGISTRY: dict[str, ProbeSpec] = {
    "msearch": ProbeSpec(
        policies={
            "oracle": _msearch_oracle,
            "informed": _msearch_informed,
            "gambler": _msearch_gambler,
            "staller": _msearch_staller,
        },
        latent_fn=lambda env: np.asarray(env.unwrapped.target_pos, dtype=np.float64),
        notes=("Static target in one of K random Gaussian modes. The belief mean is "
               "pinned at the origin, so informed (visits modes) vs gambler (steers by "
               "the mean, then sweeps) isolates exactly what a Gaussian summary cannot "
               "represent."),
    ),
    "odd_even": ProbeSpec(
        policies={
            "oracle": _odd_even_oracle,
            "informed": _odd_even_informed,
            "gambler": _odd_even_gambler,
            "staller": _odd_even_staller,
        },
        latent_fn=lambda env: int(env.unwrapped.true_state),
        notes=("Latent is an integer in [1, n]; parity is observed exactly, the value is "
               "not. informed plays the posterior mode, gambler rounds the posterior "
               "mean -- on a comb the mean is a wrong-parity (impossible) state."),
    ),
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
