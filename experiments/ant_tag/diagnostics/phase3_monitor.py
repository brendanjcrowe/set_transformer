"""
Phase 3 monitor for the stab_v1 CGF runs (per
.claude/20260730_2337_smart_ant_tag_rl_training_fix_plan.md Section 3.3).

Polls each run's logs/evaluations.npz + PID liveness. Emits one line per:
  - crossing a 25/60/90% checkpoint (approx_kl/clip_fraction tail + success-so-far)
  - an abort-criteria trip (kl>0.10 past 1.5M steps, or success pinned at 0.0
    for >1M consecutive steps after the 50% full-difficulty mark)
  - each run's process exiting

Runs until both PIDs have exited, then prints DONE_ALL and exits.
"""

import re
import sys
import time

import numpy as np

RUNS = {
    "seed0": {
        "pid": 1734885,
        "dir": "runs/ant_tag_cgf_smart/20260731_004055_seed0_stab_v1",
        "log": "phase3_logs/cgf_seed0_launch.log",
    },
    "seed7": {
        "pid": 1736530,
        "dir": "runs/ant_tag_cgf_smart/20260731_004245_seed7_stab_v1",
        "log": "phase3_logs/cgf_seed7_launch.log",
    },
}
TOTAL_TIMESTEPS = 6_000_000
CHECKPOINT_FRACS = [0.25, 0.60, 0.90]

KL_RE = re.compile(r"approx_kl\s*\|\s*([\d.eE+-]+)")
CLIP_RE = re.compile(r"clip_fraction\s*\|\s*([\d.eE+-]+)")
STEP_RE = re.compile(r"total_timesteps\s*\|\s*(\d+)")


def pid_alive(pid):
    try:
        import os
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def tail_lines(path, n=4000):
    try:
        with open(path, "r", errors="ignore") as f:
            return f.readlines()[-n:]
    except FileNotFoundError:
        return []


def current_progress(lines):
    for line in reversed(lines):
        m = STEP_RE.search(line)
        if m:
            return int(m.group(1))
    return 0


def recent_kl_clip(lines, k=5):
    kls, clips = [], []
    for line in reversed(lines):
        if len(kls) >= k and len(clips) >= k:
            break
        mk = KL_RE.search(line)
        if mk and len(kls) < k:
            kls.append(float(mk.group(1)))
        mc = CLIP_RE.search(line)
        if mc and len(clips) < k:
            clips.append(float(mc.group(1)))
    return list(reversed(kls)), list(reversed(clips))


def success_trajectory(run_dir):
    try:
        d = np.load(f"{run_dir}/logs/evaluations.npz")
    except FileNotFoundError:
        return None, None
    succ = (d["ep_lengths"] < 400).mean(axis=1)
    return d["timesteps"], succ


def main():
    state = {name: {"done_fracs": set(), "aborted": False, "exited": False} for name in RUNS}
    zero_streak_start = {name: None for name in RUNS}

    while True:
        all_exited = True
        for name, cfg in RUNS.items():
            s = state[name]
            if s["exited"]:
                continue
            alive = pid_alive(cfg["pid"])
            lines = tail_lines(cfg["log"])
            step = current_progress(lines)
            frac = step / TOTAL_TIMESTEPS

            for ckpt in CHECKPOINT_FRACS:
                if frac >= ckpt and ckpt not in s["done_fracs"]:
                    s["done_fracs"].add(ckpt)
                    kls, clips = recent_kl_clip(lines)
                    ts, succ = success_trajectory(cfg["dir"])
                    succ_str = (
                        f"last-5 evals: {np.round(succ[-5:], 3)} mean={succ[-5:].mean():.3f}"
                        if succ is not None and len(succ) >= 1
                        else "no evals yet"
                    )
                    print(
                        f"[CHECKPOINT] {name} crossed {int(ckpt*100)}% "
                        f"(step={step}): last-5 approx_kl={kls}, "
                        f"clip_fraction={clips}; {succ_str}",
                        flush=True,
                    )

            # Abort criteria (Phase 3.3): only meaningful past 1.5M steps.
            if step >= 1_500_000 and not s["aborted"]:
                kls, _ = recent_kl_clip(lines)
                if kls and min(kls) > 0.10:
                    print(
                        f"[ABORT-CANDIDATE] {name} at step={step}: last-5 "
                        f"approx_kl={kls} all > 0.10. Consider killing and "
                        f"relaunching with --learning_rate 1.5e-4, tag stab_v2_lowlr.",
                        flush=True,
                    )
                    s["aborted"] = True

                ts, succ = success_trajectory(cfg["dir"])
                if ts is not None and frac >= 0.5:
                    post50_mask = ts >= 0.5 * TOTAL_TIMESTEPS
                    post50_succ = succ[post50_mask]
                    post50_ts = ts[post50_mask]
                    if len(post50_succ) > 0:
                        if np.all(post50_succ == 0.0):
                            if zero_streak_start[name] is None:
                                zero_streak_start[name] = post50_ts[0]
                            elif step - zero_streak_start[name] > 1_000_000:
                                print(
                                    f"[ABORT-CANDIDATE] {name} at step={step}: "
                                    f"eval success pinned at 0.0 for >1M steps "
                                    f"since step={zero_streak_start[name]} after "
                                    f"the 50% mark.",
                                    flush=True,
                                )
                                s["aborted"] = True
                        else:
                            zero_streak_start[name] = None

            if not alive and not s["exited"]:
                s["exited"] = True
                ts, succ = success_trajectory(cfg["dir"])
                succ_str = (
                    f"last-5 evals: {np.round(succ[-5:], 3)} mean={succ[-5:].mean():.3f}, best={succ.max():.3f}"
                    if succ is not None
                    else "no evals"
                )
                print(f"[EXITED] {name} process ended at reported step={step}. {succ_str}", flush=True)

            if not s["exited"]:
                all_exited = False

        if all_exited:
            print("DONE_ALL", flush=True)
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
