"""Audit learned policies on Counterweighted-Den Ant-Tag.

The standard evaluator reports only total success.  This diagnostic uses
matched episode seeds and additionally records the policy's first den visit,
the hidden occupied den (analysis only), spook events, and conditional
success.  A den is considered visited when the ant first enters the den's
information/spook zone; for the hard environment this is 1.4 units from the
den center (visible_radius + cden_r == spook_radius).
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
_EVAL_DIR = _ANT_TAG_DIR / "eval_scripts"
for path in (_REPO_ROOT, _ANT_TAG_DIR, _EVAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pdomains  # noqa: E402,F401 - register environments
import variants  # noqa: E402 - env/filter/cap registry
from set_transformer.rl.particle_filters.ant_tag import (  # noqa: E402
    CounterweightedDenAntTagParticleFilter,
)


def _build_env(kind, env_id, vecnormalize_path, num_particles, seed):
    # All three arms share the same dict-obs eval env; importing the arm's
    # own eval module is what makes SB3 able to unpickle that arm's feature
    # extractor class out of the saved policy_kwargs.
    module = importlib.import_module({
        "cgf": "eval_true_reward_cgf",
        "gaussian": "eval_true_reward_gaussian",
        "st": "eval_true_reward_st",
    }[kind])
    env_fn = module.make_eval_env(
        num_particles=num_particles,
        obs_mask_indices=[-2, -1],
        seed=seed,
        env_id=env_id,
        particle_filter_class=CounterweightedDenAntTagParticleFilter,
    )
    env = DummyVecEnv([env_fn])
    env = VecNormalize.load(vecnormalize_path, env)
    env.training = False
    env.norm_reward = False
    return env


def _rate(rows, predicate=lambda row: True):
    selected = [row for row in rows if predicate(row)]
    if not selected:
        return {"successes": 0, "episodes": 0, "rate": None}
    successes = sum(row["success"] for row in selected)
    return {
        "successes": int(successes),
        "episodes": len(selected),
        "rate": float(successes / len(selected)),
    }


def _fraction(rows, key, predicate=lambda row: True):
    selected = [row for row in rows if predicate(row)]
    if not selected:
        return {"count": 0, "episodes": 0, "fraction": None}
    count = sum(bool(row[key]) for row in selected)
    return {"count": int(count), "episodes": len(selected),
            "fraction": float(count / len(selected))}


def _summarize(rows):
    visited = lambda row: row["first_den"] is not None
    first_heavy = lambda row: row["first_den"] == "heavy"
    first_occupied = lambda row: row["first_den_occupied"] is True
    first_empty = lambda row: row["first_den_occupied"] is False
    occupied_heavy = lambda row: row["occupied_den"] == "heavy"
    occupied_light = lambda row: row["occupied_den"] == "light"
    spooked = lambda row: row["spooked"]
    visited_near = lambda row: row["first_near_candidate"] is not None
    near_correct = lambda row: row["first_near_is_heavy_side"] is True
    near_wrong = lambda row: row["first_near_is_heavy_side"] is False

    visit_times = [row["first_visit_step"] for row in rows
                   if row["first_visit_step"] is not None]
    successful_lengths = [row["length"] for row in rows if row["success"]]
    return {
        "overall_success": _rate(rows),
        "occupancy_heavy_fraction": _fraction(rows, "occupied_is_heavy"),
        "visited_any_den": _fraction(rows, "visited_any_den"),
        "first_heavy_given_visit": _fraction(rows, "first_is_heavy", visited),
        "first_occupied_given_visit": _fraction(
            rows, "first_den_occupied", visited),
        "visited_near_candidate": _fraction(
            rows, "visited_near_candidate"),
        "first_near_is_heavy_side": _fraction(
            rows, "first_near_is_heavy_side", visited_near),
        "success_if_occupied_heavy": _rate(rows, occupied_heavy),
        "success_if_occupied_light": _rate(rows, occupied_light),
        "success_if_first_heavy": _rate(rows, first_heavy),
        "success_if_first_occupied": _rate(rows, first_occupied),
        "success_if_first_empty": _rate(rows, first_empty),
        "success_if_first_near_correct": _rate(rows, near_correct),
        "success_if_first_near_wrong": _rate(rows, near_wrong),
        "failure_despite_first_occupied": {
            "failures": int(sum(not row["success"] for row in rows
                                if first_occupied(row))),
            "episodes": int(sum(first_occupied(row) for row in rows)),
        },
        "spook_fraction": _fraction(rows, "spooked"),
        "success_after_spook": _rate(rows, spooked),
        "success_without_spook": _rate(rows, lambda row: not spooked(row)),
        "median_first_visit_step": (
            float(np.median(visit_times)) if visit_times else None),
        "median_success_length": (
            float(np.median(successful_lengths))
            if successful_lengths else None),
    }


def _run_policy(name, kind, model_path, vecnormalize_path, args):
    env = _build_env(kind, args.env_id, vecnormalize_path,
                     args.num_particles, args.env_seed)
    model = PPO.load(model_path, env=env, device="cpu")
    raw = env.venv.envs[0].unwrapped
    visit_radius = (float(raw.cden_spook_radius)
                    if args.visit_radius is None else args.visit_radius)
    rows = []

    for episode in range(args.n_episodes):
        episode_seed = args.env_seed + episode
        # The environment has its own seeded generator; the particle filter
        # uses NumPy's module-level RNG.  Seed both so the two policy arms see
        # matched environment states and matched initial particle samples.
        # PFs now own their RNG. Seed that stream explicitly rather than
        # relying on process-global np.random state.
        env.env_method("set_particle_filter_seed", args.pf_seed + episode)
        env.seed(episode_seed)
        obs = env.reset()

        heavy = np.asarray(raw.cden_heavy_pos, dtype=np.float64).copy()
        light = np.asarray(raw.cden_light_pos, dtype=np.float64).copy()
        near_candidates = np.asarray(raw.cden_candidates[:2],
                                     dtype=np.float64).copy()
        heavy_near_index = int(np.argmin(
            np.linalg.norm(near_candidates - heavy[None, :], axis=1)))
        occupied_is_heavy = bool(raw._occupied_is_heavy)
        first_den = None
        first_visit_step = None
        first_near_candidate = None
        first_near_visit_step = None
        spooked = False
        spook_step = None
        done = False
        length = 0
        final_info = {}

        while not done and length < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            length += 1
            info = infos[0]
            final_info = info
            ant = np.asarray(raw.data.qpos[:2], dtype=np.float64)

            if first_den is None:
                distances = np.array([
                    np.linalg.norm(ant - heavy),
                    np.linalg.norm(ant - light),
                ])
                if float(distances.min()) < visit_radius:
                    first_den = "heavy" if int(distances.argmin()) == 0 else "light"
                    first_visit_step = length

            if first_near_candidate is None:
                near_distances = np.linalg.norm(
                    near_candidates - ant[None, :], axis=1)
                if float(near_distances.min()) < visit_radius:
                    first_near_index = int(near_distances.argmin())
                    first_near_candidate = (
                        "negative" if first_near_index == 0 else "positive")
                    first_near_visit_step = length

            if bool(info.get("cden_spooked", False)) and not spooked:
                spooked = True
                spook_step = length
            done = bool(dones[0])

        # Terminal commitment failures end before max_steps too, so only use
        # the old length convention when the environment has no explicit bit.
        success = bool(final_info.get(
            "is_success", done and length < args.max_steps))
        occupied_den = "heavy" if occupied_is_heavy else "light"
        first_den_occupied = (
            None if first_den is None else first_den == occupied_den)
        first_near_is_heavy_side = (
            None if first_near_candidate is None
            else first_near_index == heavy_near_index)
        rows.append({
            "episode": episode,
            "env_seed": episode_seed,
            "occupied_den": occupied_den,
            "occupied_is_heavy": occupied_is_heavy,
            "first_den": first_den,
            "visited_any_den": first_den is not None,
            "first_is_heavy": first_den == "heavy",
            "first_den_occupied": first_den_occupied,
            "first_visit_step": first_visit_step,
            "first_near_candidate": first_near_candidate,
            "visited_near_candidate": first_near_candidate is not None,
            "first_near_is_heavy_side": first_near_is_heavy_side,
            "first_near_visit_step": first_near_visit_step,
            "spooked": spooked,
            "spook_step": spook_step,
            "success": success,
            "termination_reason": final_info.get("termination_reason"),
            "length": length,
        })

        if (episode + 1) % 50 == 0:
            print(f"{name}: {episode + 1}/{args.n_episodes} episodes")

    env.close()
    return {
        "name": name,
        "kind": kind,
        "model_path": model_path,
        "vecnormalize_path": vecnormalize_path,
        "visit_radius": visit_radius,
        "summary": _summarize(rows),
        "episodes": rows,
    }


def _paired_summary(cgf_rows, gaussian_rows, name_a="cgf", name_b="gaussian"):
    if len(cgf_rows) != len(gaussian_rows):
        raise ValueError("Policy audits do not contain the same episode count")
    both = cgf_only = gaussian_only = neither = 0
    first_agree = 0
    first_comparable = 0
    for cgf, gaussian in zip(cgf_rows, gaussian_rows):
        if cgf["env_seed"] != gaussian["env_seed"]:
            raise ValueError("Policy audits are not seed-aligned")
        cs, gs = cgf["success"], gaussian["success"]
        both += int(cs and gs)
        cgf_only += int(cs and not gs)
        gaussian_only += int(gs and not cs)
        neither += int(not cs and not gs)
        if cgf["first_den"] is not None and gaussian["first_den"] is not None:
            first_comparable += 1
            first_agree += int(cgf["first_den"] == gaussian["first_den"])
    return {
        "both_succeed": both,
        f"{name_a}_only_succeeds": cgf_only,
        f"{name_b}_only_succeeds": gaussian_only,
        "neither_succeeds": neither,
        "first_den_agreement": (
            float(first_agree / first_comparable) if first_comparable else None),
        "first_den_comparable_episodes": first_comparable,
    }


def _format_rate(value):
    if value["rate"] is None:
        return "n/a"
    return f"{100 * value['rate']:.1f}% ({value['successes']}/{value['episodes']})"


def _format_fraction(value):
    if value["fraction"] is None:
        return "n/a"
    return f"{100 * value['fraction']:.1f}% ({value['count']}/{value['episodes']})"


def _print_summary(result):
    summary = result["summary"]
    print(f"\n=== {result['name']} ===")
    print(f"success                  {_format_rate(summary['overall_success'])}")
    print(f"target occupied heavy    {_format_fraction(summary['occupancy_heavy_fraction'])}")
    print(f"visited any den          {_format_fraction(summary['visited_any_den'])}")
    print(f"first den = heavy        {_format_fraction(summary['first_heavy_given_visit'])}")
    print(f"first den = occupied     {_format_fraction(summary['first_occupied_given_visit'])}")
    print(f"visited a +/-h candidate {_format_fraction(summary['visited_near_candidate'])}")
    print(f"first +/-h = heavy side  {_format_fraction(summary['first_near_is_heavy_side'])}")
    print(f"success | occupied heavy {_format_rate(summary['success_if_occupied_heavy'])}")
    print(f"success | occupied light {_format_rate(summary['success_if_occupied_light'])}")
    print(f"success | first occupied {_format_rate(summary['success_if_first_occupied'])}")
    print(f"success | first empty    {_format_rate(summary['success_if_first_empty'])}")
    print(f"success | near correct   {_format_rate(summary['success_if_first_near_correct'])}")
    print(f"success | near wrong     {_format_rate(summary['success_if_first_near_wrong'])}")
    print(f"spooked                  {_format_fraction(summary['spook_fraction'])}")
    print(f"success | spooked        {_format_rate(summary['success_after_spook'])}")
    print(f"success | not spooked    {_format_rate(summary['success_without_spook'])}")
    print(f"median first-visit step  {summary['median_first_visit_step']}")
    print(f"median successful length {summary['median_success_length']}")


def _checkpoint_num_particles(model_path) -> int | None:
    """Particle-set size recorded in a saved policy's observation space."""
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        data, _, _ = load_from_zip_file(str(model_path), load_data=True,
                                        device="cpu", print_system_info=False)
        return int(data["observation_space"]["particles"].shape[0])
    except Exception:  # noqa: BLE001 - best-effort default, never fatal
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Every arm is optional: audit any subset of {cgf, gaussian, st}, and a
    # matched-episode pairing is emitted for each pair actually present.
    # Passing --cgf_* and --gaussian_* reproduces the original behavior.
    parser.add_argument("--cgf_model")
    parser.add_argument("--cgf_vecnormalize")
    parser.add_argument("--gaussian_model")
    parser.add_argument("--gaussian_vecnormalize")
    parser.add_argument("--st_model")
    parser.add_argument("--st_vecnormalize")
    variants.add_variant_argument(parser, default="cdens_hard")
    parser.add_argument(
        "--env_id", default=None,
        help="Override the variant's env id (rarely needed).",
    )
    parser.add_argument("--n_episodes", type=int, default=300)
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Defaults to the variant's registered episode cap.",
    )
    parser.add_argument(
        "--num_particles", type=int, default=None,
        help="Defaults to the count recorded in the checkpoints' "
             "observation space; all arms must agree.",
    )
    parser.add_argument("--env_seed", type=int, default=42000)
    parser.add_argument("--pf_seed", type=int, default=52000)
    parser.add_argument("--visit_radius", type=float, default=None)
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args()
    if args.list_variants:
        variants.print_variants()
        return

    variant = variants.resolve(args.variant)
    variants.warn_if_override_contradicts(args.variant, env_id=args.env_id)
    if args.env_id is None:
        args.env_id = variant.env_id
    if args.max_steps is None:
        args.max_steps = variants.episode_cap(args.variant)
    print(f"Variant: {args.variant} | env: {args.env_id} | "
          f"episode cap: {args.max_steps}")

    # Every arm's env must use the particle count its policy was trained with,
    # and the arms are compared on matched episodes, so they must all agree.
    counts = {}
    for kind, model in (("cgf", args.cgf_model), ("gaussian", args.gaussian_model),
                        ("st", args.st_model)):
        if model is None:
            continue
        n = _checkpoint_num_particles(model)
        if n is not None:
            counts[kind] = n
    if len(set(counts.values())) > 1:
        parser.error(f"arms were trained with different particle counts: "
                     f"{counts}; they cannot be compared on matched episodes")
    if args.num_particles is None:
        args.num_particles = next(iter(counts.values()), 100)
    elif counts and args.num_particles != next(iter(counts.values())):
        parser.error(f"--num_particles {args.num_particles} contradicts the "
                     f"checkpoints ({counts}). Omit the flag.")
    print(f"Particles per set: {args.num_particles}")

    arms = {}
    for kind, label, model, vecnormalize in (
        ("cgf", "CGF", args.cgf_model, args.cgf_vecnormalize),
        ("gaussian", "Gaussian", args.gaussian_model, args.gaussian_vecnormalize),
        ("st", "SetTransformer", args.st_model, args.st_vecnormalize),
    ):
        if model is None:
            continue
        if vecnormalize is None:
            parser.error(f"--{kind}_model given without --{kind}_vecnormalize")
        arms[kind] = _run_policy(label, kind, model, vecnormalize, args)
    if not arms:
        parser.error("give at least one of --cgf_model / --gaussian_model / "
                     "--st_model")

    result = {
        "variant": args.variant,
        "env_id": args.env_id,
        "n_episodes": args.n_episodes,
        "max_steps": args.max_steps,
        "env_seed": args.env_seed,
        "pf_seed": args.pf_seed,
    }
    result.update(arms)
    kinds = list(arms)
    for i, kind_a in enumerate(kinds):
        for kind_b in kinds[i + 1:]:
            key = ("paired" if {kind_a, kind_b} == {"cgf", "gaussian"}
                   else f"paired_{kind_a}_{kind_b}")
            result[key] = _paired_summary(
                arms[kind_a]["episodes"], arms[kind_b]["episodes"],
                name_a=kind_a, name_b=kind_b)

    for arm in arms.values():
        _print_summary(arm)
    for key, value in result.items():
        if not key.startswith("paired"):
            continue
        print(f"\n=== Matched-episode comparison ({key}) ===")
        for stat, stat_value in value.items():
            print(f"{stat:30s} {stat_value}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nSaved detailed audit to {args.json_out}")


if __name__ == "__main__":
    main()
