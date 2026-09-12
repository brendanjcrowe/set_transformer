"""Training-progress schedules for the RL harness.

2026-09-12 (change 1d of the harness centralisation, ``refactor_plans.md`` in the parent
repo): the two schedule-string parsers, moved verbatim from
``experiments/ant_tag/4_train_rl_cgf.py`` (the ST and pool arms used the same copies).
Change 3 adds the ``Schedule`` description and the callback that interpolates schedules
over training progress and pushes the values into the env wrappers.

A schedule string is ``frac:value[,frac:value...]``: waypoints as (fraction of training
progress, value), linearly interpolated between them by the curriculum callback.
"""


def parse_curriculum(curriculum: str) -> list[tuple[float, float]]:
    """``"0:100,0.4:1.5,1:1.5"`` -> ``[(0.0, 100.0), (0.4, 1.5), (1.0, 1.5)]``.

    One value per waypoint: the visibility-radius curriculum and the evasion-scale
    schedule both use this shape.
    """
    schedule = []
    for pair in curriculum.split(","):
        frac, radius = pair.strip().split(":")
        schedule.append((float(frac), float(radius)))
    return schedule


def parse_reward_schedule(reward_schedule: str) -> list[tuple[float, ...]]:
    """``"frac:distance:entropy[:tag_bonus[:spread_gain]],..."`` -> 5-tuples.

    Missing trailing fields are 0, so every pre-2026-09-08 three- or four-field schedule
    keeps its meaning; fewer than three or more than five fields raise.
    """
    schedule = []
    for entry in reward_schedule.split(","):
        parts = [float(part) for part in entry.strip().split(":")]
        # frac:distance:entropy[:tag_bonus[:spread_gain]] -- missing trailing
        # fields are 0, so every pre-2026-09-08 schedule keeps its meaning.
        if len(parts) in (3, 4):
            parts.extend([0.0] * (5 - len(parts)))
        if len(parts) != 5:
            raise ValueError("Each reward schedule entry must have 3 to 5 values: "
                             "frac:distance:entropy[:tag_bonus[:spread_gain]]")
        schedule.append(tuple(parts))
    return schedule
