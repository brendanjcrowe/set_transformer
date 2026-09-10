"""The generated configs must match the live registry.

These files are read by humans in merge reviews and paper appendices, so a stale one is
worse than none — it looks authoritative while describing code that no longer exists.
Regenerate with `python experiments/benchmark/generate_configs.py`.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "experiments" / "benchmark" / "generate_configs.py"
CONFIGS = ROOT / "experiments" / "benchmark" / "configs"

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
yaml = pytest.importorskip("yaml")


def test_generated_configs_are_not_stale():
    proc = subprocess.run(
        [sys.executable, str(GEN), "--check"], cwd=ROOT,
        capture_output=True, text=True,
        env={"MPLBACKEND": "Agg", "QT_QPA_PLATFORM": "offscreen",
             "PATH": __import__("os").environ["PATH"],
             "HOME": __import__("os").environ.get("HOME", "/tmp")},
    )
    assert proc.returncode == 0, (
        "benchmark configs are stale — run "
        "`python experiments/benchmark/generate_configs.py`\n" + proc.stdout + proc.stderr)


def test_every_registered_method_has_a_config():
    from set_transformer.rl.benchmark.registry import METHOD_ORDER
    on_disk = {p.stem for p in (CONFIGS / "methods").glob("*.yaml")}
    assert on_disk == set(METHOD_ORDER), on_disk.symmetric_difference(set(METHOD_ORDER))


def test_every_registered_env_has_a_config():
    from set_transformer.rl.benchmark.registry import ENV_REGISTRY
    on_disk = {p.stem for p in (CONFIGS / "envs").glob("*.yaml")}
    assert on_disk == set(ENV_REGISTRY), on_disk.symmetric_difference(set(ENV_REGISTRY))


def test_configs_are_valid_yaml_mappings():
    files = sorted(CONFIGS.rglob("*.yaml"))
    assert files
    for f in files:
        doc = yaml.safe_load(f.read_text())
        assert isinstance(doc, dict) and doc, f


def test_method_configs_agree_with_the_capacity_contract():
    """The bottleneck match is the claim the comparison rests on; assert the published
    files state it, not merely the code."""
    from set_transformer.rl.benchmark.registry import (
        MATCHED_STAT_DIM, METHOD_REGISTRY, PARAM_PARITY_EXEMPT_KINDS,
        PARAM_PARITY_TOLERANCE,
    )
    counts = {}
    for f in (CONFIGS / "methods").glob("*.yaml"):
        doc = yaml.safe_load(f.read_text())
        spec = METHOD_REGISTRY[doc["method"]]
        if not spec.is_pretrainable:
            continue
        assert doc["particle_stat_dim"] == MATCHED_STAT_DIM, doc["method"]
        if spec.encoder_kind not in PARAM_PARITY_EXEMPT_KINDS:
            counts[doc["method"]] = doc["params"]["encoder_only"]
    lo, hi = min(counts.values()), max(counts.values())
    assert (hi - lo) / lo <= PARAM_PARITY_TOLERANCE, counts
