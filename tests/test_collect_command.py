"""The shared dataset collector (``rl/collect.py``, batch 7.4 of plan section 7): the
``Collection`` records both domains declare, parser assembly, placement under the root layout
(decision 1), the .npz contract through the package route, refusals, and the two entry points'
re-exports. Numerical parity with the pre-7.4 scripts is checked by
``tests/tools/pretrain_parity.py`` (plan 7e-bis, cases 14-18); the scripts' command-line
behaviour by ``test_pretrain_collect_behaviour_inventory.py`` (C1-C6) and the Odd-Even
contract tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from set_transformer.rl import collect, run_records
from set_transformer.rl.domains import get as get_domain
from set_transformer.rl.domains.base import Collection

_ST_ROOT = Path(__file__).resolve().parents[1]
ANT_TAG_COLLECT = _ST_ROOT / "experiments" / "ant_tag" / "2_collect_pf_dataset.py"
ODD_EVEN_COLLECT = _ST_ROOT / "experiments" / "odd_even" / "2_collect_pf_dataset.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Records and parser
# ---------------------------------------------------------------------------

def test_both_domains_declare_a_collection_with_their_scripts_defaults():
    ant_tag, odd_even = get_domain("ant_tag"), get_domain("odd_even")
    assert isinstance(ant_tag.collection, Collection) and isinstance(odd_even.collection, Collection)
    # The scripts' defaults for the shared flags: Ant-Tag seed 42 / 100 particles / 200 steps;
    # Odd-Even seed 0, steps and particles resolved from the variant.
    assert ant_tag.collection.defaults == {"seed": 42, "num_particles": 100, "timesteps": 200, "num_episodes": 200}
    assert odd_even.collection.defaults == {"seed": 0, "num_episodes": 200}
    a = collect.build_parser(ant_tag).parse_args([])
    assert (a.seed, a.num_particles, a.timesteps, a.num_episodes, a.variant) == (42, 100, 200, 200, "base")
    o = collect.build_parser(odd_even).parse_args([])
    assert (o.seed, o.num_particles, o.timesteps, o.variant) == (0, None, None, "oe50")


def test_the_scripts_spellings_are_aliases_of_the_shared_flags():
    parser = collect.build_parser(get_domain("ant_tag"))
    a = parser.parse_args(["--num_trajectories", "7", "--output_file", "x.npz"])
    b = parser.parse_args(["--num_episodes", "7", "--output", "x.npz"])
    assert a.num_episodes == b.num_episodes == 7 and a.output_file == b.output_file == "x.npz"
    help_text = " ".join(parser.format_help().split())
    for flag in ("--num_trajectories", "--output_file", "--pursuit_fraction", "--rebalance_no_upsample",
                 "--collapsed_threshold", "--locomotion_policy_path", "--env_id", "--particle_filter"):
        assert flag in help_text, flag
    odd_even_help = " ".join(collect.build_parser(get_domain("odd_even")).format_help().split())
    for flag in ("--rebalance_by", "--early_step", "--collapse_step", "--diffuse_ess", "--early_frac",
                 "--particle_filter", "--output"):
        assert flag in odd_even_help, flag
    assert "--pursuit_fraction" not in odd_even_help


def test_domain_is_required_and_listings_exit_early(capsys):
    with pytest.raises(SystemExit) as exc:
        collect.main(["--variant", "oe50"])
    assert exc.value.code == 2 and "--domain is required" in capsys.readouterr().err
    assert collect.main(["--domain", "odd_even", "--list_variants"]) is None
    assert "oe50_short" in capsys.readouterr().out
    assert collect.main(["--domain", "ant_tag", "--list_variants"]) is None
    assert "cdens_terminal" in capsys.readouterr().out


def test_dataset_path_helper_is_variant_first_under_the_root(tmp_path):
    assert run_records.data_dir("odd_even", "oe50", root=tmp_path) == tmp_path / "odd_even" / "oe50" / "data"
    assert run_records.dataset_path("ant_tag", "smart", root=tmp_path) == (
        tmp_path / "ant_tag" / "smart" / "data" / "smart_pf_dataset.npz")
    assert run_records.dataset_path("ant_tag", "smart", tag="v2", root=tmp_path).name == "smart_pf_dataset_v2.npz"


# ---------------------------------------------------------------------------
# The package route, end to end (tiny)
# ---------------------------------------------------------------------------

def test_odd_even_collection_lands_under_the_root_with_its_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {"sha": "test"})
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    root = tmp_path / "root"
    out = collect.main(["--domain", "odd_even", "--variant", "oe50_short", "--num_episodes", "3",
                        "--timesteps", "10", "--seed", "0", "--output_root", str(root), "--run_tag", "t"])
    assert out == root / "odd_even" / "oe50_short" / "data" / "oe50_short_pf_dataset_t.npz" and out.exists()
    assert not (tmp_path / "elsewhere" / "data").exists() and not (tmp_path / "elsewhere" / "runs").exists()
    with np.load(out, allow_pickle=True) as z:
        assert set(z.files) == {"particles", "weights", "particle_scale", "particle_centre", "steps", "metadata"}
        assert z["particles"].dtype == np.float32 and z["particles"].shape[1:] == (50, 1)
        assert z["weights"].shape == z["particles"].shape[:2] and z["steps"].dtype == np.int32
        assert float(z["particle_scale"]) == 24.5 and float(z["particle_centre"]) == 25.5
        # RAW states: the env's centring undone, so the file holds the integers 1..50.
        assert z["particles"].min() == 1.0 and z["particles"].max() == 50.0
        meta = json.loads(str(z["metadata"]))
    assert set(meta) == {"variant", "env_id", "particle_filter_class", "num_particles", "dim_particles",
                         "particle_scale", "n_dist_size", "episode_cap", "particle_centre", "step_index_min",
                         "step_index_max", "args", "git"}
    assert (meta["variant"], meta["n_dist_size"], meta["episode_cap"]) == ("oe50_short", 50, 30)   # the short variant's cap
    assert meta["particle_filter_class"] == "OddEvenExactSupportParticleFilter"
    # The script's defaults were resolved from the variant and recorded.
    assert (meta["args"]["timesteps"], meta["args"]["num_particles"], meta["args"]["seed"]) == (10, 50, 0)
    assert meta["git"] == {"sha": "test"}


def test_ant_tag_collection_with_an_explicit_output_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    out = tmp_path / "a" / "smart.npz"
    result = collect.main(["--domain", "ant_tag", "--variant", "smart", "--num_trajectories", "1", "--timesteps", "3",
                           "--num_particles", "20", "--seed", "0", "--pursuit_fraction", "0",
                           "--fully_observed_fraction", "0", "--no_rebalance", "--output_file", str(out)])
    assert result == out and out.exists()
    with np.load(out, allow_pickle=True) as z:
        assert set(z.files) == {"particles", "weights", "particle_scale", "metadata"}
        assert z["particles"].shape == (4, 20, 2) and float(z["particle_scale"]) == 4.5     # reset + 3 steps
        meta = json.loads(str(z["metadata"]))
    assert set(meta) == {"variant", "env_id", "particle_filter_class", "num_particles", "dim_particles",
                         "particle_scale", "args", "git"}
    assert meta["particle_filter_class"] == "SmartAntTagParticleFilter" and meta["args"]["num_episodes"] == 1


def test_npy_output_is_refused_before_anything_is_collected(tmp_path):
    with pytest.raises(ValueError, match="Output must be .npz"):
        collect.main(["--domain", "odd_even", "--variant", "oe50_short", "--num_episodes", "1",
                      "--output", str(tmp_path / "bad.npy")])
    assert not list(tmp_path.iterdir())


def test_a_domain_without_a_collector_is_refused(capsys):
    from dataclasses import replace
    bare = replace(get_domain("odd_even"), collection=None)
    with pytest.raises(ValueError, match="declares no dataset collector"):
        collect.build_parser(bare)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def test_entry_points_re_export_the_names_the_tests_and_diagnostics_read():
    ant_tag = _load(ANT_TAG_COLLECT, "collect_cmd_test_ant_tag")
    for name in ("_weighted_spread", "_rebalance_by_spread", "_pursuit_action", "_find_particle_filter",
                 "resolve_particle_filter", "make_ant_tag_belief_env", "get_ant_tag_arena_scale"):
        assert callable(getattr(ant_tag, name)), name
    odd_even = _load(ODD_EVEN_COLLECT, "collect_cmd_test_odd_even")
    for name in ("collect_dataset_for_test", "_rebalance", "_effective_sample_size", "make_odd_even_belief_env"):
        assert callable(getattr(odd_even, name)), name
    assert odd_even.EARLY_STEP == 3 and odd_even.COLLAPSE_STEP == 21
