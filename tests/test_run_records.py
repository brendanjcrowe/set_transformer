"""set_transformer.rl.run_records: the shared run bookkeeping (change 1d of the harness
centralisation) and the root-level output layout (plan section 2b).

How the scripts USE the moved functions is pinned by tests/test_harness_behaviour_inventory.py
(run-dir shape through each script, run_config.json contents, the monkeypatched training
drives). This file tests the package module directly and, for the layout functions, every
branch of the output-root resolution.
"""

import importlib
import re
import sys
from pathlib import Path

import pytest

from set_transformer.rl import run_records as rr

_ST_ROOT = Path(__file__).resolve().parents[1]
_ANT_TAG_DIR = _ST_ROOT / "experiments" / "ant_tag"


# ---------------------------------------------------------------------------
# The moved functions, called directly
# ---------------------------------------------------------------------------

def test_default_run_dir_is_cwd_relative_with_a_sanitised_tag():
    path = rr.default_run_dir(3, "ant_tag_cgf_smart", run_tag="6M vis/0.2")
    assert re.fullmatch(r"runs/ant_tag_cgf_smart/\d{8}_\d{6}_seed3_6M_vis_0\.2", path), path
    assert re.fullmatch(r"runs/x/\d{8}_\d{6}_seed0", rr.default_run_dir(0, "x"))


def test_run_status_roundtrip_and_lookup_one_directory_up(tmp_path):
    model = tmp_path / "models" / "cgf_agent.zip"
    model.parent.mkdir()
    status = rr.write_run_status(str(model), completed=False, error=KeyboardInterrupt("stop"),
                                 timesteps=1200, total_timesteps=3000)
    assert status["status"] == "failed" and status["error"] == "KeyboardInterrupt: stop"
    assert status["timesteps"] == 1200 and status["total_timesteps"] == 3000
    assert (model.parent / rr.RUN_STATUS_FILENAME).is_file()
    assert rr.read_run_status(str(model)) == status
    nested = model.parent / "best_model" / "best_model.zip"      # resolves to the same run
    nested.parent.mkdir()
    assert rr.read_run_status(str(nested)) == status
    assert rr.read_run_status(str(tmp_path / "elsewhere" / "x.zip")) is None
    done = rr.write_run_status(str(model), completed=True, error=None, timesteps=3000, total_timesteps=3000)
    assert done["status"] == "completed" and done["error"] is None


def test_git_provenance_names_both_repos_and_never_raises():
    provenance = rr.git_provenance()
    assert set(provenance) == {"set_transformer", "pomdp-domains"}
    for entry in provenance.values():
        assert "path" in entry
        assert ("head" in entry and "dirty" in entry and "status" in entry) or "error" in entry
    # This checkout: the path derived from the package file is the set_transformer repo.
    assert Path(provenance["set_transformer"]["path"]) == _ST_ROOT


def test_thread_settings_record_the_env_vars_and_torchs_count(monkeypatch):
    """PITFALLS section 12, item 3: a CPU run reproduces to the digit only at its thread
    count, so the record says what it was. Unset variables are None, not missing."""
    import torch
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    threads = rr.thread_settings()
    assert set(threads) == {"omp", "mkl", "openblas", "torch"}
    assert threads["omp"] == "8" and threads["mkl"] is None and threads["openblas"] is None
    assert threads["torch"] == torch.get_num_threads() and isinstance(threads["torch"], int)


def test_scripts_hand_back_the_package_functions_under_their_old_names():
    """The scripts import the moved functions under their historical underscore names; the
    tests that monkeypatch `module._default_run_dir` etc. rely on those attributes existing,
    and the arms alias them off 4_train_rl_cgf -- so they must be the SAME objects."""
    pytest.importorskip("mujoco", reason="the Ant-Tag scripts import the env module")
    from set_transformer.rl import curriculum
    preexisting = set(sys.modules)
    sys.path.insert(0, str(_ANT_TAG_DIR))
    try:
        cgf = importlib.import_module("4_train_rl_cgf")
        st = importlib.import_module("4_train_rl_st")
        gauss = importlib.import_module("4_train_rl_gaussian")
        pool = importlib.import_module("4_train_rl_pool")
        assert cgf._git_provenance is rr.git_provenance is st._git_provenance is pool._git_provenance
        assert cgf._write_run_config is rr.write_run_config is gauss._write_run_config
        assert cgf._tee_stdout_stderr is rr.tee_stdout_stderr
        assert cgf._parse_curriculum is curriculum.parse_curriculum is st._parse_curriculum
        assert pool._default_run_dir is rr.default_run_dir           # always passes run_subdir
        # The three arms whose train_* functions call it with only a seed keep their default.
        for module, subdir in ((cgf, "ant_tag_cgf"), (st, "ant_tag_st"), (gauss, "ant_tag_gaussian")):
            assert module._default_run_dir(7).startswith(f"runs/{subdir}/"), module.__name__
            assert module._default_run_dir(7, "elsewhere").startswith("runs/elsewhere/")
    finally:
        sys.path.remove(str(_ANT_TAG_DIR))
        for name in set(sys.modules) - preexisting:
            module = sys.modules.get(name)
            file = getattr(module, "__file__", None) or ""
            if str(_ST_ROOT / "experiments") in file:
                sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Root-level output layout (not yet wired into any script)
# ---------------------------------------------------------------------------

def _fake_checkout(tmp_path, *, as_submodule):
    parent = tmp_path / "parent"
    checkout = parent / "set_transformer"
    checkout.mkdir(parents=True)
    if as_submodule:
        (parent / ".gitmodules").write_text(
            '[submodule "set_transformer"]\n\tpath = set_transformer\n\turl = x\n'
            '[submodule "pomdp-domains"]\n\tpath = pomdp-domains\n\turl = y\n')
    return parent, checkout


def test_output_root_explicit_flag_wins_over_everything(tmp_path):
    _parent, checkout = _fake_checkout(tmp_path, as_submodule=True)
    root = rr.output_root(tmp_path / "flag", environ={rr.OUTPUT_ROOT_ENV: str(tmp_path / "env")},
                          checkout=checkout)
    assert root == (tmp_path / "flag").resolve()


def test_output_root_env_var_beats_repo_detection(tmp_path):
    _parent, checkout = _fake_checkout(tmp_path, as_submodule=True)
    root = rr.output_root(environ={rr.OUTPUT_ROOT_ENV: str(tmp_path / "env")}, checkout=checkout)
    assert root == (tmp_path / "env").resolve()
    assert rr.OUTPUT_ROOT_ENV == "RL_BMDP_RUNS"


def test_output_root_is_the_parent_repo_when_this_checkout_is_its_submodule(tmp_path):
    parent, checkout = _fake_checkout(tmp_path, as_submodule=True)
    assert rr.parent_repo(checkout) == parent.resolve()
    assert rr.output_root(environ={}, checkout=checkout) == parent.resolve() / "runs"


def test_output_root_falls_back_to_the_checkout_when_standalone(tmp_path):
    parent, checkout = _fake_checkout(tmp_path, as_submodule=False)
    assert rr.parent_repo(checkout) is None
    assert rr.output_root(environ={}, checkout=checkout) == checkout.resolve() / "runs"
    # A .gitmodules that lists OTHER submodules does not make this checkout one of them.
    (parent / ".gitmodules").write_text('[submodule "other"]\n\tpath = other\n\turl = z\n')
    assert rr.parent_repo(checkout) is None
    assert rr.output_root(environ={}, checkout=checkout) == checkout.resolve() / "runs"


def test_output_root_is_never_the_current_working_directory(tmp_path, monkeypatch):
    _parent, checkout = _fake_checkout(tmp_path, as_submodule=False)
    monkeypatch.chdir(tmp_path)
    root = rr.output_root(environ={}, checkout=checkout)
    assert root.is_absolute() and root != Path.cwd() / "runs"


def test_this_checkout_resolves_to_the_parent_repo_runs_folder():
    """The real layout: set_transformer/ is a submodule of rl_for_beliefmdps/, whose
    .gitmodules lists it, so by default new runs go to rl_for_beliefmdps/runs."""
    checkout = rr.checkout_root()
    assert checkout == _ST_ROOT and (checkout / "set_transformer" / "rl").is_dir()
    parent = rr.parent_repo(checkout)
    if parent is None:
        pytest.skip("standalone clone: no parent repo lists this checkout")
    assert (parent / ".gitmodules").is_file()
    assert rr.output_root(environ={}) == parent / "runs"


def test_run_dir_shape_is_variant_first_and_keeps_todays_leaf(tmp_path):
    path = rr.run_dir("ant_tag", "smart_mid_slow_v15", "cgf", 3, "6M vis/0.2",
                      root=tmp_path, timestamp="20260912_120000")
    assert path == (tmp_path / "ant_tag" / "smart_mid_slow_v15" / "rl" / "cgf"
                    / "20260912_120000_seed3_6M_vis_0.2")
    pretrain = rr.run_dir("odd_even", "oe50_short", "st", 0, root=tmp_path, kind="pretrain",
                          timestamp="20260912_120000")
    assert pretrain == tmp_path / "odd_even" / "oe50_short" / "pretrain" / "st" / "20260912_120000_seed0"
    # The leaf is exactly what default_run_dir names its runs today.
    assert rr.run_leaf(3, "6M vis/0.2", "20260912_120000") == "20260912_120000_seed3_6M_vis_0.2"
    today = Path(rr.default_run_dir(3, "x", run_tag="6M vis/0.2")).name
    assert re.fullmatch(r"\d{8}_\d{6}_seed3_6M_vis_0\.2", today), today
    # Without an explicit root it goes under output_root(), never the cwd.
    default = rr.run_dir("ant_tag", "smart", "cgf", 0, timestamp="t")
    assert default.is_absolute() and default.parts[-5:-1] == ("ant_tag", "smart", "rl", "cgf")


def test_pretrain_and_eval_dirs_sit_beside_the_rl_runs_of_the_variant(tmp_path):
    """Change 5.2: pretraining output and eval summaries share the variant's folder with
    the RL runs, so everything about one env is in one place."""
    assert rr.pretrain_dir("ant_tag", "smart_hard", "st_pretrain_smart_hard_plain", root=tmp_path) == (
        tmp_path / "ant_tag" / "smart_hard" / "pretrain" / "st_pretrain_smart_hard_plain")
    assert rr.eval_dir("odd_even", "oe50_short", root=tmp_path) == (
        tmp_path / "odd_even" / "oe50_short" / "eval")
    # Without an explicit root both go under output_root(), never the cwd.
    assert rr.pretrain_dir("ant_tag", "smart", "x").is_absolute()
    assert rr.eval_dir("ant_tag", "smart").parts[-3:] == ("ant_tag", "smart", "eval")


def test_domain_of_variant_finds_the_owner_or_refuses():
    from set_transformer.rl import domains
    assert domains.domain_of_variant("smart").name == "ant_tag"
    assert domains.domain_of_variant("oe50_short").name == "odd_even"
    assert domains.domain_of_variant("smart", "pdomains-ant-tag-smart-v0").name == "ant_tag"
    with pytest.raises(ValueError, match="belongs to 0 domains"):
        domains.domain_of_variant("smart", "pdomains-odd-even-50-v0")
    with pytest.raises(ValueError, match="belongs to 0 domains"):
        domains.domain_of_variant("no_such_variant")
