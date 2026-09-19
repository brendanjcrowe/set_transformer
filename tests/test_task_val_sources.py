"""`--val_sources`: the task objective's validation LOSS can be restricted to the held-out rows of chosen
DAgger sources (the record validated on policy rows only); training rows, the tail and the metrics are unchanged."""
import json
import numpy as np
import pytest
import torch

from set_transformer.rl.domains.hunt import TaskData, parse_val_sources


def _write(tmp_path, n=200, with_sources=True):
    rng = np.random.default_rng(0)
    arrays = dict(particles=rng.normal(size=(n, 10, 2)).astype(np.float32), weights=np.full((n, 10), 0.1, np.float32),
                  agent=rng.normal(size=(n, 2)).astype(np.float32), centers=rng.normal(size=(n, 5, 2)).astype(np.float32),
                  alive=np.ones((n, 5), np.float32), counts=rng.integers(5, 30, (n, 5)).astype(np.float32),
                  sigmas=np.full((n, 5), np.nan, np.float32), target=rng.normal(size=(n, 2)).astype(np.float32),
                  target_index=np.zeros(n, np.int64), metadata=np.array(json.dumps({"variant": "least_mass", "particle_scale": 10.0})))
    if with_sources:
        arrays["source_round"] = np.tile(np.array([0, 1, 2, 0], np.int8), n // 4)
    path = tmp_path / "d.npz"
    np.savez(path, **arrays)
    return path


def test_parse_val_sources():
    assert parse_val_sources(None) is None
    assert parse_val_sources("") is None
    assert parse_val_sources("1,2") == (1, 2)
    assert parse_val_sources(" 2 ") == (2,)


def test_val_sources_select_only_the_tail_rows_of_those_sources(tmp_path):
    data = TaskData(str(_write(tmp_path)), 0.1, torch.device("cpu"), val_sources=(1, 2))
    assert data.n_train == 180 and data.n_val == 20                     # split unchanged
    rows = data.val_loss_rows.numpy()
    src = data.val["source_round"].numpy()
    assert len(rows) == 10 and set(src[rows]) == {1, 2}                  # half the tail rows are sources 1 / 2
    batches = list(data.val_loss_batches(4))
    assert sum(len(b) for b in batches) == 10 and all(torch.is_tensor(b) for b in batches)


def test_without_val_sources_the_whole_tail_is_used(tmp_path):
    data = TaskData(str(_write(tmp_path)), 0.1, torch.device("cpu"))
    assert data.val_sources is None and data.val_loss_rows is None
    batches = list(data.val_loss_batches(8))
    assert all(isinstance(b, slice) for b in batches) and sum(b.stop - b.start for b in batches) == 20


def test_val_sources_need_a_source_round_array(tmp_path):
    with pytest.raises(ValueError, match="source_round"):
        TaskData(str(_write(tmp_path, with_sources=False)), 0.1, torch.device("cpu"), val_sources=(1,))


def test_val_sources_with_no_matching_rows_raise(tmp_path):
    with pytest.raises(ValueError, match="no validation row"):
        TaskData(str(_write(tmp_path)), 0.1, torch.device("cpu"), val_sources=(7,))


def _write_ant_tag(tmp_path, n=100):
    """A labelled Ant-Tag file (position head only): no source_round array, as the collector writes it."""
    rng = np.random.default_rng(0)
    path = tmp_path / "ant.npz"
    np.savez(path, particles=rng.normal(size=(n, 10, 2)).astype(np.float32), weights=np.full((n, 10), 0.1, np.float32),
             ant=rng.normal(size=(n, 2)).astype(np.float32), target=rng.normal(size=(n, 2)).astype(np.float32),
             step=np.arange(n, dtype=np.int32), metadata=np.array(json.dumps({"variant": "smart", "particle_scale": 4.5})))
    return path


def test_ant_tag_forwards_val_sources_and_refuses_without_source_round(tmp_path):
    from set_transformer.rl.domains.ant_tag import AntTagTaskData
    ok = AntTagTaskData(str(_write_ant_tag(tmp_path)), 0.1, torch.device("cpu"), heads=("position",))
    assert ok.val_sources is None and ok.val_loss_rows is None            # default: unchanged behaviour
    with pytest.raises(ValueError, match="source_round"):                 # the flag is never silently ignored
        AntTagTaskData(str(_write_ant_tag(tmp_path)), 0.1, torch.device("cpu"), heads=("position",), val_sources=(1,))
