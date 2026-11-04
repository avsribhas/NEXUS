import numpy as np
from nexus.realdata.stam import matrix_correlation, stam, stam_sums, train_test_correlation

def rec(x, cells, frames, ids):
    return {"neural": np.asarray(x, dtype=float), "stim_cells": np.array(cells), "stim_frames": np.array(frames), "ids": ids}

LABELS = {"A": 0, "B": 1}

def test_area_is_mean_times_window():
    x = np.zeros((10, 2))
    x[4:6, 1] = 2.0
    r = rec(x, [0], [4], ["A", "B"])
    total, count = stam_sums([r], LABELS, 2)
    assert total[1, 0] == 4.0 and count[1, 0] == 1 and total[0, 0] == 0.0 and count[0, 1] == 0

def test_nan_frames_are_skipped_in_the_mean():
    x = np.zeros((10, 2))
    x[4:6, 1] = 2.0
    x[5, 1] = np.nan
    r = rec(x, [0], [4], ["A", "B"])
    total, count = stam_sums([r], LABELS, 2)
    assert total[1, 0] == 4.0

def test_baseline_subtracts():
    x = np.ones((10, 2))
    r = rec(x, [0], [4], ["A", "B"])
    total, count = stam_sums([r], LABELS, 2, baseline=2)
    assert total[1, 0] == 0.0 and count[1, 0] == 1

def test_window_past_end_and_unlabelled_skipped():
    x = np.zeros((10, 2))
    r = rec(x, [0], [9], ["A", "B"])
    total, count = stam_sums([r], LABELS, 2)
    assert np.all(count == 0)
    r = rec(x, [0], [0], ["", "B"])
    total, count = stam_sums([r], LABELS, 2)
    assert np.all(count == 0)

def test_stam_nan_where_no_events():
    r = rec(np.ones((10, 2)), [0], [0], ["A", "B"])
    m = stam([r], LABELS, 2)
    assert m[1, 0] == 2.0 and np.isnan(m[0, 1])
    m = stam([r], LABELS, 2, min_events=2)
    assert np.all(np.isnan(m))

def test_matrix_correlation_drops_diagonal():
    a = np.array([[100.0, 1.0, 2.0], [3.0, -100.0, 4.0], [5.0, 6.0, np.nan]])
    b = a.copy()
    r, n = matrix_correlation(a, b)
    assert abs(r - 1.0) < 1e-9 and n == 6
    r, n = matrix_correlation(a, b, off_diagonal=False)
    assert abs(r - 1.0) < 1e-9 and n == 8

def test_train_test_recovers_shared_signal():
    rng = np.random.default_rng(0)
    effect = np.random.default_rng(99).normal(size=(4, 4))
    recordings = []
    for _ in range(20):
        x = rng.normal(size=(2000, 4)) * 0.1
        frames = np.arange(10, 1920, 100)
        cells = np.arange(len(frames)) % 4
        for f, c in zip(frames, cells):
            x[f:f + 20, :] += effect[:, c]
        recordings.append(rec(x, cells, frames, ["A", "B", "C", "D"]))
    r, n = train_test_correlation(recordings, 10, 20, seed=0)
    assert r > 0.9
    assert n == 12

def test_recording_with_fewer_cells_than_labels():
    labels = {"A": 0, "B": 1, "C": 2, "D": 3}
    r = rec(np.ones((10, 2)), [0], [0], ["A", "C"])
    total, count = stam_sums([r], labels, 2)
    assert count[2, 0] == 1
    assert total[2, 0] == 2.0
    assert count.sum() == 2
