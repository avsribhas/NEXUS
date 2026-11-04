import numpy as np
import pytest
from nexus.realdata.creamer import identified_stims, label_index, load_all, load_recording

def write_recording(path, neural, stim_cells, stim_frames, ids):
    path.mkdir(parents=True)
    np.savetxt(path / "neural_data.txt", neural, delimiter=",")
    np.savetxt(path / "nan_mask.txt", np.isnan(neural).astype(float), delimiter=",")
    np.savetxt(path / "stim_cell_indicies.txt", np.array(stim_cells, dtype=float))
    np.savetxt(path / "stim_frame_indicies.txt", np.array(stim_frames, dtype=float))
    (path / "cell_ids.txt").write_text("\n".join(f'"{i}"' for i in ids) + "\n")

@pytest.fixture
def tmp_path(tmp_path):
    return tmp_path

def test_load_keeps_values_and_nan(tmp_path):
    neural = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    write_recording(tmp_path / "pumpprobe_a", neural, [1, -3], [0, 2], ["AVAL", ""])
    r = load_recording(tmp_path / "pumpprobe_a")
    assert r["recording_id"] == "pumpprobe_a"
    assert np.array_equal(r["neural"], neural, equal_nan=True)
    assert r["mask"].tolist() == [[False, True], [False, False], [False, False]]
    assert r["ids"] == ["AVAL", ""]
    assert r["stim_cells"].tolist() == [1, -3]

def test_identified_stims_drops_negative(tmp_path):
    neural = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    write_recording(tmp_path / "pumpprobe_a", neural, [1, -3], [0, 2], ["AVAL", ""])
    r = load_recording(tmp_path / "pumpprobe_a")
    cells, frames = identified_stims(r)
    assert cells.tolist() == [1]
    assert frames.tolist() == [0]

def test_mismatch_raises(tmp_path):
    neural = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
    write_recording(tmp_path / "pumpprobe_a", neural, [1, -3], [0, 2], ["AVAL"])
    with pytest.raises(ValueError):
        load_recording(tmp_path / "pumpprobe_a")

def test_load_all_and_labels(tmp_path):
    neural = np.ones((3, 2))
    write_recording(tmp_path / "pumpprobe_b", neural, [0, 1], [0, 1], ["B", ""])
    write_recording(tmp_path / "pumpprobe_a", neural, [0, 1], [0, 1], ["A", "B"])
    rs = load_all(tmp_path)
    assert [x["recording_id"] for x in rs] == ["pumpprobe_a", "pumpprobe_b"]
    assert label_index(rs) == {"A": 0, "B": 1}
