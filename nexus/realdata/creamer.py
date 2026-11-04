from pathlib import Path
import numpy as np

def load_recording(path) -> dict:
    """Load neural recording data from a directory."""
    path = Path(path)
    neural = np.loadtxt(path / "neural_data.txt", delimiter=",", ndmin=2)
    mask = np.loadtxt(path / "nan_mask.txt", delimiter=",", ndmin=2) > 0
    stim_cells = np.loadtxt(path / "stim_cell_indicies.txt", ndmin=1).astype(int)
    stim_frames = np.loadtxt(path / "stim_frame_indicies.txt", ndmin=1).astype(int)
    ids = [line.strip().strip('"') for line in (path / "cell_ids.txt").read_text().splitlines()]
    if neural.shape != mask.shape or len(ids) != neural.shape[1] or len(stim_cells) != len(stim_frames):
        raise ValueError("Data inconsistencies")
    return {"recording_id": path.name, "neural": neural, "mask": mask, "stim_cells": stim_cells, "stim_frames": stim_frames, "ids": ids}

def identified_stims(rec) -> tuple:
    """Filter stim_cells and stim_frames to include only non-negative values."""
    return rec["stim_cells"][rec["stim_cells"] >= 0], rec["stim_frames"][rec["stim_cells"] >= 0]

def load_all(root) -> list:
    """Load all neural recordings from a root directory."""
    return [load_recording(p) for p in sorted(Path(root).glob("pumpprobe_*"))]

def label_index(recordings) -> dict:
    """Create a dictionary mapping unique cell IDs to their index positions."""
    all_ids = sorted(set(id for rec in recordings for id in rec["ids"] if id))
    return {id: idx for idx, id in enumerate(all_ids)}
