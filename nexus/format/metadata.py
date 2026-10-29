import subprocess
import pathlib
import numpy as np
import pynwb
from nexus.format.ndx_nexus import NexusRunMetadata

SCHEMA_VERSION = "0.1.0"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

def _git_commit() -> str:
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()

def _git_dirty() -> bool:
    result = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip() != ''

def _gpu_info() -> tuple:
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], capture_output=True, text=True, check=True)
        line = result.stdout.strip().split('\n')[0]
        model, driver = line.split(', ')
        return (model.strip(), driver.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return (None, None)

def collect_run_metadata(seed: int, simulated: bool, simulator: str, simulator_version: str, thread_count: int, use_gpu: bool = False) -> dict:
    gpu_model, gpu_driver = _gpu_info() if use_gpu else (None, None)
    return {
        'schema_version': SCHEMA_VERSION,
        'simulated': simulated,
        'simulator': simulator,
        'simulator_version': simulator_version,
        'pynwb_version': pynwb.__version__,
        'seed': seed,
        'thread_count': thread_count,
        'gpu_model': gpu_model,
        'gpu_driver': gpu_driver,
        'git_commit': _git_commit(),
        'git_dirty': _git_dirty()
    }

def make_run_metadata(metadata: dict) -> NexusRunMetadata:
    kwargs = {
        'name': 'nexus_run_metadata',
        'schema_version': metadata['schema_version'],
        'simulated': bool(metadata['simulated']),
        'git_dirty': bool(metadata['git_dirty']),
        'simulator': metadata['simulator'],
        'simulator_version': metadata['simulator_version'],
        'pynwb_version': metadata['pynwb_version'],
        'git_commit': metadata['git_commit'],
        'seed': np.int64(metadata['seed']),
        'thread_count': np.int64(metadata['thread_count'])
    }
    if metadata['gpu_model'] is not None:
        kwargs['gpu_model'] = metadata['gpu_model']
    if metadata['gpu_driver'] is not None:
        kwargs['gpu_driver'] = metadata['gpu_driver']
    return NexusRunMetadata(**kwargs)

def read_run_metadata(nwbfile) -> dict:
    m = nwbfile.lab_meta_data["nexus_run_metadata"]
    return {
        'schema_version': str(m.schema_version),
        'simulated': bool(m.simulated),
        'simulator': str(m.simulator),
        'simulator_version': str(m.simulator_version),
        'pynwb_version': str(m.pynwb_version),
        'seed': int(m.seed),
        'thread_count': int(m.thread_count),
        'gpu_model': str(m.gpu_model) if m.gpu_model is not None else None,
        'gpu_driver': str(m.gpu_driver) if m.gpu_driver is not None else None,
        'git_commit': str(m.git_commit),
        'git_dirty': bool(m.git_dirty)
    }
