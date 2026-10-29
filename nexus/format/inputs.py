import copy
import numpy as np

PREFIX = "simulator_input__"

class InputRecorder:
    def __init__(self, simulator):
        self.simulator = simulator
        self.received = None

    def __call__(self, params: dict):
        self.received = copy.deepcopy(params)
        return self.simulator(params)

def write_inputs(nwbfile, params: dict) -> None:
    for key in sorted(params):
        nwbfile.add_scratch(np.asarray(params[key]), name=PREFIX + key, description=f"Simulator input {key}, exactly as passed to the simulator.")

def read_inputs(nwbfile) -> dict:
    return {name[len(PREFIX):]: np.asarray(nwbfile.get_scratch(name)) for name in nwbfile.scratch if name.startswith(PREFIX)}

def check_inputs_match(stored: dict, received: dict) -> list:
    mismatched_keys = []
    for key in sorted(set(stored.keys()).union(received.keys())):
        if key not in stored or key not in received or not np.array_equal(np.asarray(stored[key]), np.asarray(received[key]), equal_nan=True):
            mismatched_keys.append(key)
    return mismatched_keys
