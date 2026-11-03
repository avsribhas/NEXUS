import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import math
import numpy as np
import tarjuman
from neuroml.utils import validate_neuroml2
from tarjuman.validate import reset_times

def validate_circuit(nml_path) -> None:
    """
    Validate a NeuroML 2 circuit file.
    
    Args:
    nml_path (str): Path to the NeuroML 2 file.
    
    Raises:
    Exception: If the file is invalid.
    """
    validate_neuroml2(str(nml_path))

def run_jaxley(lems_path, delta_t=None) -> tuple:
    """
    Run a Jaxley simulation using a LEMS file.
    
    Args:
    lems_path (str): Path to the LEMS file.
    delta_t (float, optional): Time step for the simulation.
    
    Returns:
    tuple: A tuple containing the time array, traces dictionary, and faithfulness of the model.
    """
    result, model = tarjuman.run_lems(str(lems_path), strict=False, delta_t=delta_t)
    time = np.asarray(result.time)
    traces = {key: np.asarray(value) for key, value in result.traces.items()}
    faithful = bool(model.report.is_faithful)
    return time, traces, faithful

def analytic_iaf_isi(c_nf, g_ns, e_leak_mv, thresh_mv, reset_mv, i_na) -> float:
    """
    Calculate the exact interval between resets of a leaky integrate-and-fire cell.
    
    Args:
    c_nf (float): Membrane capacitance in nF.
    g_ns (float): Leak conductance in nS.
    e_leak_mv (float): Leak reversal potential in millivolts.
    thresh_mv (float): Threshold potential in millivolts.
    reset_mv (float): Reset potential in millivolts.
    i_na (float): Applied current in nA.
    
    Returns:
    float: The interval between resets in milliseconds.
    """
    tau_ms = c_nf / g_ns * 1000.0
    v_inf_mv = e_leak_mv + i_na / g_ns * 1000.0
    if v_inf_mv <= thresh_mv:
        return math.inf
    return tau_ms * math.log((v_inf_mv - reset_mv) / (v_inf_mv - thresh_mv))
