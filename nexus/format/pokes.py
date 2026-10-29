from ndx_ophys_devices import Effector
from ndx_optogenetics import (
    OptogeneticEffectors,
    OptogeneticEpochsTable,
    OptogeneticExperimentMetadata,
    OptogeneticPulsesTable,
    OptogeneticSitesTable,
)

EFFECTOR_NAME = "simulated_current_injection"
CUSTOM_COLUMNS = {
    "target_neuron": "Neuron that received the poke.",
    "amplitude": "Poke amplitude, in amplitude_unit.",
    "amplitude_unit": "Unit of amplitude.",
    "modality": "How the poke was delivered, for example current.",
}

def add_pokes(nwbfile, pokes: list) -> None:
    if not pokes:
        raise ValueError("pokes must not be empty")
    
    effector = Effector(name=EFFECTOR_NAME, label="simulated", description="Current injected into a simulated neuron; not a light-gated effector.")
    effectors = OptogeneticEffectors(effectors=[effector])
    
    sites = OptogeneticSitesTable(description="One site: the simulated current injection.")
    sites.add_row(effector=effector)
    
    metadata = OptogeneticExperimentMetadata(
        optogenetic_sites_table=sites,
        optogenetic_effectors=effectors,
        stimulation_software="nexus"
    )
    nwbfile.add_lab_meta_data(metadata)
    
    pulses = OptogeneticPulsesTable(name="optogenetic_pulses", description="Simulated pokes, one row per poke.", target_tables={"optogenetic_sites": sites})
    for column_name, description in CUSTOM_COLUMNS.items():
        pulses.add_column(name=column_name, description=description)
    
    for poke in pokes:
        pulses.add_row(
            start_time=float(poke["start_time"]),
            stop_time=float(poke["stop_time"]),
            power_in_mW=float("nan"),
            wavelength_in_nm=float("nan"),
            optogenetic_sites=[0],
            target_neuron=str(poke["target_neuron"]),
            amplitude=float(poke["amplitude"]),
            amplitude_unit=str(poke["amplitude_unit"]),
            modality=str(poke["modality"])
        )
    
    nwbfile.add_time_intervals(pulses)
    
    epochs = OptogeneticEpochsTable(name="optogenetic_epochs", description="One epoch per simulated poke.", target_tables={"optogenetic_sites": sites})
    for poke in pokes:
        epochs.add_row(
            start_time=float(poke["start_time"]),
            stop_time=float(poke["stop_time"]),
            stimulation_on=True,
            pulse_length_in_ms=(float(poke["stop_time"]) - float(poke["start_time"])) * 1000.0,
            period_in_ms=float("nan"),
            number_pulses_per_pulse_train=1,
            number_trains=1,
            intertrain_interval_in_ms=float("nan"),
            power_in_mW=float("nan"),
            wavelength_in_nm=float("nan"),
            optogenetic_sites=[0]
        )
    
    nwbfile.add_time_intervals(epochs)

def read_pokes(nwbfile) -> list:
    t = nwbfile.intervals["optogenetic_pulses"]
    pokes = []
    for i in range(len(t)):
        poke = {
            "start_time": float(t["start_time"][i]),
            "stop_time": float(t["stop_time"][i]),
            "target_neuron": str(t["target_neuron"][i]),
            "amplitude": float(t["amplitude"][i]),
            "amplitude_unit": str(t["amplitude_unit"][i]),
            "modality": str(t["modality"][i])
        }
        pokes.append(poke)
    return pokes
