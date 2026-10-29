from nexus.format.ndx_nexus import NexusDatasheets, NexusGroundTruth, NexusWiringCorruption
import math

SWITCH_TYPES = ("release", "gain", "winner_take_all", "global_shutoff", "stochastic", "none")
TAU_REF_DEFINITIONS = ("trigger_time_constant", "median_membrane_time_constant")
CORRUPTION_KINDS = ("missing_edge", "false_edge", "sign_flip", "weight_error")
DATASHEET_TEXT = ["gate_id", "trigger_channel", "threshold_unit", "tau_ref_definition", "reach_criterion"]
DATASHEET_FLOAT = ["trigger_gain", "threshold", "tau_on_over_tau_ref", "tau_off_over_tau_ref", "reach"]
DATASHEET_COLUMNS = ["gate_id", "trigger_channel", "trigger_gain", "threshold", "threshold_unit", "tau_on_over_tau_ref", "tau_off_over_tau_ref", "tau_ref_definition", "reach", "reach_criterion", "priority", "cost_atp_per_flip"]
CORRUPTION_COLUMNS = ["pre_neuron", "post_neuron", "kind", "true_weight", "stored_weight"]

def make_ground_truth(switch_type: str, is_decoy: bool, twin_id: str, base_circuit_id: str, animal_id: str, datasheets: list, wiring_corruption: list) -> NexusGroundTruth:
    sheets = NexusDatasheets(name="datasheets", description="Planted datasheets, one row per gate.")
    for d in datasheets:
        sheets.add_row(
            gate_id=str(d["gate_id"]),
            trigger_channel=str(d["trigger_channel"]),
            threshold_unit=str(d["threshold_unit"]),
            tau_ref_definition=str(d["tau_ref_definition"]),
            reach_criterion=str(d["reach_criterion"]),
            trigger_gain=float(d["trigger_gain"]),
            threshold=float(d["threshold"]),
            tau_on_over_tau_ref=float(d["tau_on_over_tau_ref"]),
            tau_off_over_tau_ref=float(d["tau_off_over_tau_ref"]),
            reach=float(d["reach"]),
            priority=int(d["priority"]),
            cost_atp_per_flip=float("nan") if d["cost_atp_per_flip"] is None else float(d["cost_atp_per_flip"])
        )
    corruption = NexusWiringCorruption(name="wiring_corruption", description="Corrupted synapses, one row each.")
    for r in wiring_corruption:
        corruption.add_row(
            pre_neuron=str(r["pre_neuron"]),
            post_neuron=str(r["post_neuron"]),
            kind=str(r["kind"]),
            true_weight=float(r["true_weight"]),
            stored_weight=float(r["stored_weight"])
        )
    return NexusGroundTruth(name="nexus_ground_truth", switch_type=switch_type, is_decoy=bool(is_decoy), twin_id=twin_id, base_circuit_id=base_circuit_id, animal_id=animal_id, datasheets=sheets, wiring_corruption=corruption)

def read_ground_truth(nwbfile) -> dict:
    g = nwbfile.lab_meta_data["nexus_ground_truth"]
    datasheets = []
    for i in range(len(g.datasheets)):
        datasheets.append({
            "gate_id": str(g.datasheets["gate_id"][i]),
            "trigger_channel": str(g.datasheets["trigger_channel"][i]),
            "threshold_unit": str(g.datasheets["threshold_unit"][i]),
            "tau_ref_definition": str(g.datasheets["tau_ref_definition"][i]),
            "reach_criterion": str(g.datasheets["reach_criterion"][i]),
            "trigger_gain": float(g.datasheets["trigger_gain"][i]),
            "threshold": float(g.datasheets["threshold"][i]),
            "tau_on_over_tau_ref": float(g.datasheets["tau_on_over_tau_ref"][i]),
            "tau_off_over_tau_ref": float(g.datasheets["tau_off_over_tau_ref"][i]),
            "reach": float(g.datasheets["reach"][i]),
            "priority": int(g.datasheets["priority"][i]),
            "cost_atp_per_flip": None if math.isnan(float(g.datasheets["cost_atp_per_flip"][i])) else float(g.datasheets["cost_atp_per_flip"][i])
        })
    wiring_corruption = []
    for i in range(len(g.wiring_corruption)):
        wiring_corruption.append({
            "pre_neuron": str(g.wiring_corruption["pre_neuron"][i]),
            "post_neuron": str(g.wiring_corruption["post_neuron"][i]),
            "kind": str(g.wiring_corruption["kind"][i]),
            "true_weight": float(g.wiring_corruption["true_weight"][i]),
            "stored_weight": float(g.wiring_corruption["stored_weight"][i])
        })
    return {
        "switch_type": str(g.switch_type),
        "is_decoy": bool(g.is_decoy),
        "twin_id": str(g.twin_id),
        "base_circuit_id": str(g.base_circuit_id),
        "animal_id": str(g.animal_id),
        "datasheets": datasheets,
        "wiring_corruption": wiring_corruption
    }
