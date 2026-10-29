import math
import re
from pynwb import NWBHDF5IO
from nexus.format.metadata import read_run_metadata
from nexus.format.ground_truth import CORRUPTION_KINDS, DATASHEET_FLOAT, DATASHEET_TEXT, SWITCH_TYPES, TAU_REF_DEFINITIONS, read_ground_truth

def check_metadata(nwbfile) -> list:
    problems = []
    if "nexus_run_metadata" not in nwbfile.lab_meta_data:
        problems.append("missing nexus_run_metadata")
    else:
        metadata = read_run_metadata(nwbfile)
        for key in ["schema_version", "simulator", "simulator_version", "pynwb_version", "git_commit"]:
            if not metadata[key]:
                problems.append(f"nexus_run_metadata.{key} is empty")
        if metadata["git_commit"] and not re.match(r'[0-9a-f]{40}', metadata["git_commit"]):
            problems.append("nexus_run_metadata.git_commit is not a 40-character commit hash")
    if nwbfile.subject is None:
        problems.append("missing subject")
    else:
        if not nwbfile.subject.subject_id or not nwbfile.subject.subject_id.strip():
            problems.append("subject.subject_id is empty")
        if not nwbfile.subject.species or not nwbfile.subject.species.strip():
            problems.append("subject.species is empty")
    return problems

def check_ground_truth(nwbfile, allow_null_cost: bool = True) -> list:
    problems = []
    if "nexus_run_metadata" not in nwbfile.lab_meta_data:
        return ["missing nexus_run_metadata"]
    if not read_run_metadata(nwbfile)["simulated"]:
        return []
    if "nexus_ground_truth" not in nwbfile.lab_meta_data:
        return ["missing nexus_ground_truth"]
    gt = read_ground_truth(nwbfile)
    sheets = gt["datasheets"]
    if gt["switch_type"] not in SWITCH_TYPES:
        problems.append(f"invalid switch_type: {gt['switch_type']}")
    if gt["switch_type"] == "none" and sheets:
        problems.append("switch_type none must have no datasheets")
    if gt["switch_type"] != "none" and not sheets:
        problems.append("a switch circuit needs at least one datasheet")
    if gt["is_decoy"] and gt["switch_type"] != "none":
        problems.append("a decoy must have switch_type none")
    if nwbfile.subject is None or nwbfile.subject.subject_id != gt["animal_id"]:
        problems.append("subject_id must equal animal_id")
    for d in sheets:
        for key in DATASHEET_TEXT:
            if not d[key]:
                problems.append(f"empty {key} in datasheet {d['gate_id']}")
        for key in DATASHEET_FLOAT:
            if not math.isfinite(d[key]):
                problems.append(f"non-finite {key} in datasheet {d['gate_id']}")
        if math.isfinite(d["reach"]) and (d["reach"] < 0 or d["reach"] > 1):
            problems.append(f"invalid reach {d['reach']} in datasheet {d['gate_id']}")
        if math.isfinite(d["tau_on_over_tau_ref"]) and d["tau_on_over_tau_ref"] <= 0:
            problems.append(f"invalid tau_on_over_tau_ref {d['tau_on_over_tau_ref']} in datasheet {d['gate_id']}")
        if math.isfinite(d["tau_off_over_tau_ref"]) and d["tau_off_over_tau_ref"] <= 0:
            problems.append(f"invalid tau_off_over_tau_ref {d['tau_off_over_tau_ref']} in datasheet {d['gate_id']}")
        if d["tau_ref_definition"] not in TAU_REF_DEFINITIONS:
            problems.append(f"invalid tau_ref_definition {d['tau_ref_definition']} in datasheet {d['gate_id']}")
        if d["cost_atp_per_flip"] is None and not allow_null_cost:
            problems.append(f"missing cost_atp_per_flip in datasheet {d['gate_id']}")
        if d["cost_atp_per_flip"] is not None and (not math.isfinite(d["cost_atp_per_flip"]) or d["cost_atp_per_flip"] < 0):
            problems.append(f"invalid cost_atp_per_flip {d['cost_atp_per_flip']} in datasheet {d['gate_id']}")
    priorities = sorted(d["priority"] for d in sheets)
    if sheets and priorities != list(range(1, len(sheets) + 1)):
        problems.append("gate priorities must be exactly 1 to n")
    for r in gt["wiring_corruption"]:
        if r["kind"] not in CORRUPTION_KINDS:
            problems.append(f"invalid corruption kind: {r['kind']}")
    return problems

def check_file(path, allow_null_cost: bool = True) -> list:
    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()
        return check_metadata(nwbfile) + check_ground_truth(nwbfile, allow_null_cost)
