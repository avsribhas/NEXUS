import re
import xml.etree.ElementTree as ET
from pathlib import Path

NML_NS = "http://www.neuroml.org/schema/neuroml2"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
SCHEMA_LOCATION = "http://www.neuroml.org/schema/neuroml2 https://raw.github.com/NeuroML/NeuroML2/development/Schemas/NeuroML2/NeuroML_v2.3.xsd"
NEVER_FIRE_MV = 60.0
ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

def _num(x) -> str:
    return format(float(x), ".9g")

def validate_spec(spec: dict) -> list:
    """Return a list of problems with a circuit spec; empty if it is fine."""
    problems = []
    if not ID_PATTERN.match(spec["circuit_id"]):
        problems.append(f"circuit_id: invalid format")
    seen_ids = set()
    for neuron in spec["neurons"]:
        nid = neuron["id"]
        if not ID_PATTERN.match(nid):
            problems.append(f"neuron {nid}: invalid format")
        if nid in seen_ids:
            problems.append(f"neuron {nid}: duplicate id")
        seen_ids.add(nid)
        if neuron["C_nF"] <= 0 or neuron["g_leak_nS"] <= 0:
            problems.append(f"neuron {nid}: invalid C_nF or g_leak_nS")
    for i, graded in enumerate(spec["graded"]):
        pre, post = graded["pre"], graded["post"]
        if pre not in seen_ids or post not in seen_ids:
            problems.append(f"graded synapse {i}: unknown neuron")
        if graded["g_nS"] < 0 or graded["delta_mV"] <= 0 or graded["k_per_ms"] <= 0:
            problems.append(f"graded synapse {i}: invalid parameters")
    for i, gap in enumerate(spec["gap"]):
        a, b = gap["a"], gap["b"]
        if a not in seen_ids or b not in seen_ids:
            problems.append(f"gap junction {i}: unknown neuron")
        if a == b:
            problems.append(f"gap junction {i}: a == b")
        if gap["g_nS"] < 0:
            problems.append(f"gap junction {i}: invalid conductance")
    return problems

def write_network(spec: dict, path, pulses=()) -> None:
    """Write a spec as a NeuroML 2 network file, with optional current pulses."""
    problems = validate_spec(spec)
    if problems:
        raise ValueError("\n".join(problems))
    root = ET.Element("neuroml", attrib={"xmlns": NML_NS, "xmlns:xsi": XSI_NS, "xsi:schemaLocation": SCHEMA_LOCATION, "id": spec["circuit_id"]})
    if spec["graded"]:
        ET.SubElement(root, "silentSynapse", id="silent")
    for k, gap in enumerate(spec["gap"]):
        ET.SubElement(root, "gapJunction", id=f"gj_{k}", conductance=_num(gap["g_nS"])+"nS")
    for k, graded in enumerate(spec["graded"]):
        ET.SubElement(root, "gradedSynapse", id=f"gs_{k}", conductance=_num(graded["g_nS"])+"nS", delta=_num(graded["delta_mV"])+"mV", Vth=_num(graded["vth_mV"])+"mV", k=_num(graded["k_per_ms"])+"per_ms", erev=_num(graded["erev_mV"])+"mV")
    for k, pulse in enumerate(pulses):
        ET.SubElement(root, "pulseGenerator", id=f"pg_{k}", delay=_num(pulse["delay_ms"])+"ms", duration=_num(pulse["duration_ms"])+"ms", amplitude=_num(pulse["amplitude_nA"])+"nA")
    for neuron in spec["neurons"]:
        ET.SubElement(root, "iafCell", id=f"cell_{neuron['id']}", leakConductance=_num(neuron["g_leak_nS"])+"nS", leakReversal=_num(neuron["e_leak_mV"])+"mV", thresh=_num(NEVER_FIRE_MV)+"mV", reset=_num(neuron["e_leak_mV"])+"mV", C=_num(neuron["C_nF"])+"nF")
    network = ET.SubElement(root, "network", id="net")
    for neuron in spec["neurons"]:
        ET.SubElement(network, "population", id=f"pop_{neuron['id']}", component=f"cell_{neuron['id']}", size="1")
    for k, gap in enumerate(spec["gap"]):
        gj = ET.SubElement(network, "electricalProjection", id=f"gap_{k}", presynapticPopulation=f"pop_{gap['a']}", postsynapticPopulation=f"pop_{gap['b']}")
        ET.SubElement(gj, "electricalConnection", id="0", preCell="0", postCell="0", synapse=f"gj_{k}")
    for k, graded in enumerate(spec["graded"]):
        graded_proj = ET.SubElement(network, "continuousProjection", id=f"graded_{k}", presynapticPopulation=f"pop_{graded['pre']}", postsynapticPopulation=f"pop_{graded['post']}")
        ET.SubElement(graded_proj, "continuousConnection", id="0", preCell="0", postCell="0", preComponent="silent", postComponent=f"gs_{k}")
    for k, pulse in enumerate(pulses):
        ET.SubElement(network, "explicitInput", target=f"pop_{pulse['target']}[0]", input=f"pg_{k}")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="UTF-8", xml_declaration=True)

def write_lems(spec: dict, nml_filename: str, path, length_ms: float, step_ms: float) -> None:
    """Write a LEMS file that runs the network and records every neuron's voltage."""
    root = ET.Element("Lems")
    ET.SubElement(root, "Target", component="sim1")
    for filename in ["Cells.xml", "Networks.xml", "Simulation.xml", nml_filename]:
        ET.SubElement(root, "Include", file=filename)
    sim = ET.SubElement(root, "Simulation", id="sim1", length=_num(length_ms)+"ms", step=_num(step_ms)+"ms", target="net")
    output_file = ET.SubElement(sim, "OutputFile", id="of0", fileName=spec["circuit_id"]+".v.dat")
    for neuron in spec["neurons"]:
        ET.SubElement(output_file, "OutputColumn", id=neuron["id"], quantity=f"pop_{neuron['id']}[0]/v")
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="UTF-8", xml_declaration=True)
