from pathlib import Path
import pytest
from nexus.circuits.build import load_motifs

COMMON, MOTIFS = load_motifs(Path(__file__).resolve().parents[1] / "configs" / "motifs")
SWITCHES = sorted(name for name, m in MOTIFS.items() if m["gates"])

def targets(motif, neuron) -> set:
    return {synapse["post"] for synapse in motif["synapses"] if synapse["pre"] == neuron}

@pytest.mark.parametrize("name", SWITCHES)
def test_trigger_differs_from_inputs(name):
    """
    A trigger that shares a target with an input can't be told apart from that input.
    """
    motif = MOTIFS[name]
    triggers = {g["trigger"] for g in motif["gates"]}
    for trig in triggers:
        for other in motif["inputs"]:
            if other not in triggers:
                assert targets(motif, trig).isdisjoint(targets(motif, other))

def test_there_are_four_switch_types():
    assert SWITCHES == ["gain", "global_shutoff", "release", "winner_take_all"]
