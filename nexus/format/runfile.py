from pynwb import NWBFile
from pynwb.file import Subject

EXPERIMENTER = ["s9m Labs"]
INSTITUTION = "s9m Labs"
EXPERIMENT_DESCRIPTION = "NEXUS Phase 1 v2: simulated circuits with planted switches, and real recordings stored in the same format."
KEYWORDS = ["NEXUS", "state-dependent gating", "simulation"]
SUBJECT_DESCRIPTION = "For simulated runs, one wiring-corruption draw of a base circuit (a simulated animal)."

def new_run_file(identifier: str, session_description: str, session_start_time, subject_id: str, species: str) -> NWBFile:
    subject = Subject(subject_id=subject_id, species=species, sex="U", description=SUBJECT_DESCRIPTION)
    return NWBFile(session_description=session_description,
                    identifier=identifier,
                    session_start_time=session_start_time,
                    experimenter=EXPERIMENTER,
                    institution=INSTITUTION,
                    lab=INSTITUTION,
                    experiment_description=EXPERIMENT_DESCRIPTION,
                    keywords=KEYWORDS,
                    subject=subject)
