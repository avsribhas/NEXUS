from pynwb import NWBFile
from pynwb.file import Subject

EXPERIMENTER = ["s9m Labs"]
INSTITUTION = "s9m Labs"
EXPERIMENT_DESCRIPTION = "NEXUS Phase 1 v2: simulated circuits with planted switches, and real recordings stored in the same format."
KEYWORDS = ["NEXUS", "state-dependent gating", "simulation"]
SUBJECT_DESCRIPTION = "For simulated runs, one wiring-corruption draw of a base circuit (a simulated animal). Age is the life stage of the connectome or model the circuit is built from."
DEFAULT_SEX = {"Caenorhabditis elegans": "XX"}

def new_run_file(identifier: str, session_description: str, session_start_time, subject_id: str, species: str, age: str = None, sex: str = None) -> NWBFile:
    if sex is None:
        sex = DEFAULT_SEX.get(species, "U")
    
    subject_kwargs = {
        "subject_id": subject_id,
        "species": species,
        "sex": sex,
        "description": SUBJECT_DESCRIPTION
    }
    
    if age is not None:
        subject_kwargs.update({"age": age, "age__reference": "birth"})
    
    subject = Subject(**subject_kwargs)
    
    return NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start_time,
        experimenter=list(EXPERIMENTER),
        institution=INSTITUTION,
        lab=INSTITUTION,
        experiment_description=EXPERIMENT_DESCRIPTION,
        keywords=list(KEYWORDS),
        subject=subject
    )
