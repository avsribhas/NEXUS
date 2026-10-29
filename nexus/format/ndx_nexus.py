from pathlib import Path
from pynwb import get_class, load_namespaces

NAMESPACE = "ndx-nexus"
NAMESPACE_PATH = Path(__file__).parent / "spec" / "ndx-nexus.namespace.yaml"
load_namespaces(str(NAMESPACE_PATH))

NexusRunMetadata = get_class("NexusRunMetadata", NAMESPACE)
NexusDatasheets = get_class("NexusDatasheets", NAMESPACE)
NexusWiringCorruption = get_class("NexusWiringCorruption", NAMESPACE)
NexusGroundTruth = get_class("NexusGroundTruth", NAMESPACE)
