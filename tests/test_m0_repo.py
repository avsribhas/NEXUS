import importlib.metadata
import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
LAYOUT_PACKAGES = ["format", "circuits", "perturb", "models", "baselines", "degrade", "energy", "realdata", "score"]
LAYOUT_DIRS = ["prereg", "configs", "tests", "logs"]
REQUIRED_PINS = {"pynwb": "3.1.3", "ndx-optogenetics": "0.4.1"}

def load_pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)

def all_requirement_strings(pyproject: dict) -> list[str]:
    return pyproject["project"]["dependencies"] + sum(pyproject["project"].get("optional-dependencies", {}).values(), [])

def normalize(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")

def test_no_claude_md_in_archive():
    collected = [p for p in (ROOT / "archive").rglob("*") if p.name == "CLAUDE.md"]
    assert collected == [], collected

def test_layout_packages_exist():
    for name in LAYOUT_PACKAGES:
        assert (ROOT / "nexus" / name / "__init__.py").is_file(), f"Missing path: {ROOT / 'nexus' / name / '__init__.py'}"
    for name in LAYOUT_DIRS:
        assert (ROOT / name).is_dir(), f"Missing path: {ROOT / name}"

def test_requires_python_312():
    assert load_pyproject()["project"]["requires-python"] == "==3.12.*"

def test_every_dependency_pinned_exactly():
    for s in all_requirement_strings(load_pyproject()):
        req = Requirement(s)
        specs = list(req.specifier)
        assert len(specs) == 1 and specs[0].operator == "==", f"{s}"

def test_required_pins():
    pins = {normalize(Requirement(s).name): list(Requirement(s).specifier)[0].version for s in load_pyproject()["project"]["dependencies"]}
    for name, version in REQUIRED_PINS.items():
        assert pins.get(name) == version, f"{name}"

def test_installed_versions_match_pins():
    for s in load_pyproject()["project"]["dependencies"]:
        req = Requirement(s)
        wanted = list(req.specifier)[0].version
        installed = importlib.metadata.version(req.name)
        assert installed == wanted, f"{req.name}: installed {installed}, pinned {wanted}"

def test_dockerfile_base_pinned_by_digest():
    lines = (ROOT / "Dockerfile").read_text().splitlines()
    from_lines = [line for line in lines if line.strip().upper().startswith("FROM ")]
    assert len(from_lines) >= 1
    pattern = re.compile(r"^FROM\s+\S+@sha256:[0-9a-f]{64}(\s+AS\s+\S+)?\s*$", re.IGNORECASE)
    for line in from_lines:
        assert pattern.match(line.strip()), f"{line}"

def test_no_imports_from_archive():
    for path in (ROOT / "nexus").rglob("*.py"):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                assert "archive" not in stripped, f"{path}: {stripped}"
