# NEXUS

NEXUS measures the rules that decide which parts of a nervous system switch on, given the
animal's state. Each rule is written up as a datasheet with six fields:

| Field | What it says |
|---|---|
| Trigger | which input or state flips the switch, and how strongly |
| Threshold | how big that input has to be |
| Timing | how fast the switch turns on and off |
| Reach | how much of the circuit changes when it flips |
| Priority | which switch wins when two compete |
| Cost | the energy one flip costs, in ATP |

## Where things stand

Phase 1 is building a test bench: simulated circuits with switches planted in them on
purpose, some with no switch at all, and some that only look like they have one. The point
is to check that NEXUS finds the switches that are there and doesn't invent ones that aren't,
before it's trusted on real recordings.

| Step | What it is | Status |
|---|---|---|
| M0 | Archive the old study, pin the software, set up the repo | Done |
| M1 | One file format for every run, a checker, a locked test set | In progress |
| M2 | Pre-registration: fix the tests and pass marks before any held-out data exists | Not started; needs sign-off from Abhiram and Udeern |
| M3 to M7 | Planted circuits, poke-and-predict, degraded data, energy, a real-data check | Not started |

Done so far in M1:
- Every run is saved as an NWB file, with the planted answers, the run settings and the
  simulated pokes stored inside it.
- A checker rejects any run that is missing its answers, its settings, or its energy number.
- The exam lock: the files that decide the result get hashed into one lock file, and one call
  says whether any changed. Held-out data can only be opened through one function that logs
  every opening and stops at a set limit. No held-out data will be made until a key holder is
  named.
- A small reference circuit, written in NeuroML, runs in the Jaxley simulator and matches the
  exact textbook answer to within one time step. The same check found a problem: the
  translator we use (tarjuman 0.5.1) drops the circuit's synaptic input while reporting the
  conversion as complete. NEURON and the official NeuroML simulator (jNeuroML) both run the
  same circuit with the synapse working, which confirms it. NEURON is the second simulator
  that every circuit type is checked against; its spike times match Jaxley's exactly.
- The train/test split works on four kinds of unit: whole circuits, single neurons, kinds
  of poke, and brain states. Every trial of a unit lands on the same side, whatever its
  random seed.
- Training and simulation repeat exactly on the same device. Training on CPU and GPU differs
  by about 6e-08.

There are no results yet. Nothing here should be read as a finding.

## How the work is done

All code is written by a small local model (Qwen 2.5 Coder 7B) on a Windows machine. Claude
writes the instructions, reviews every file, runs the tests and keeps the log. Tests are
written before the code they check. Tests that don't depend on an open decision are frozen
with SHA-256 hashes (`tests/frozen.sha256`), so they can't be quietly changed to make
something pass.

Everything runs in WSL2 (Ubuntu 24.04, Python 3.12) on that machine, with an RTX 4050 GPU.

## Running the tests

On Linux or WSL2, with [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --frozen --extra gpu
.venv/bin/python -m pytest -q -rfs
```

Drop `--extra gpu` on a machine without an NVIDIA GPU; the GPU test then skips. The NEURON
tests also need Java and a C compiler (on Ubuntu: `default-jre-headless` and
`build-essential`). The
timestamp test needs the internet and only runs with `NEXUS_NETWORK_TESTS=1`.

## What's where

| Path | Contents |
|---|---|
| `CLAUDE.md` | The Phase 1 build spec and the current status box |
| `logs/phase1-v2-log.md` | The build log: every step, decision, mistake and test run |
| `nexus/format/` | File format, checker, train/test splits and locking (the M1 code) |
| `nexus/circuits/` | Runs NeuroML circuits in Jaxley and in NEURON |
| `configs/circuits/` | The reference circuit |
| `nexus/` (other folders) | Empty for now; filled in M3 to M7 |
| `tests/` | The test suite and the frozen-test hashes |
| `prereg/` | The pre-registration, once written (M2) |
| `pyproject.toml`, `uv.lock` | Exact software versions |
| `Dockerfile` | A container recipe, kept but not used; the build runs in WSL2 |
| `archive/phase1-betse/` | The earlier Phase 1 study, closed |

## The earlier study

The first Phase 1 study (a stand-in for the BETSE bioelectric simulator, August to September
2026) is closed and archived in [`archive/phase1-betse/`](archive/phase1-betse/). Its
[README](archive/phase1-betse/README.md) says what it did and did not establish. In short:
the model matched the simulator closely, but the test of whether the wiring mattered was
confounded, and it failed against real experimental data.
