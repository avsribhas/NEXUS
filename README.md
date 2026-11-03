# Dusk

Dusk (called NEXUS until 2 November 2026; the code package is still `nexus`) measures the rules that decide which parts of a nervous system switch on, given the
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
is to check that Dusk finds the switches that are there and doesn't invent ones that aren't,
before it's trusted on real recordings. Phase 1 uses graded, non-spiking circuits only, under 6 brain states.

| Step | What it is | Status |
|---|---|---|
| M0 | Archive the old study, pin the software, set up the repo | Done |
| M1 | One file format for every run, a checker, a locked test set | Done, except decisions waiting on Udeern |
| M2 | Pre-registration: fix the tests and pass marks before any held-out data exists | Approved and timestamped, version 0.2 ([`prereg/phase1-prereg.md`](prereg/phase1-prereg.md)); registered on OSF (osf.io/yrghs), private under embargo until 1 November 2030 |
| M3 | Planted circuits: switch motifs, decoys, the poke protocol, wiring errors, run files | Mostly built. The 200-circuit pilot ran (all accepted, about 1 minute and 35 MB per circuit). Every circuit type matches NEURON. Still missing: the stochastic switch (proposed to drop), twins for winner-take-all, sourced wiring error rates |
| M4 | Poke-and-predict: scoring, baselines, Dusk itself | Scoring locked before training; baselines and first versions of Dusk built and evaluated on the pilot's validation split ([log](logs/phase1-v2-log.md), [files](logs/pilot/m4/)) |
| M5 to M7 | Degraded data, energy, a real-data check | Not started (M6, energy, is out of this version's scope) |

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
- A second reference circuit with graded synapses and gap junctions (the kind worm circuits
  use) runs the same in Jaxley and NEURON, to within 0.01 mV. These connection types are safe
  to build on; spike-triggered synapses aren't, until the translator is fixed.
- The train/test split works on four kinds of unit: whole circuits, single neurons, kinds
  of poke, and brain states. Every trial of a unit lands on the same side, whatever its
  random seed.
- Training and simulation repeat exactly on the same device. Training on CPU and GPU differs
  by about 6e-08.

Done so far in M3 and M4:
- Eight kinds of graded circuit, built from small motif tables: four switch types (release,
  gain, winner-take-all, global shutoff), plain circuits, and three decoys that look like a
  switch but have no gate (shunting inhibition, adaptation and a bistable pair). Each circuit's planted
  answers are measured by probing it, and it's kept only if the measured switch behaves as
  planted.
- Every animal gets the same kinds of poke under 6 brain states with 3 noise seeds:
  50 single pokes, plus paired pokes (one input held while another is poked) and paired
  steps, so threshold and timing can be measured. 58 to 98 pokes, 1,044 to 1,764 trials.
- The scoring code was written and locked before any model was trained.
- Seven linear baselines that know the wiring, or not, and the first version of Dusk:
  one network that reads the measured responses and writes the datasheets, and one that
  predicts responses to new pokes from the wiring diagram.
- Proposed changes to the pre-registration are in
  [`prereg/revision-0.3-draft.md`](prereg/revision-0.3-draft.md). They need approval
  before any held-out data is made.

The only results so far are development numbers on the pilot's validation split (40 circuits, one training seed per model). They guide model work and are not findings. No held-out data exists yet.

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
| `nexus/circuits/` | Runs circuits in Jaxley and NEURON; motif builder, probe, wiring errors, run files, library |
| `nexus/perturb/` | The poke protocol |
| `nexus/score/` | Scoring (locked) |
| `nexus/baselines/` | Linear baselines |
| `nexus/models/` | Dusk: data loading, summary features, recovery, response model, identifiability |
| `configs/` | Reference circuits, motif tables, wiring error rates, scoring priors |
| `scripts/` | Acceptance survey, pilot, cache, pilot evaluation, locking |
| `tests/` | The test suite and the frozen-test hashes |
| `prereg/` | The approved pre-registration, its pass marks as numbers, and the timestamped locks |
| `pyproject.toml`, `uv.lock` | Exact software versions |
| `Dockerfile` | A container recipe, kept but not used; the build runs in WSL2 |
| `archive/phase1-betse/` | The earlier Phase 1 study, closed |

## The earlier study

The first Phase 1 study (a stand-in for the BETSE bioelectric simulator, August to September
2026) is closed and archived in [`archive/phase1-betse/`](archive/phase1-betse/). Its
[README](archive/phase1-betse/README.md) says what it did and did not establish. In short:
the model matched the simulator closely, but the test of whether the wiring mattered was
confounded, and it failed against real experimental data.
