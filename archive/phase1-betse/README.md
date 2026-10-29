# Archive: NEXUS Phase 1, BETSE surrogate study (August to September 2026)

This folder holds the completed first Phase 1 study: a graph neural network trained to imitate
the BETSE bioelectric simulator on sheets of non-excitable cells, predicting each cell's
voltage after 5 s. It is closed. Do not extend it, retrain it, or import from it.

The authority for this study is "NEXUS Phase 1: Technical Report (Rev 3)", committed here as
[`REV3.md`](REV3.md). A plain-language version is [`logs/research-report.md`](logs/research-report.md).
Both supersede every other document in this folder.

## What was established

These results are backed by committed artifacts (see Rev 3 sections 6 and 2):

- **C1 against the simulator.** On spatially uniform v1 tissues the MPNN reproduces BETSE to
  0.767 ± 0.012 mV (in-distribution test, mean ± sd over 3 seeds), against a target of about
  8 mV. The graph-blind MLP reaches 0.797 ± 0.016 mV.
- **C4 speed.** Median 22.3 ms on CPU and 7.4 ms on GPU against 41.3 s for BETSE, over 100
  configurations run one at a time: about 1,856x (CPU) and 5,582x (GPU), v1 model on v1 tissues.
- **The v1 null and its cause.** On v1 the MPNN and MLP are not distinguishable out of
  distribution (per-seed differences change sign). About 99% of target variance lies between
  tissues (within-tissue sd 1.33 mV, between-tissue sd 14.83 mV), so the graph had nothing to
  carry.
- **MPNN training is nondeterministic** on CPU and GPU (6 identical runs: 0.780 ± 0.013 mV);
  the MLP reproduces exactly.

## What was not established

- **C2 (graph structure is necessary) is untested.** v1 cannot test it (above).
- **v2 is confounded.** The stored per-cell inputs do not match what BETSE simulated: a median
  73.9% of per-cell input variance was never simulated (IQR 62.3% to 86.8%, 3,000 configurations
  replayed). A stand-in with zero coupling reproduces an MPNN win of 1.95x. Every v2 number is
  log-only: no v2 output was committed.
- **Experimental validation failed.** v1 MPNN error 12.46 mV against a 5.78 mV threshold on six
  published records; 5.35 mV on the two representable channel blockades; 3 of 6 sign-correct.
- **Gap-junction blockade is predicted as about 0** (+0.017 mV against a measured +18.8 mV for
  complete uncoupling). Rev 3 reads this as most likely a limit of the BETSE configuration, not
  a learning failure; the direct check (R3) was not run.

## Missing sources

Rev 3 cites `scripts/v2_realization.py` (the 73.9% measurement) and
`scripts/confound_experiment.py` (the zero-coupling stand-in) as delivered with it. Neither is
in the repository on any branch. They are recorded as absent, and the numbers they produced are
log-only until the scripts are recovered.

## Documents in this folder that overstate

These were left unedited so the record shows what was written at the time. Read them against
Rev 3.

| File | What it overstates or gets wrong |
|---|---|
| `README.phase1-original.md` | "All seven milestones complete"; reports v1 only (R9); gap-junction result described as the model having learned that junctions do not matter |
| `logs/DATASET_CARD.md` | "The MPNN's 2.81x advantage confirms the graph" hypothesis; v2 MPNN test_ood given as 1.868 (that is the abl_depth_k6 run; the headline run is 1.816); wavenumber said to be drawn per configuration (it is per channel); v2 slowness attributed to slower convergence |
| `data/synthetic/DATASET_CARD.md` | Diverged copy of the dataset card (R9) |
| `DEVIATIONS.md` and `logs/DEVIATIONS.md` | Two diverged copies (R9); the root copy reads "complete in 46 simulated seconds" where "4 to 6" lost its dash |
| `logs/director-log.md` | v2 results reported as confirming the graph hypothesis; the log is the only source for all v2 numbers |
| `logs/research-report.md` at commit d82c41a | "v2 dataset confirms the graph hypothesis; Phase 1 complete". Replaced on branch `phase1-v2` by the plain-language report; the original is in git history |

## Status of Rev 3's open items (R1 to R9)

All are closed by decision: the study is not being extended. Two are recorded here, not fixed.

| Item | Work | Status |
|---|---|---|
| R1 | Make v2 inputs match the simulation and regenerate | Closed by decision |
| R2 | Controlled C2 test on the R1 data | Closed by decision |
| R3 | BETSE-direct gap-junction control | Closed by decision |
| R4 | Commit v2 outputs and publish both datasets | **Recorded, not fixed.** No v2 artifact or dataset is in the repository or published; no v2 number can be cited |
| R5 | Deterministic training | Closed by decision (Phase 1 v2 builds determinism tests from the start) |
| R6 | v2 out-of-distribution breakdown | Closed by decision |
| R7 | Wider experimental validation | Closed by decision |
| R8 | Interior-cell error mechanism | Closed by decision; unexplained |
| R9 | Repository text consistency | **Recorded, not fixed.** See the table above |

## Layout

| Path | Contents |
|---|---|
| `REV3.md` | Rev 3 technical report (authority) |
| `logs/research-report.md` | Plain-language report from Rev 3 |
| `nexus/`, `scripts/`, `tests/` | Phase 1 code and its 92-test suite, all written by a local 7B model under a director process |
| `data/`, `outputs/` | Dataset cards, manifests and v1 run outputs (datasets themselves were never published) |
| `logs/` | Director log, deviations, dataset card |
| `nexus-phase1-prd.md`, `pytest.ini` | Original spec and test config |
| `CLAUDE.director-executor.md`, `claude-md-addendum.md` | The Phase 1 (BETSE) director-executor instructions. Renamed so a session does not load them as current instructions; the root `CLAUDE.md` says which parts still apply |
