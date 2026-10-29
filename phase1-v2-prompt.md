# NEXUS Phase 1 (v2): build the switch-finding test bench

You are working in the NEXUS repository (github.com/avsribhas/NEXUS) for s9m Labs, in a session on Abhiram's Mac. Read this whole prompt before writing any code. In M0 you save it to the repo as `CLAUDE.md` so every session starts from it.

Section 0 lists decisions that are proposed but not yet confirmed. Do not freeze any test that depends on an unconfirmed decision.

## 0. Before any work

### 0.1 Environment check
1. You run on the Mac. The GPU machine is the Windows laptop with an RTX 4050 (6 GB), reachable over Tailscale. Verify, in order:
   - `tailscale status` shows it.
   - SSH works (old values were host `abhiram-lenovo`, user `windows-rig`; confirm with Abhiram if they fail).
   - `nvidia-smi` runs on it.

   If any step fails, stop and report. Do not guess hostnames.
2. CPU work (format, checker, tests, generators at small scale) runs on the Mac. GPU work (Jaxley batches, PyTorch training, GPU reproducibility tests) runs on the Windows machine over SSH, inside the same pinned container if Docker with GPU support is available there; otherwise in a pinned conda env, with the gap logged.
3. Never store credentials in the repo.

### 0.2 Decisions to confirm with Abhiram (and Udeern where marked)
Ask these as one message at the start. Each item has a proposed default. Work that does not depend on an item may proceed, but no test that depends on an unconfirmed item is frozen.

**D1. Branch.** Create `phase1-v2` from `main` and push only there.

**D2. Archive source.** The archive README is written from "NEXUS Phase 1: Technical Report (Rev 3)" (Drive file 1824DW5JhZtBmGme6YAtby_EkEKiRD5LfsDF--Zg7Chg). It supersedes `logs/research-report.md` and the README in the repo, which still say v2 "confirms the graph hypothesis".
- Commit Rev 3 as `archive/phase1-betse/REV3.md`.
- Rev 3 cites `scripts/v2_realization.py` and `scripts/confound_experiment.py`, which are not in the repo on any branch. Ask Abhiram for them. If they cannot be found, record them as absent.
- Ask which of Rev 3's open items R1 to R9 are closed by decision. Default: all closed by decision, except R4 (reproducibility flag) and R9 (doc errata), which are recorded in the archive README, not fixed.

**D3. Six fields** (Abhiram and Udeern). The default is one datasheet per gate; a circuit carries a list of zero or more datasheets.

| Field | Type | Units | Planted as | Scored by |
|---|---|---|---|---|
| Trigger | categorical identity (which input or state channel) plus a gain | gain dimensionless | channel ID plus gain value | exact match on identity, interval coverage on gain |
| Threshold | number | units of the trigger signal | value | interval coverage |
| Timing | two ratios: tau_on / tau_ref and tau_off / tau_ref | dimensionless | values | interval coverage |
| Reach | fraction of circuit neurons whose activity changes by more than a stated amount when the gate flips | 0 to 1 | value | interval coverage |
| Priority | rank among gates in the circuit | ordinal | rank | pairwise order agreement |
| Cost | computed by M6 from ion flow, not planted | ATP per flip | null until M6 | interval coverage after M6 |

- tau_ref is the trigger time constant when the trigger is smooth. For step triggers and stochastic switches, tau_ref is the median membrane time constant of the gated population; state this in the schema.
- For the stochastic switch, Timing is the two mean dwell times over tau_ref, and Threshold is the input level where the on and off rates are equal.

**D4. Brain state and animal** (Abhiram and Udeern).
- A brain state is a slow exogenous modulatory input at 4 discrete levels that scales one or more conductances or gains. It is distinct from Trigger.
- For the global-shutoff switch, the slow integrator is the Trigger and brain state is a separate input; flag this case in the log.
- A simulated "animal" is one wiring-corruption draw of a base circuit.
- A "trial" is one poke under one state with one noise seed. All seeds and repeats of a unit stay on one side of any split.

**D5. Format.**
- NWBelegans is followed as naming and NeuroPAL ID conventions. Its `ndx-multichannel-volume` extension is not required, because it pins pynwb < 3, which conflicts with `ndx-optogenetics` 0.4.x (pynwb >= 3.1, the only version with the pulse table).
- Pin `ndx-optogenetics==0.4.1` and pynwb 3.x exactly. Unpinned, pip silently falls back to 0.2.0, which has no pulse table.
- A simulated poke uses the `ndx-optogenetics` epoch and pulse tables with an effector named `simulated_current_injection`, plus custom columns `target_neuron`, `amplitude`, `amplitude_unit` and `modality`.
- DANDI 001075 appears to store stimulation with `ndx-patterned-ogen`. Verify by reading one asset, then write an adapter into the same table.

**D6. Simulator and circuit format.**
- Jaxley is primary; Brian2 is the spot-check.
- NeuroML2 is the source of truth for circuit structure and parameters. Jaxley has no NeuroML2 reader, and pyNeuroML exports to NEURON and Brian2, not Jaxley, so a NeuroML2-to-Jaxley translator is an explicit M1 deliverable.
- Evaluate `tarjuman` (PyPI, single maintainer, released 2026-10-14, partial c302 coverage) pinned to an exact version. If it fails the equivalence tests, write an in-house translator limited to a whitelist of component types.
- Equivalence test: the same NeuroML2 circuit run in Jaxley and in Brian2 (exported through pyNeuroML) matches to a stated tolerance.
- The General Software Protocols name NEURON and Brian2 as simulators, not Jaxley. Confirm with Udeern that Jaxley as primary is intended.

**D7. Held-out key.** Ask who holds the key (name and public key). Default: held-out labels AND held-out response targets are encrypted to that person's public key inside the generator (age or GPG). Until a key holder is named, test the mechanism with a placeholder key and generate no held-out data.

**D8. P2 loss.** Research Phases says P2 is "beats the wiring-only baseline on unseen nudges, or we learn that it doesn't". Default: switch recovery and refusing false switches are the stop gate. A P2 loss is a reported result, not a stop.

**D9. Creamer number.** M7 originally said "about 0.82 relative correlation". The bioRxiv v1 abstract (10.1101/2024.09.22.614271) reports 0.92 of the perturbation-reproducibility ceiling; a later version could not be checked. Ask where 0.82 comes from. Until answered, the target is "the metric and number in the current version of the paper", and the log states which version you read.

## 1. What NEXUS is now

NEXUS measures the rules that decide which parts of a nervous system switch on, given the animal's state. Each rule is written as a **datasheet** with six fields: Trigger, Threshold, Timing, Reach, Priority, Cost. Types, units and scoring are in D3. Every continuous field is reported as an interval, never a single value.

Phase 1 runs before any funding and before any wet lab. Its job is to prove NEXUS works as a tool on circuits where the right answer is known, and to prove real recordings flow through the same code. The Phase 1 checkpoint: **NEXUS finds planted switches in simulated circuits and refuses to invent switches that are not there.** Whether NEXUS beats a fitted wiring-only model on unseen state-dependent nudges is reported either way (D8). If switch recovery or refusal fails, fix the tool before anything else.

## 2. What happened to the old Phase 1

The September 2026 Phase 1 is a completed study. It trained an MPNN surrogate of the BETSE simulator on a generic non-excitable cell sheet (regular hexagonal lattice, BETSE "mammal" ion profile; Xenopus and planarian data appear only as validation targets) to predict Vmem after 5 s of equilibration.
- Per Rev 3: the v2 graph-advantage result is confounded, because a median 73.9% of per-cell input variance never reached the simulator. Claim C2 (graph structure is necessary) is untested.
- It cannot fill any datasheet field: no switching, no neurons, inputs are simulator parameters.
- Do not extend it, retrain it, or import from it.

Carry forward these four methods, rewritten for the new data:
1. The within-graph to between-graph target variance diagnostic. Run it on every new dataset; a near-zero ratio means the graph is not being tested.
2. A check that stored inputs match the inputs that generated the labels. The old repo did not have this check, which is how the v2 confound got through. Build it new, as a frozen test that runs on every generator.
3. The matched-baseline differential validation protocol (`scripts/validate_experimental.py` in the old repo).
4. The frozen-test-suite build method: tests are written and frozen before implementation.

The old director-executor workflow (Claude directs, a local 7B model writes all code) is retired. You write the code.

## 3. Source documents

The specs below are condensed from these. If a detail here conflicts with them, stop and ask.
- Research Phases (Udeern): https://docs.google.com/document/d/1QCkegr6ORFBdGYzSdxZedV_ex67P5Pln3gQt9RSdJ8I/edit
- General Software Protocols (P0 to P6): https://docs.google.com/document/d/1chycTVj4ZeA026a0ltFaueepmuKm-8AJ4D-CM2NebFw/edit
- Engineering Plan, Steps 1 and 2: https://docs.google.com/document/d/1Ke1jTZ4c4mQgre4VvW0cjI6WeNfiOwuTJKegvTZsenc/edit
- Literature reviews P0 to P6: https://drive.google.com/drive/folders/1WTZUCJ02aWMqunf1LGXeo71hkdAQD6Qk
- Rev 3 technical report (authority for the archive only): https://docs.google.com/document/d/1824DW5JhZtBmGme6YAtby_EkEKiRD5LfsDF--Zg7Chg/edit

Known conflict inside the sources: GSP P1 step 1 plants all six fields, while GSP P6 computes Cost from ion flow. This prompt follows P6 (D3).

## 4. Scope

In scope: P0, P1, P2, P3, a first pass of P6, and one real-data anchor. Out of scope: P4 (cross-simulator transfer), P5 (transfer across circuits), the dense and MoE language-model baselines (Suraj's Engineering Step 1 track), the language-model half of the leak test (Engineering Step 3), P6 chip export (Phase 2), and anything wet lab.

## 5. Milestones

Work through these in order. Each milestone ends with frozen tests passing and a short entry in `logs/phase1-v2-log.md`. Write each milestone's tests before its implementation and freeze them, but only after the decisions they depend on are confirmed.

### M0. Repo restructure
- On branch `phase1-v2`, move all old Phase 1 code, data pointers, figures, logs and docs into `archive/phase1-betse/`.
- Archive the old `CLAUDE.md` and `claude-md-addendum.md` as `archive/phase1-betse/CLAUDE.director-executor.md` and `archive/phase1-betse/claude-md-addendum.md`. Never leave a file named `CLAUDE.md` inside `archive/`: it would load the retired workflow as instructions.
- Write `archive/phase1-betse/README.md` from Rev 3. It covers:
  - what was established: C1 met against BETSE on spatially uniform v1 tissue, about 0.77 mV over 3 seeds; C4 speedup about 1,856x on CPU on v1;
  - what was not: C2 untested; v2 confounded; experimental validation failed (12.46 mV against a 5.78 mV threshold, v1 model); gap-junction blockade predicted as about 0;
  - a list of the in-repo documents that overstate (research-report "confirms", "Phase 1 COMPLETE"), so readers know Rev 3 supersedes them;
  - the status of R1 to R9 per D2.
- Replace the root `README.md` on the branch with a short pointer: what NEXUS is now, where the archive is, and that results are not public yet. `main` and the public README do not change until Abhiram approves a merge.
- Save this prompt as the root `CLAUDE.md`.
- New layout: `nexus/format`, `nexus/circuits`, `nexus/perturb`, `nexus/models`, `nexus/baselines`, `nexus/degrade`, `nexus/energy`, `nexus/realdata`, `nexus/score`, `prereg/`, `configs/`, `tests/`, `logs/`.
- Pin everything in a container (Dockerfile, base image pinned by digest) plus `pyproject.toml` with exact versions. Python 3.11 or 3.12. JAX for Jaxley, PyTorch for learned models.
  - First record which combinations actually resolve (pip dry-runs for pynwb, ndx-optogenetics 0.4.1, dandi, nwbinspector, jax, jaxley, brian2, pyNeuroML, sbi, torch). Log CUDA version conflicts between JAX and PyTorch in one image.
  - `leifer_lab_to_nwb` pins Python 3.12 and numpy < 2, which conflicts with current JAX. Keep it out of the main image (separate env, used only if needed).

### M1. P0: data format and test locking
- Every run, simulated or real, is an NWB file, with pokes stored as in D5.
- Circuits are defined in NeuroML2, plus the translator and equivalence tests (D6).
- Ground-truth extension:
  - One small NWB extension holds planted ground truth in its own group: a list of datasheets per D3 (Cost nullable), switch type, decoy flag, degenerate-twin ID, and the wiring-corruption record.
  - Check the NWB extension catalog first; write a custom one only if nothing fits.
  - Add a frozen test that round-trips one file using every extension in use.
- Every file stores schema version, simulator version, pynwb version, seed, thread count, GPU model, driver and git commit. GPU fields may be null for CPU runs.
- A checker rejects any simulated run missing ground truth (except Cost before M6) or metadata.
- Stored-input check (method 2): a frozen test asserts that the parameters written to the file equal those the simulator received.
- Split manifest. Each held-out kind has its own split unit:
  - whole circuits: circuit ID;
  - neurons: neuron IDs within training circuits;
  - nudge types: nudge-type labels;
  - brain states: state labels.

  Keep every trial, repeat and seed of a unit on one side of each split. Carve a validation split from training circuits that mirrors all four kinds; model and architecture decisions use only the validation split.
- Locking:
  - SHA-256 hash the manifest, the scoring script, the thresholds and (after M5 prep) the identifiable-field list. Stamp each with OpenTimestamps.
  - Held-out labels and response targets are encrypted per D7.
  - Log every access to the held-out set and enforce the access limit set in the pre-registration.
- Reproducibility:
  - PyTorch reproduces bitwise with the same seed on the same machine (deterministic mode, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Test on CPU on the Mac and on GPU on the Windows machine.
  - Jaxley reruns are checked to a stated tolerance.
  - Test across the two machines and record what happens instead of assuming.

### M2. Pre-registration (gate: stop here for approval)
Write `prereg/phase1-prereg.md` before any held-out data is generated. It fixes the tests, metrics, thresholds, and what counts as pass, fail, tie and inconclusive. Propose these defaults for approval:

- **Recovery.** A planted continuous field is recovered when the NEXUS 90% interval covers the planted value AND its width is at most a pre-registered fraction of the prior range (propose 0.5).
  - Identity and rank fields are scored per D3.
  - Pass: at least 80% of switch circuits have all identifiable behavior-level fields recovered. Cost is excluded until M6.
  - Also report empirical 90% coverage and an interval score (or CRPS) for NEXUS and for sbi.
  - Include a prior-only estimator as a null row; it must fail.
- **Identifiability.** Pre-register the method (profile likelihood or SBI posterior on perfect simulated training data), the cutoff, and the unit. Compute the identifiable-field list on training data and hash it before any scoring.
- **No false switches.** At least 90% of no-switch and decoy circuits get "no switch". This matches Engineering Plan Step 2.
- **Degenerate twins.** Twins share planted six-field values and differ in biophysical parameters. A twin pair matches when both datasheets meet the Recovery rule and their intervals overlap. Pre-register the pass fraction (propose 80%).
- **Leak.** Close a planted gate in the circuit and measure the share of its direct effect on downstream activity that other paths restore, using resample ablation and path patching on the circuit. Report the distribution over gates and pokes; the median must be at most 10%. State that this measures the generator's redundancy, not NEXUS's ability, and report it as a property of the library.
- **Prediction (P2).**
  - Score as a fraction of the noise ceiling. Estimate the ceiling by rerunning each circuit with at least 10 new noise seeds and using a split-half estimator with Spearman-Brown correction.
  - Exclude circuits whose ceiling falls below a stated floor (propose 0.1), and report the excluded count.
  - Carry the ceiling's uncertainty into the bootstrap.
  - The comparator is baseline 1, both pooled and per-state; NEXUS must beat both.
  - Pass: a bootstrap 95% CI over circuits of the paired difference excludes zero, on held-out state-dependent nudges.
  - Report all four held-out kinds.
- **Tie rule.** If baseline 4 (linear model with a state-by-input term) comes within a pre-registered margin of NEXUS (propose 0.02 of the noise ceiling) on state-dependent nudges, there is no switching result to report, and the log says so. A tie overrides a P2 pass.
- **Mixed outcomes.** Report results per switch type. Pre-register what happens when some types pass and others fail (propose: Phase 1 passes only for the passing types, and the report says which).
- **Exclusions.** Pre-register exclusion rules for diverging runs and runs that fail the checker, and the number of held-out circuits per switch type and per held-out kind.

Hash and OpenTimestamp the document. Then stop and send Abhiram and Udeern the thresholds for approval. Do not open or generate held-out data before approval. Abhiram does the OSF registration, not you.

### M3. P1: circuits with planted switches
Build one generator per switch type. Each starts from a published model, takes the D3 fields as inputs, and samples parameters from published and measured ranges, not round numbers:
- **Release (disinhibition):** Gurney-Prescott-Redgrave basal ganglia form; thalamic motifs per Halassa and Sherman 2019.
- **Volume knob (gain):** Servan-Schreiber, Printz and Cohen 1990; Aston-Jones and Cohen 2005 locus coeruleus form.
- **Winner-take-all (priority):** mutual-inhibition form per Wang 2002 and the Seeds et al. 2014 fly grooming hierarchy.
- **Global shutoff:** sleep flip-flop per Saper, Chou and Scammell 2001, driven by a slow integrator; worm version per Nichols et al. 2017.
- **Two-state stochastic switch** with input-dependent rates; worm switches flip probabilistically with exponential dwell times (Flavell et al. 2013).

Also:
- Worm-like circuits use graded, non-spiking neurons (OpenWorm c302 as reference). Fly-like and mouse-like circuits spike.
- One third to one half of circuits have no switch. Make them adversarial: decoys that mimic a switch through adaptation, shunting inhibition, or recurrent bistability with no gate. Add degenerate twins per M2.
- Wiring corruption from measured error rates: proofreading and synapse detection (Scheffer 2020, Buhmann 2021), transmitter prediction accuracy (Eckstein 2024), animal-to-animal variation (Witvliet 2021). Corrupt signs and weights, not only edges.
- Randomize size, density and noise.
- Jaxley mechanics:
  - Jaxley channels receive no PRNG key, so draw all noise in advance from recorded seeds and feed it in through `data_stimuli`. Drive the stochastic switch's state through `data_clamps`.
  - Add a fixed-seed reproducibility test and a KS test that dwell times are exponential.
  - The Brian2 spot-check uses the same noise arrays (TimedArray) and compares traces to a stated tolerance.
- Validation test for every switch type: open versus closed behave differently as labeled, and decoys contain no gate.
- Variance diagnostic (method 1) on the library.
- Library target: about 5,000 circuits x 50 pokes x 4 states x 3 seeds.
  - Jaxley compiles once per distinct circuit structure. Build the generator around padded size buckets, or pack several circuits into one disconnected network with absent synapses masked to zero.
  - Run a 200-circuit pilot on the Windows GPU first. Report compile time, run time per circuit and GPU utilisation separately, then re-estimate total cost before scaling.

### M4. P2: poke and predict
- Nudge tool: gentle to strong single-neuron pokes, two competing signals at once, and the same pokes under different brain states.
- Baselines, all trained on the same training split:
  1. Fitted connectome-constrained linear dynamical model in the style of Creamer, Leifer and Pillow, pooled and per-state.
  2. Synapse-count predictor.
  3. Wiring-blind model, plus a fully connected and a shuffled-connectome control.
  4. Linear model with a state-by-input interaction term.
  5. Simulation-based inference (`sbi`) for recovering the fields. It returns posteriors, so it shows which fields cannot be recovered at all.
- NEXUS model: takes activity, the perturbation and the corrupted connectome. It outputs (a) the predicted response and (b) either "no switch" or a list of datasheets with intervals.
  - Start with the simplest architecture that can represent state-dependent gating. Add complexity only when a validation-split test shows the simple version fails.
  - Log every architecture decision and why.
- Hash and timestamp NEXUS predictions before any scoring.
- Score all models in one table per held-out kind. Bootstrap error bars over circuits, not neurons. Minimum three seeds per model.
- Report the no-switch false-positive rate. Cite Das and Fiete 2020 as context on spurious mechanism inference from connectivity; it gives no switch false-positive rate to compare against.

### M5. P3: degrade the data
- Levels:
  1. perfect voltage and current;
  2. spike times for fly-like and mouse-like circuits, graded voltage for worm-like circuits;
  3. calcium imaging;
  4. calcium plus corrupted wiring;
  5. a reference row with the frame rate and noise of the best public data (Randi 2023 and Atanas 2023 for worm).
- Calcium converter parameterized by indicator (GCaMP6, jGCaMP8) from published kinetics, with saturation, photobleaching, noise and real frame rates.
- Use the identifiable-field list hashed in M2. Fields not identifiable from perfect data are marked and NEXUS is not charged for them.
- Output: a six-fields x levels table. Each cell gives the M2 recovered fraction, empirical coverage, and posterior width relative to the prior range. The Cost row covers only switch types that M6 covers.

### M6. P6: energy, first pass
- Validate the pipeline first: reproduce published energy costs for the Alle et al. 2009 mossy fiber and Hallermann et al. 2012 cell models within a stated tolerance.
  - Carter and Bean 2009 is an experimental study with no released model. Use its per-cell-type Na+ entry ratios as a check against a named HH model, after confirming the values against the full text.
  - If validation fails, stop and report; nothing downstream uses an unvalidated pipeline.
- Spiking cells: sodium-to-ATP accounting (3 Na+ per ATP).
- Worm-like graded cells: calcium and potassium accounting, with stoichiometries stated explicitly. This route has no published validation target, so every worm number is labelled unvalidated until a bioenergetics expert signs off. Verify against current WormBase that the worm lacks voltage-gated sodium channel genes before relying on it.
- Report spike cost and synaptic cost separately (Harris, Jolivet and Attwell 2012), as a range across states, with units stated.
- State the accounting boundary: list what the ion count excludes (resting potential, calcium handling, vesicle cycling, glia) and the assumption for each.
- Milestone: energy per flip for at least one switch type, which must be a fly-like or mouse-like spiking switch.

### M7. Real-data anchor
- Load DANDI 001075 (Randi et al. 2023, worm signal propagation atlas) through the same NWB code path as the simulations. Read it with pynwb using the namespaces cached in the files; confirm the stimulation extension (D5).
- Reproduce Creamer et al.'s connectome-constrained result (D9).
  - The authors' code fits Randi 2023 data from OSF (osf.io/qxhjd), not from DANDI 001075. Reproduce on OSF first, then on 001075, and report both.
  - Use the paper's metric (fraction of the split-half ceiling on the stimulus-triggered matrix) and record the neuron count used (the paper says 156; released matrices are 154 x 154).
  - Include the unconstrained and shuffled-connectome controls. Report the number you get, whatever it is.
- Write loaders for DANDI 000776 (Atanas 2023) and 000727 (Brezovec fly). Confirm they load and pass the checker. Do not train on them in this phase.
- Check and record the license of every dataset and tool used (wormneuroatlas is GPLv3; some datasets have unstated licenses).

## 6. Compute

- Develop and test on the Mac (CPU) and the Windows laptop (GPU, over Tailscale). Use cloud GPUs only for runs whose results will be kept.
- Cloud is Google Cloud on the Antler credits. Write SkyPilot task files with auto-stop and spot instances, checkpointing so a reclaimed machine loses minutes. L4 for debugging, A100 for the circuit library.
- Expected totals from the Engineering Plan: circuit library 45 to 450 A100-hours. Re-estimate from the M3 pilot, including compile time, before scaling.
- Gate: before the first cloud launch, and before any job estimated above $200, stop and send Abhiram the task file, machine type, hours and dollar estimate. Do not create cloud projects, billing accounts or quota requests yourself.
- Track every run (config, commit, seed, metrics, cost) in self-hosted MLflow; version data with DVC. Ask Abhiram where the MLflow server and DVC remote live before first use. Commit all outputs that results depend on.

## 7. Rules

- Never look at held-out ground truth before predictions are hashed.
- Publish results whether they pass or fail. A failed checkpoint is a result; write it up with the same care.
- No novelty claims in any file: no "first", "novel", "nobody has", "unprecedented". Cite the closest prior work instead (Dag et al. 2023, Creamer et al., Randi et al. 2023, Das and Fiete 2020, the sbi literature).
- Every number in a report comes with its range or CI and the seed count behind it.
- Every number in a report comes from a committed artifact. Otherwise mark it "log-only".
- Stop and ask when a spec here is ambiguous, contradicts a source doc, or a test can only pass by weakening it. Never edit a frozen test to make it pass; log the conflict.
- Do not push to `main`, change the public README on `main`, or make new results public without Abhiram's go-ahead.
- Writing style for every log and report: no em dashes, no filler. Lead with what the result shows and why it matters, then the numbers. Test counts are not results.

## 8. Final deliverable

`logs/phase1-v2-report.md`, structured as:
1. What Phase 1 shows. One paragraph: did NEXUS find planted switches and refuse decoys, and did it beat the wiring-only model on state-dependent nudges.
2. The results:
   - P1 recovery and false-switch results, per switch type;
   - the P2 table per held-out kind;
   - the P3 recoverability table;
   - the P6 energy number with its validation;
   - the M7 real-data reproduction.
3. What failed and what it means for Phase 2.
4. What carries into Phase 2 (public worm and fly data, first datasheets).
5. Compute used and cost.
6. Every deviation from this prompt and the pre-registration, with the reason.
