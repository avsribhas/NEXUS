# NEXUS Phase 1 pre-registration

**Status: version 0.2, approved 2 November 2026** by Abhiram Sribhashyam and Udeern
Varanasi (Udeern's approval relayed by Abhiram). Written by Claude. Registered privately on
OSF at Abhiram's request. No held-out data exists. None will be generated until the
held-out key holder is named (section 10, item 6).

This document doesn't change from here. Any later change is a new, dated, timestamped
version that says what changed and why. Earlier versions stay in the repository's history
(draft 0.1 is commit f6747f6).

**Changes from draft 0.1:** 6 brain states instead of 4 (section 3 and 4); Phase 1 uses
non-spiking circuits only (section 3); approvals recorded (sections 10 and 11).

## 1. What Phase 1 tests

NEXUS reads a circuit's activity, the pokes applied to it and an imperfect wiring diagram.
For each circuit it says either "no switch" or gives a datasheet for every switch it finds:
Trigger, Threshold, Timing, Reach, Priority and Cost. Each continuous field comes as an
interval, not a single number.

Phase 1 runs NEXUS on simulated circuits where the right answers were planted on purpose,
and asks four questions:

1. Does it find the switches that are there, with the right values? (Recovery)
2. Does it refuse to invent switches that aren't? (No false switches)
3. Do two circuits built differently but with the same switch get the same datasheet?
   (Degenerate twins)
4. Does it predict what an unseen poke will do better than a model that only knows the
   wiring? (Prediction)

Questions 1 and 2 are the gate: if either fails, the tool gets fixed before anything else
(Phase 1 checkpoint). Question 4 is reported whether it passes or fails, and a loss doesn't
stop the program (decision D8).

## 2. Fixed before any held-out data exists

These are written, hashed and timestamped before the held-out set is generated:

- this document;
- `prereg/thresholds.json`: every pass mark below, as numbers;
- the split manifest (section 4);
- the scoring script (written in M4, before NEXUS is trained);
- the list of fields that can be recovered at all (section 5.2), computed on training data
  only.

NEXUS's predictions on the held-out set are hashed and timestamped before any scoring. The
held-out answers are encrypted to a named key holder (section 7) and can't be read before
then.

## 3. The circuit library

- **Size.** About 5,000 circuits, each with about 50 pokes, under 6 brain states, with 3
  noise seeds. The build spec planned 4 states; 6 adds half again to the simulation cost.
  The exact count is fixed after the 200-circuit pilot and added here, as a new version,
  before generation.
- **Switch types.** Five, each built from a published model: release (disinhibition),
  volume knob (gain), winner-take-all, global shutoff, and the two-state stochastic switch.
- **Circuits with no switch.** Between a third and a half of the library: plain circuits,
  plus decoys that look like a switch (adaptation, shunting inhibition, or bistability with
  no gate).
- **Degenerate twins.** Pairs with the same planted datasheet values and different
  biophysical parameters.
- **Wiring corruption.** Each simulated animal is one draw of wiring errors (signs, weights
  and edges) at measured error rates. A circuit's "animal" is that draw (D4).
- **No spiking circuits in this phase (Abhiram, 2 November 2026).** Every circuit uses
  graded, non-spiking neurons connected by graded synapses and gap junctions, the way worm
  circuits work. Switch types whose published models spike (for example winner-take-all,
  after Wang 2002) are built as graded versions, and the report says so. Two reasons: the
  team wants the non-spiking results first, and the M1 checks found that the translator
  from NeuroML to Jaxley (tarjuman 0.5.1) drops spike-triggered synapses, while graded
  synapses and gap junctions match NEURON to within 0.01 mV. Spiking circuits come in a
  later version of this plan.
- **Consequence for energy (M6).** The build spec's energy milestone needs one spiking
  switch, and the worm energy route has no published check. M6 is out of this version's
  scope, and Cost is not scored in it.

## 4. What is held out

Four kinds of unseen material, each split on its own unit. Every trial, repeat and noise
seed of a unit sits on the same side of the split.

| Held-out kind | Unit | Share held out | Share for validation |
|---|---|---|---|
| Whole circuits | circuit ID | 20% | 20% |
| Neurons | neuron ID, within training circuits | 20% | 20% |
| Nudge types | nudge-type label | 20% | 20% |
| Brain states | state label | 20% | 20% |

- At least one unit of each kind goes to test and one to validation, and at least three
  units of each kind must exist.
- A trial counts as test data if any of its four units is held out.
- Model and design choices use the validation split only.
- **With 6 brain states, this leaves 4 for training, 1 for validation and 1 for testing.**
  (Draft 0.1 had 4 states, which left only 2 for training.)
- **Held-out circuits per switch type:** at least 100. With 100 circuits, an observed pass
  rate of 80% has a 95% interval of about 71% to 87%, which is enough to tell 80% from
  70%. With about 3,000 switch circuits over five types and 20% held out, each type gets
  about 120.

## 5. Tests and pass marks

All numbers are also in `prereg/thresholds.json`. Where the two disagree, this document
wins, and the disagreement is a bug to fix before approval.

### 5.1 Recovery

- A continuous field counts as recovered when NEXUS's 90% interval contains the planted
  value **and** the interval is no wider than **0.5** of that field's prior range. A wide
  interval that happens to contain the answer doesn't count.
- Trigger identity is scored by exact match, and its gain by the interval rule. Priority is
  scored by pairwise order agreement with the planted ranks (D3).
- **Pass:** in at least **80%** of held-out switch circuits, every identifiable field (5.2)
  is recovered. Cost is left out until the energy work (M6) can compute it.
- Also reported, for NEXUS and for `sbi`: the share of 90% intervals that actually contain
  the answer (it should be near 90%) and an interval score (CRPS).
- **Null row.** An estimator that ignores the data and returns the prior must fail this
  test. If it passes, the test is too easy, and that's reported.

### 5.2 Which fields can be recovered at all

- Method: run `sbi` on perfect simulated training data. A field is identifiable for a switch
  type when its 90% posterior interval is no wider than 0.5 of the prior range in at least
  80% of that type's training circuits.
- Unit: one field for one switch type.
- The list is computed on training data only, then hashed and timestamped before any
  scoring. NEXUS isn't charged for fields on the "not identifiable" list, and the report
  names them.

### 5.3 No false switches

- **Pass:** at least **90%** of held-out no-switch and decoy circuits get "no switch".
  Reported for plain circuits and decoys separately, and for each decoy kind.

### 5.4 Degenerate twins

- A twin pair matches when both datasheets meet the recovery rule and their intervals
  overlap on every identifiable field.
- **Pass:** at least **80%** of held-out twin pairs match.

### 5.5 Leak

- Close a planted gate, then measure how much of its direct effect on downstream activity
  other paths restore (resample ablation and path patching).
- Reported as a distribution over gates and pokes. The median must be at most **10%**.
- This measures how redundant the generated circuits are, not how good NEXUS is. It's
  reported as a property of the library.

### 5.6 Prediction (P2)

- Each circuit is rerun with at least **10** new noise seeds to estimate its noise ceiling,
  using a split-half estimate with Spearman-Brown correction. Scores are fractions of that
  ceiling.
- Circuits with a ceiling below **0.1** are left out of this test, and the count left out
  is reported.
- The uncertainty in each ceiling is carried into the bootstrap.
- The comparison is against baseline 1 (a fitted wiring-constrained linear model, as in
  Creamer, Leifer and Pillow), both pooled over states and fitted per state. NEXUS has to
  beat both.
- **Pass:** on held-out, state-dependent nudges, the 95% bootstrap interval (over circuits)
  of the paired difference between NEXUS and baseline 1 excludes zero, in NEXUS's favour.
- All four held-out kinds are reported, along with the other baselines (synapse count,
  wiring-blind, fully connected, shuffled wiring, and the state-by-input linear model).

### 5.7 Tie rule

- Scores are fractions of the noise ceiling. If baseline 4 (a linear model with a
  state-by-input term) scores within **0.02** of NEXUS on state-dependent nudges, there's no
  switching result to report, and the report says so. A tie overrides a prediction pass.

### 5.8 Mixed results

- Results are reported per switch type. Phase 1 passes only for the types that pass, and
  the report names the ones that fail.

### 5.9 Exclusions

Decided before generation. Excluded runs are counted and reported per switch type and per
reason.

- A run is excluded if any voltage is NaN or infinite, if any voltage leaves -150 to +80 mV,
  or if the run fails the file checker.
- A circuit is excluded if more than 10% of its runs are excluded.
- If more than 5% of the circuits of one switch type are excluded, that type's generator
  gets fixed and the type is regenerated before any scoring.
- Nothing is excluded after predictions are hashed.

## 6. Reproducibility

Measured in M1. Every reported number names the device it came from.

| What | Same device | Across devices |
|---|---|---|
| PyTorch training | bit for bit | CPU against GPU differ by up to 6e-08 per parameter; tolerance **1e-6** |
| Jaxley simulation | bit for bit (GPU, float32) | to be measured before generation; tolerance stated then |
| Jaxley against NEURON, spike times | | within **one time step** (0.01 ms) |
| Jaxley against NEURON, graded voltage | | within **0.02 mV** |

Every simulated run must also store, in its file, the inputs the simulator actually
received. A frozen test reads them back and compares them for every generator (the check
that would have caught the old Phase 1 confound).

## 7. Locking and access

- The files in section 2 are hashed into one lock file and timestamped with OpenTimestamps.
  One command checks later that none of them changed.
- Held-out answers and held-out response targets are encrypted to the key holder's public
  key when they're generated. Until a key holder is named, no held-out data is generated.
- Every opening of the held-out set goes through one function that logs who, what and why,
  and stops at the limit. A failed opening counts.
- **Access limit: 3 openings.** One for the scoring run, one for a rerun after a bug in the
  scoring code (the bug and the fix are logged first), and one spare. Using the spare is
  reported.
- Results are published whether they pass or fail.

## 8. The energy field (Cost)

- Every switch's datasheet must carry a Cost value; the checker rejects files without one
  (P0 protocol). Until the energy pipeline (M6) exists, generators record a blank Cost and
  log each time they do.
- Cost is scored only after M6, and only for switch types M6 covers.

## 9. What would count as a failure of this plan itself

Each of these would be reported, not hidden:

- the prior-only null row passes recovery (the test is too easy);
- fewer than 100 held-out circuits survive exclusions for a switch type;
- the identifiability step marks most fields "not identifiable" even on perfect data (the
  generators don't express the fields);
- any lock check fails, or an opening of the held-out set is missing from the log.

## 10. Open before approval

Settled on 2 November 2026 (Abhiram, with Udeern's agreement relayed by Abhiram):

1. **D3:** the six fields, their types, units and scoring, as in section 5.1. Accepted.
2. **D4:** a brain state is a slow input, now at 6 levels; an animal is one wiring
   corruption draw; a trial is one poke under one state with one seed. Accepted.
3. **Brain states:** 6 (4 for training).
4. **D6:** Jaxley as the main simulator, with NEURON 9.0.2 as the check. Accepted.
5. **Spiking circuits:** not in this version. Graded circuits only.
7. **The numbers** in sections 4 to 7: accepted as proposed.

Still open:

6. **D7:** who holds the held-out key (name and public key). Held-out data waits for it.

## 11. Approval

| Name | Role | Decision | Date |
|---|---|---|---|
| Abhiram Sribhashyam | Founder | Approved | 2 November 2026 |
| Udeern Varanasi | Head of Research | Approved (relayed by Abhiram) | 2 November 2026 |
