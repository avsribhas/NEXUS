# Dusk Phase 1 pre-registration: proposed changes for version 0.3

**Status: draft, not approved.** Written by Claude on 3 November 2026 for Abhiram and
Udeern. Version 0.2 ([`phase1-prereg.md`](phase1-prereg.md)) stays in force until this is
approved, and then 0.3 is written out in full, hashed and timestamped. No held-out data
exists, so none of these changes can have been shaped by held-out results.

Each item says what changes, why, and whether it needs a decision.

## Changes that follow from building M3 and M4

1. **Name.** The program is called Dusk (it was NEXUS). The code package keeps the name
   `nexus`.

2. **Pokes per animal (section 3).** Version 0.2 says "about 50 pokes". The protocol now
   has 50 single pokes plus two new kinds, because threshold and timing couldn't be
   measured from single pokes alone:
   - **paired pokes:** one input held at 10, 20 or 30 mV for the whole trial while another
     gets the standard 15 mV, 100 ms poke, for every ordered pair of inputs and every hold
     level;
   - **paired steps:** one input held at 15 mV while another steps to 30 mV from 1500 to
     1800 ms, for every ordered pair.

   A two-input circuit gets 58 pokes (1,044 trials); winner-take-all, with four inputs,
   gets 98 (1,764 trials). Every input is always one of the single-poke targets, so each
   paired poke has an unpaired reference.

3. **Library size (section 3).** Version 0.2 fixes the count after the 200-circuit pilot.
   The pilot ran on the rig (RTX 4050, 6 GB) on 3 November 2026, 28 or 29 circuits of each
   of the 7 types built at the time (the bistable decoy came later). Files are in
   `logs/pilot/m3/`.
   - **Acceptance:** 200 of 200, a mean of 1.00 to 1.07 tries per circuit by type.
   - **Time:** 3.06 hours in all, 55 seconds per circuit on average (43 s for shunting
     decoys up to 80 s for winner-take-all). These times include Jaxley compilation; the
     pilot did not time compilation on its own.
   - **Disk:** 6.9 GB, 35 MB per circuit on average (26 to 60 MB by type).
   - **GPU:** 33% mean use in 30-second samples, above 50% in a third of them, peak memory
     5.2 GB. The run is held back by the CPU side (probe and file writing), not the GPU.

   **Proposed count: 5,000 circuits, as planned.** At pilot rates that is about 77 hours
   and 175 GB on the rig with 3 seeds; with option (b) of item 8, about 115 hours and
   260 GB. The rig's drive has 363 GB free, so disk, not time, is the limit. No cloud
   machine is needed, so the cost is the rig's electricity.

4. **Scoring scales (section 5.1).** Gain and both timing ratios are scored on a log10
   scale, threshold and reach on a linear one, with the prior ranges in
   `configs/scoring_priors.json`. This was locked and timestamped on 3 November 2026
   (`lock-scoring-0.1.json`), before any training. It belongs in the document itself.

5. **Identifiability (section 5.2).** How the step works, now that it's built:
   - one neural posterior per switch type, trained on the measured summary features of
     training circuits;
   - each circuit's interval comes from a model that did not train on it (5 folds);
   - "perfect simulated data" means voltages with recording noise but no wiring
     corruption. Corruption only changes the stored diagram, which the features don't use;
   - a switch type with fewer than 20 planted triggers in training is left undecided.

   With the full library every type has far more than 20, so this only matters in
   development.

## Fixes to the locked scoring code (need approval)

Both were found on the first end-to-end run, on a handful of development animals. Neither
changes a pass mark.

6. **A noise-ceiling sample of zero breaks the prediction bootstrap.** The bootstrap
   divides each circuit's score by a sampled ceiling. A circuit whose mean ceiling is
   just above the 0.1 floor can still have split-half samples of exactly 0, which gives
   infinity. One shunting decoy had 28 of 100 samples at 0. **Proposed:** floor each sample
   at 0.1, the floor that section 5.6 already applies to whole circuits. The order matters:
   circuits are excluded on their unfloored mean ceiling first, and only the kept circuits'
   samples are floored. Flooring first lifts every mean to at least 0.1, and then nothing is
   ever excluded.

   The pilot evaluation hit this on 40 validation circuits: 215 of 4,000 ceiling samples
   were 0, and both bootstrap intervals came out as NaN. Re-scored with the fix as stated
   (`scripts/m4_floor_check.py`, `logs/pilot/m4/floor_check.json`), the intervals are
   finite and the point estimates barely move.

7. **Tied priorities get credit.** The priority check counts a predicted tie as agreement
   whenever the planted order happens to run the other way. A model that ranks every gate
   equal, as the prior-only null does, then passes about half of all one-pair circuits,
   depending only on how the gates are labelled. Section 9 counts a passing null as a
   failure of the plan, so this has to be fixed. **Proposed:** a predicted tie counts as
   disagreement.

## Open decisions

8. **Noise seeds for the prediction test (sections 3 and 5.6).** Section 3 says 3 seeds;
   section 5.6 says each scored circuit is rerun with at least 10 for its noise ceiling.
   With 3, the ceiling is a one-against-two split, which is noisy (item 6 is one
   symptom). Options:
   - **(a)** 10 seeds for every trial on the test side of any split. Test trials include
     held-out neurons, poke types and states inside training circuits, about 54% of all
     trials, so simulation time goes up by about 2.3 times.
   - **(b)** 10 seeds for held-out circuits only, and report the other three held-out
     kinds with a 3-seed ceiling, saying so. Adds about half (1.5 times) to simulation time.
   - **(c)** keep 3 seeds and drop the "at least 10" from section 5.6.

   Recommendation: (b). It keeps the main comparison (unseen circuits) on a sound ceiling
   at a moderate cost.

9. **Wiring error rates (section 3).** Version 0.2 says "measured error rates". The
   current rates (10% of edges missing, 5% false edges, 5% sign flips, weight errors with
   a spread of 0.3 in log10 units) are placeholders with no source. They need either a
   published source or an explicit statement that they're assumptions, plus a check of
   how the results change at half and double the rates.

10. **Degenerate twins (section 5.4).** The old twins were scaled copies with identical
    voltages, so the twins test couldn't fail. The new twin search perturbs each
    capacitance, leak and conductance separately (about 25%), re-measures, and keeps the
    copy if every field lands within a tolerance of the original. "Same planted values"
    can't be met exactly by measured values, so the tolerance needs to be written down.
    At 0.1 of the prior range, with 10 tries, twins were found for 2 of 3 circuits of
    release, gain and global shutoff, and 0 of 3 of winner-take-all. Decisions needed:
    - the tolerance (proposed: 0.1 of the prior range per field);
    - what happens when no twin is found. Pairing only the circuits that found one would
      drop winner-take-all and favour circuits whose fields barely depend on their
      parameters. Proposed: smaller perturbations for winner-take-all, and the report
      gives the twin-finding rate per type next to the twins result.

11. **The stochastic switch (section 3).** The bistable decoy is now built, and every
    circuit type passes the NEURON check. The two-state stochastic switch is not built. It
    flips at random, so the probe, which measures a fixed gate ratio, can't measure its
    planted answers as it stands. Proposed: drop it from this version and say so, rather
    than delay the library for a new probe.

12. **How the NEURON check is run (section 6).** At a 0.01 ms step NEURON diverges on the
    stiff graded synapses of latched or strongly driven cells, and Jaxley in float32
    drifts at finer steps. Run in float64 at 0.0025 ms, all 8 circuit types agree with
    NEURON to within 0.02 mV (worst seen 0.007 mV). Proposed: write that method into
    section 6.

13. **The library's time step.** The library is simulated at 0.2 ms. There, voltages are
    0.03 to 0.3 mV from the converged solution for most neurons and up to 0.87 mV in the
    latched bistable pair. Planted answers are measured at the same step as the data, so
    the bench is consistent with itself, but the circuits are the NeuroML model at 0.2 ms,
    not the continuous one. Options: keep 0.2 ms and state it, or use 0.1 ms and double
    simulation time. Recommendation: keep 0.2 ms and state it; errors under 1 mV are
    small next to the 10 mV recording noise.

14. **D7, the held-out key holder.** Udeern holds the key (Abhiram, 3 November 2026).
    Held-out answers are encrypted with age (X25519) to Udeern's public key inside the
    generator. Udeern keeps the private key and decrypts once, after predictions are hashed
    and timestamped. Still needed: Udeern's public key (the `age1...` line). Nothing held out
    is generated until it arrives.
