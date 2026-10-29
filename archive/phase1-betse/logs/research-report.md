# NEXUS Phase 1 (BETSE study): what we did and what we learned

October 29, 2026. This is the readable version of "NEXUS Phase 1: Technical Report (Rev 3)",
which is committed next to it as [`../REV3.md`](../REV3.md). Rev 3 is the authority; if this
file and Rev 3 disagree, Rev 3 wins. This file replaces an earlier report of the same name
(last updated 2026-09-03, still readable in git history at commit d82c41a). That earlier version
said the study "confirms the graph hypothesis" and that Phase 1 was complete. Neither statement
holds, and section 5 explains why.

The study is finished and will not be extended. NEXUS has moved on to a different Phase 1
(see the root `CLAUDE.md`). Four methods from this study carry forward; they are listed in
section 9.

## 1. The short version

We trained a neural network to imitate a physics simulator of electrically coupled cells. It
imitates the simulator well and runs about a thousand times faster. But the question the study
was built to answer, whether knowing how cells are wired together actually helps predict their
voltage, was never properly tested.

- **Accuracy against the simulator: works.** The model's average error is about 0.77 mV, ten
  times better than the target we set.
- **Speed: works.** About 22 ms per tissue on a CPU against about 41 s for the simulator.
- **Does the wiring matter? Not tested.** In the first dataset every cell in a tissue was
  identical, so the wiring had nothing to carry. The second dataset seemed to show the wiring
  mattering a lot, but the inputs we gave the model did not match what the simulator actually
  ran. That mismatch alone can produce the same win with no wiring effect at all.
- **Against real lab measurements: fails.** Average error 12.46 mV on six published
  experiments, against a 5.78 mV target. Most of the failure comes from biology the simulator
  setup could not represent, not from the model.

## 2. What we were trying to find out

NEXUS asks whether you can predict what a nervous system does from its molecular parts. That is
far too hard to attempt directly, so this study tried the smallest version with clean answers:
sheets of non-excitable cells (no spikes), where each cell's resting voltage is set by its ion
channels and by small pores, called gap junctions, that connect it to its neighbours.

The ground truth came from BETSE, an existing simulator of this kind of tissue. We ran it
thousands of times with different channel settings and asked a neural network to predict each
cell's voltage from two things:

1. how many of each ion channel each cell has, and
2. which cells are connected to which.

Two models were compared:

- **MPNN** (message-passing neural network). It sees both the channels and the connections.
  Each cell passes information to its neighbours over several rounds, which is a learned
  version of current flowing through gap junctions.
- **MLP** (a plain per-cell network). It sees only one cell's channels. It cannot see neighbours
  or connections at all.

If the MPNN clearly beats the MLP, and the reason is the connections, then the wiring carries
real information. That was the central claim (C2). There were four claims in total:

| # | Claim | Result |
|---|---|---|
| C1 | The model predicts voltage to within 10% of the voltage range | Met against the simulator. Not met against real measurements |
| C2 | Knowing the wiring is necessary | **Not tested** (sections 4 and 5) |
| C3 | The model handles kinds of change it never trained on | Met against the simulator, with caveats (section 7) |
| C4 | The model is much faster than the simulator | Met |

## 3. How the experiment was set up

- **Simulator.** BETSE 1.5.0. Each tissue is a regular honeycomb of 40 to 490 hexagonal cells.
  (The old report called it an irregular mesh; the code sets the irregularity to zero.)
- **What is predicted.** Each cell's voltage after 5 seconds of simulated time. This is not a
  true resting state: a slow pump-driven drift means the tissue never fully settles in this
  parameter range. The stored field is named `vmem_steady_state` for historical reasons only.
- **Inputs.** Eight channel numbers per cell. Two of them (two pump types) were zero in every
  record, so only six actually vary. The "Nav" input means the cell's resting sodium leak, not
  voltage-gated sodium channels, which are closed at rest.
- **Data.** 13,800 simulations per dataset, none failed. Split into 8,000 training, 1,000
  validation, 1,000 in-distribution test and 2,000 out-of-distribution test tissues.
- **Out-of-distribution tests.** Four kinds of change the model did not train on: blocking one
  channel, blocking gap junctions (in practice a 99% block, not a full one), a gradient of one
  channel across the tissue, and a quarter of cells overexpressing one channel.
- **Two datasets.**
  - **v1:** every cell in a tissue has the same channels.
  - **v2:** channel levels vary in a wave across each tissue, so cells differ from their
    neighbours.

## 4. Result: on uniform tissues, the wiring did not matter (v1)

On v1 both models are very accurate, and they are nearly tied:

| Test set | MLP (3 seeds) | MPNN (3 seeds) | Difference |
|---|---|---|---|
| Same kind of tissue as training | 0.797 ± 0.016 mV | 0.767 ± 0.012 mV | MPNN 0.031 mV better |
| New kinds of change | 1.190 ± 0.053 mV | 1.162 ± 0.057 mV | MPNN 0.029 mV better on average, but worse on one of the three seeds |

Numbers are mean ± standard deviation over 3 seeds. Both models explain 98% of the variation
(R² 0.980 to 0.981). The target was 10% of the simulator's voltage range, about 8 mV, so both
pass by a factor of ten.

**Why the tie happened.** If every cell in a tissue is identical, neighbouring cells sit at the
same voltage and no current flows between them. The connections then have nothing to do,
except at the tissue's edge. We measured this directly: within a tissue, voltage varies by
1.33 mV (standard deviation); between tissues it varies by 14.83 mV. About 99% of all the
variation is between tissues. A model that just predicts each tissue's average gets almost
everything right, and the per-cell MLP can do that. The dataset could not test the wiring.

This check (how much of the target varies within a graph, against between graphs) is the most
useful thing this study produced. A ratio near zero means the graph is not being tested,
whatever the model scores.

## 5. Result: the v2 "win" came from a broken input, not from the wiring

v2 was built to fix the v1 problem by making cells differ. In the logs, the MPNN then beat the
MLP by a wide margin: 1.070 mV against 3.008 mV. The old report read this as confirming that
the wiring matters. It does not, for three reasons.

**The inputs did not describe what was simulated.** The dataset stored a smooth wave of channel
levels for every cell. But the code that set up each simulation did something else:

- three channels (Kir, K_leak and the Na/K pump) were set once for the whole tissue, at the
  tissue average;
- the other three were applied in eight blocks of cells, sorted by a single channel.

So the model was shown per-cell detail that the simulator never used. Replaying the exact code
on 3,000 v2 setups, a median of 73.9% of the per-cell input variation was never simulated
(middle half of tissues: 62.3% to 86.8%).

**That mismatch alone produces the win.** Picture the task from the model's side. A cell's true
voltage depends mostly on tissue averages. Its input shows a wave around those averages. One
cell alone cannot work out the average, so the MLP fails. The MPNN can average over a few
neighbours, and more rounds of averaging get closer. That predicts a big MPNN win and "deeper is
better", with no gap-junction physics involved at all.

Rev 3 tested this with a stand-in simulator that copies the v2 setup but has **no coupling
between cells at all** (1,600 training and 400 test tissues, one seed per row):

| Labels | What the model was shown | Model | Error (mV) |
|---|---|---|---|
| No coupling | stored inputs, as in v2 | MLP | 3.67 |
| No coupling | stored inputs | MPNN | 1.88 |
| No coupling | stored inputs | MPNN with connections shuffled at random | 1.15 |
| No coupling | stored inputs | MLP plus one tissue-average number | 1.17 |
| No coupling | what was really simulated | MLP | 0.15 |
| With coupling | what was really simulated | MLP | 0.62 |
| With coupling | what was really simulated | MPNN | 0.35 |

With no coupling, the MPNN still beats the MLP by about 2x, close to the 2.8x seen in v2.
Shuffling the connections makes the MPNN better, not worse, and a plain MLP given a single
tissue average does just as well. So the thing being learned is a tissue average, not the
wiring. Only when the inputs match the simulation does the MPNN win for the right reason
(0.35 against 0.62 mV with real coupling).

This stand-in is not BETSE. It shows the mechanism and its rough size. It cannot say how much
of the real v2 margin it explains.

**The v2 runs were also noisy and uncommitted.** Three runs of the same v2 MPNN setup gave
1.070, 0.934 and 0.945 mV, a spread about six times larger than on v1 (standard deviation
0.076 against 0.013 mV). That is too much noise to support the old claims about the best
network depth or a "coupling length of 4 to 6 hops". None of the v2 outputs were committed to
the repository, so every v2 number exists only in the director's log.

**Where the numbers in this section come from.** The 1.070 and 3.008 mV results and the
replicate runs are log-only. The 73.9% figure and the stand-in table come from Rev 3, which
cites two scripts (`scripts/v2_realization.py` and `scripts/confound_experiment.py`). Those
scripts are not in the repository on any branch, so these numbers are also log-only until the
scripts are recovered.

**Verdict.** The v2 margin is real. Its cause is not shown to be the wiring. Claim C2 is
untested on both datasets.

## 6. Result: training runs do not repeat exactly

Six runs of the identical MPNN setup with the same seed gave 0.780 ± 0.013 mV in distribution
and 1.174 ± 0.032 mV out of distribution, stopping anywhere from epoch 48 to 91. Two MLP runs
matched to every printed digit. A short controlled run showed the MPNN drifting on both CPU and
GPU while the MLP stayed exact. The likely cause is the step that sums messages from
neighbours, whose addition order is not fixed when run in parallel.

This matters for reading small differences. The v1 in-distribution MPNN advantage at seed 42
(0.032 mV) is 2.5 run-to-run standard deviations; the out-of-distribution advantage (0.042 mV)
is only 1.3.

A second training issue: the learning-rate schedule always assumes 200 epochs, but early
stopping can end a run much sooner. A run that stops at epoch 35 never reaches the low learning
rates at the end of the schedule, so how well a run converges depends partly on when a noisy
validation curve happens to flatten.

## 7. Result: new kinds of change, speed and smaller findings

**New kinds of change (C3).** Against the simulator, error on the four held-out kinds of change
was 1.16 mV overall (v1, 3 seeds), ranging from about 0.53 to 3.30 mV across individual
combinations of change and channel (single runs). Two caveats:

- Two thirds of the "gradient" tests barely move the voltage (about 0.5 mV), so they are easier
  than ordinary test data and test nothing.
- The "overexpression" test feeds inputs at 4 when training only saw 0 to 1, so it mixes
  spatial generalization with plain extrapolation.

**Speed (C4).** On 100 tissues, run one at a time on the same machine: the simulator took a
median 41.3 s (middle half 38.6 to 48.8 s). The model took 22.3 ms on CPU and 7.4 ms on GPU,
including setup. That is 1,856x and 5,582x faster. The simulator time includes about 4 s of
start-up, and the simulator computes a whole time course while the model outputs one snapshot,
so the comparison favours the model somewhat. An earlier figure of 117.2 s per simulation was
measured under heavy parallel load and is withdrawn.

**Training-set size** was the only setting whose effect rose above run-to-run noise: error fell
from 0.886 to 0.817 to 0.787 to 0.767 mV as training tissues went from 1,000 to 8,000 (single
runs). Network depth, the physics-based extra loss and input scaling could not be told apart at
one run each.

**Edge versus interior cells.** Cells in the middle of a tissue had about three times the error
of cells at the edge, for both models (MLP 0.949 against 0.324 mV; MPNN 0.881 against
0.276 mV). The old explanation, irregular cell shapes, cannot be right, because all cells are
identical hexagons. This is unexplained.

## 8. Result: against real lab measurements, the model fails

We compared predictions with published experiments using a matched-baseline method: find up to
40 simulated tissues whose average voltage is within 3 mV of the measured control cells, apply
the same change (for example, block a channel), and compare the predicted voltage shift with
the measured one. Of 30 candidate records, 29 were independently confirmed; six could be mapped
onto the model's inputs. Results for the v1 MPNN (seed 42):

| Experiment | Change applied | Measured shift (mV) | Predicted shift (mV) | Right direction? |
|---|---|---|---|---|
| kcnh6 knockdown, frog embryo | remove K_leak | +20.0 | +22.2 | yes |
| barium, frog kidney | remove Kir | +13.0 | +4.5 | yes |
| barium, locust tubule | remove Kir | -18.0 | +16.7 | no |
| carbenoxolone 100 µM, stem-cell heart cells | halve gap junctions | +3.1 | 0.000 | no |
| carbenoxolone 200 µM, stem-cell heart cells | gap junctions to 20% | +7.5 | -0.012 | no |
| isolated vs sheet, stem-cell heart cells | remove gap junctions | +18.8 | +0.017 | yes, but only barely above zero |

Average error 12.46 mV against a 5.78 mV target: **fails**. On the two channel blocks the
setup can represent properly, the error is 5.35 mV. Three of six predictions point the right
way, not two as the old report said, though one of the three is a 0.017 mV prediction.

**The locust failure was predicted in advance.** That tissue is driven by a pump (V-ATPase) that
was switched off in every simulation, so the setup cannot represent it. It was included on
purpose as a case the model should fail.

**The gap-junction failures are probably the simulator setup, not the model.** The old report
said the model "learned that gap junctions do not matter" because they never mattered in
training. Two facts point elsewhere:

- Training already included tissues with gap junctions turned all the way down (conductance
  was drawn from 0 to 50 nS), so "no coupling" was not new to the model.
- The MLP, which never sees gap junctions, matched the MPNN to within 0.03 mV. If coupling moved
  tissue voltage by anything like 18.8 mV in the simulator, a model blind to it would be off by
  several millivolts. In a sheet of similar cells, currents between neighbours cancel out
  across the tissue, so the average voltage barely moves.

The real heart cells changed by 18.8 mV for reasons a single uniform cell population cannot
express. A direct simulator check would have settled this in under two hours; it was not run,
because the study was closed.

**Coverage was thin.** The validation sources named in the original plan mostly change pumps and
receptors this setup does not include. Values were taken only from text and tables, not from
figures. The carbenoxolone strengths (halve, cut to 20%) are placeholders, so only their sign
and order mean anything.

## 9. What carries forward

Phase 1 v2 does not reuse any code, data or models from this study. It keeps four methods:

1. **The within-graph against between-graph variance check** (section 4). Run it on every new
   dataset.
2. **A check that stored inputs match what the simulator received.** This study had no such
   check, which is how the v2 problem got through every test. Phase 1 v2 builds it as a frozen
   test that runs on every data generator.
3. **The matched-baseline comparison against real experiments** (section 8;
   `scripts/validate_experimental.py` in this archive).
4. **Writing and freezing the tests before writing the code.**

## 10. Corrections to the earlier version of this report

| Earlier version said | What is true | Source |
|---|---|---|
| v2 "confirms the graph hypothesis"; Phase 1 complete | The v2 margin is real but its cause is unproven; C2 is untested | Section 5 |
| The mesh is an irregular (Voronoi) tessellation | Regular hexagons; irregularity set to 0.0 | `nexus/data/betse_generator.py` line 66 |
| Kir and K_leak are set per group of cells | Set once per tissue from the tissue average | `betse_generator.py` lines 75 to 86, 113 to 128 |
| v2 gives every cell a distinct channel vector | True of the stored input, false of the simulation | Section 5 |
| 2 of 6 validation predictions had the right sign | 3 of 6 | `outputs/experimental_validation.json` |
| Gap-junction blindness persists because full uncoupling was outside training | Zero conductance was inside the training range | `nexus/data/config_sampler.py` line 27 |
| The coupling length is 4 to 6 hops | Not resolvable with one run per depth and v2's noise | Section 5 |
| Interior-cell error comes from irregular cell areas | Cells are identical hexagons; cause unknown | Section 7 |
| BETSE takes 117.2 s per simulation | 41.3 s median, measured one at a time | `outputs/speed_benchmark.json` |
| v2 headline MPNN "continues to epoch ~140"; MLP R² 0.9086 | Stopped at epoch 67; MLP R² 0.915 (both log-only) | Director log |
| v2 generation took about 18 h | 13,800 × 95 s ÷ 12 workers ≈ 30 h | Arithmetic |

Rev 3 section 11 has the full list, including sentences damaged by an earlier punctuation
clean-up.

## 11. How the code was built

Every line in `nexus/` and `scripts/` in this archive was written by a small local model
(qwen2.5-coder:7b, running through Ollama), directed by a Claude Opus process that wrote no
code. The 92-test suite was written before any implementation and treated as the definition of
done.

- **When the instructions gave exact function signatures, imports and line-level behaviour**,
  the small model wrote every file, including the three that combine several modules.
- **When instructions were looser**, it failed one file (the trainer) with five defects at once,
  then passed immediately when re-instructed in full detail. That is one case, so it is a lead,
  not a finding.
- **Its mistakes were local and mechanical**: dropped lines, code moved out of a loop, an
  invented convention replacing a stated one. None touched the message-passing maths.
- **Most correction rounds traced back to the director's own instructions**, not to the small
  model. The v2 input mismatch in section 5 is that kind of error: every test passed because no
  test compared the stored inputs with what BETSE received.

## 12. Open items at closure

Rev 3 listed nine follow-up items (R1 to R9). All are closed by decision, because the study is
not being extended. Two are kept on record without being fixed:

- **R4 (nothing from v2 was committed).** No v2 output, dataset, figure or evaluation file is in
  the repository, and neither dataset is published. No v2 number can be cited as anything but
  log-only.
- **R9 (inconsistent repository text).** The archived README, dataset cards and `DEVIATIONS.md`
  still contain the overstated claims and small errors listed in the archive README. They are
  left as they were, so the record shows what was originally written.
