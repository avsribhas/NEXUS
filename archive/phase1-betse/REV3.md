# NEXUS Phase 1: Technical Report

Revision 3, October 26, 2026. Supersedes logs/research-report.md (last updated 2026-09-03). Audited against repository avsribhas/NEXUS at commit d82c41a. Every number below is either read from a committed artifact, recomputed from one, or measured for this revision; the source is named where it matters. Numbers that exist only as prose in the director log are marked log-only.

## 1. Summary

The learned surrogate reproduces BETSE on spatially uniform tissues to about 0.77 mV and runs three to four orders of magnitude faster than the simulator. Those results, and the v1 finding that the graph network is statistically indistinguishable from a graph-blind baseline on uniform tissues, are backed by committed artifacts and hold.

The v2 result that the previous report presents as its central finding ("spatial heterogeneity confirms the graph hypothesis", MPNN 1.070 mV vs MLP 3.008 mV) does not support that conclusion as stated. Four problems, each established from the code or from measurement:

- The v2 inputs do not describe what BETSE simulated. The sampler writes continuous sinusoidal variation on six channels into each record, but the generator applies three of those channels (Kir, K_leak, NaKATP) once per tissue at the tissue mean, and the other three as eight quantile groups sorted on a single column. Replaying the exact code path on 3,000 v2 configurations: a median of 73.9% of the per-cell input variance the model sees was never simulated (IQR 62.3% to 86.8%).
- That mismatch alone favours any model that averages over neighbours. In a toy surrogate that reproduces the v2 input and realization pipeline but has zero gap-junction physics in its labels, the MPNN beats the per-cell MLP by 1.95x (1.88 vs 3.67 mV). An MPNN whose edges are rewired at random within each tissue does better than the real-mesh MPNN (1.15 mV), and so does a graph-blind MLP given one pooled tissue-mean feature (1.17 mV). The v2 margin is therefore not evidence that the network learned junctional coupling.
- v2 replicate spread is about six times the v1 spread. Three runs of the identical K=6 configuration gave 1.070, 0.934 and 0.945 mV (sd 0.076 mV vs 0.013 mV on v1). The K=4 to K=8 depth ordering and the "coupling length of 4 to 6 hops" inference do not survive that spread.
- No v2 run is committed. There is no v2 summary.json, evaluation JSON, ablation table, figure or validation output in the repository. Every v2 number is log-only.

The experimental gap-junction failure also needs a different reading. A conductance-blind MLP matched the MPNN to within 0.03 mV on v1 tissues whose gap-junction conductance was sampled uniformly from 0 to 50 nS. In the simulator, as parameterized, tissue-mean Vmem barely depends on coupling. The existing evidence points to the model reproducing BETSE here, not failing to learn it. The measured 18.8 mV depolarization in isolated cardiomyocytes is outside what a homogeneous-tissue BETSE configuration can express, and a direct BETSE run on the matched ensembles (section 10, R3) would settle it in under two hours of compute.

What remains defensible for a paper: the surrogate result on v1; the v1 null and its diagnosis (uniform sampling reduces a graph problem to a pointwise one); the nondeterminism finding; the OOD heterogeneity analysis; the speed benchmark; and the representability failures in experimental validation. The graph hypothesis is untested in a valid design on both datasets.

## 2. Question and claims

Phase 1 asks whether a learned model can predict per-cell resting membrane potential (Vmem) in a non-excitable tissue from per-cell ion-channel densities and gap-junction adjacency, using BETSE simulations as ground truth. It is the smallest version of the NEXUS question that has clean ground truth and no spiking dynamics.

| # | Claim | Metric | Status (this revision)  |
|---|---|---|---|
| C1 | Accuracy | MAE below 10% of Vmem range | Met against BETSE (v1: 0.767 ± 0.012 mV, 3 seeds). Not met against experiment (12.46 mV vs 5.78 mV threshold, n = 6) |
| C2 | Graph structure is necessary | MPNN beats density-only MLP beyond noise, for a reason attributable to coupling | Untested. v1: null out of distribution, 3.9% in distribution. v2: large margin, but confounded by the input/label mismatch (section 7) |
| C3 | Generalization to held-out perturbation families | per-family OOD MAE | Met against BETSE on v1 (1.16 mV aggregate, 0.53 to 3.30 mV per family). Two thirds of spatial_gradient does not test generalization (section 6.4) |
| C4 | Speedup | median latency, same machine | Met. 41.3 s BETSE vs 22.3 ms CPU / 7.4 ms GPU; 1,856x and 5,582x (v1 model, v1 tissues) |

## 3. Ground truth and the physics mapping

Simulator. BETSE 1.5.0, driven as a subprocess per simulation through its four stages (config, seed, init, sim). The extracellular solver is disabled. Meshes are regular hexagonal lattices: the generator sets lattice disorder to 0.0 (betse_generator.py, line 66). The previous report describes the mesh as a disordered Voronoi tessellation; the code does not produce one. Cell counts still differ from the request (40 to 490 for a requested 50 to 500) because boundary cells are culled.

Target. Vmem after 5 s of simulated equilibration. A Na+/K+-ATPase-driven concentration drift prevents any steady state over this parameter space, so the stored key vmem_steady_state is a compatibility name and must not be described as a steady state.

Mapping. Each density is converted to a fraction of its sampling maximum, clipped at 4.0:

| Input channel | BETSE parameter | Scope in the simulation  |
|---|---|---|
| Nav | Dm_Na = 1e-18 (1 + 3f) | per tissue profile |
| Ca | Dm_Ca = 1e-18 (0.1 + 1.9f) | per tissue profile |
| Cl | Dm_Cl = 1e-18 (0.1 + 1.9f) | per tissue profile |
| Kir | Kir2p1 max Dm = 3e-16 f | whole tissue, from the tissue mean |
| K_leak | K_Leak max Dm = 2e-17 f | whole tissue, from the tissue mean |
| NaKATP | alpha_NaK = 1e-7 (0.5 + f) | whole tissue, from the tissue mean |
| HKATP, VATP | none | held at 0 in every record |
| gap junctions | surface area 1e-9 + f (1e-7 − 1e-9) | whole tissue |

The scope column is read from betse_generator.py: the network channel objects and alpha_NaK are set once from normalize_channel_densities, which averages a per-cell array over cells, and the per-group profile loop writes only the four Dm_* constants. The previous report states that Kir and K_leak are per-profile properties; in this code they are not.

Three consequences carry through every result: the Nav feature is resting sodium permeability, not voltage-gated channel density (the gated Nav channel is closed at rest and is set to zero); two of eight inputs are constant, so the effective input dimension is six; and gj_blockade leaves 1% of open-junction surface area, so it is a near-blockade. The stored edge weight for a gj_blockade record is exactly 0 (gjopen × 0), while BETSE simulated that residual 1%.

The prior report's section 2.1 lists H+/K+-ATPase and V-ATPase among BETSE's pumps, while its section 2.3 says the configuration surface used exposes no equivalent. Whether those pumps can be enabled under a different ion profile was not checked. It matters: two of the PRD's named validation experiments (Beane et al. 2011, Adams and Levin 2012) perturb exactly these pumps.

## 4. Data

|  | v1 | v2  |
|---|---|---|
| Baseline tissues | spatially uniform (per-cell density sd = 0) | per-cell sinusoidal modulation along x on the six mapped channels |
| Simulations attempted / failed | 13,800 / 0 | 13,800 / 0 (log-only) |
| Splits | 8,000 train, 1,000 val, 1,000 test_id, 2,000 test_ood | same structure |
| Mean wall clock per simulation, 12 workers | 117.2 s | about 95 s (log-only) |
| Committed in repo | dataset card, figures, all run outputs | manifest and dataset card only |

v2 modulation, as implemented. One amplitude per tissue (15 to 50%), and an independent wavenumber (1 to 3 cycles) and phase per channel (config_sampler.py, lines 32 to 40). The v2 dataset card says the wavenumber is drawn per configuration; it is drawn per channel. The card also attributes the slower v2 simulations to heterogeneous tissues "converging more slowly"; every v2 simulation runs a fixed 5 s, and every v2 baseline now carries eight tissue profiles where v1 baselines carried none, which is the more likely cost given the per-profile setup overhead documented for v1.

Reproducibility of the datasets. Neither dataset is in the repository or linked from it. ConfigSampler now defaults to spatial_heterogeneity=True and generate_dataset.py has no flag for it, so v1 cannot be regenerated from the committed scripts without editing code. The heterogeneity block also consumes draws from the same random generator that later picks the perturbed channel, so v1 and v2 OOD records with the same ID target different channels; the two OOD splits are not a matched design.

OOD perturbation families. channel_blockade zeroes one mapped channel; gj_blockade sets conductance to 0; spatial_gradient writes a linear ramp on Nav, Ca or Cl; exogenous_expression sets a quarter of cells to 4x the channel maximum on Nav, Ca or Cl. Each family uses its own sampler seed, so OOD tissues are not paired with baseline tissues.

## 5. Models and training

MPNN (663,553 parameters at K = 6). Node encoder 8 → 64 → 128, edge encoder 1 → 32 → 64, K message-passing layers with mean aggregation into the destination node, residual update and LayerNorm, decoder 128 → 64 → 1, fixed output map ×30 − 50 mV.

Baseline MLP (26,625 parameters). Per-cell 8 → 128 → 128 → 64 → 1 with the same output map. It sees neither edges nor conductances.

Training. AdamW, learning rate 1e-3 with cosine annealing to 1e-5 over 200 epochs, weight decay 1e-4, batch 32 graphs, gradient clipping at 1.0, MAE loss in mV, early stopping with patience 20 on validation MAE and restoration of the best state. One interaction matters for every comparison in this report: the cosine schedule spans 200 epochs regardless of when early stopping fires, so a run that stops at epoch 35 never reaches a low learning rate while a run that reaches epoch 200 completes the schedule. Convergence quality is tied to when a noisy validation curve happens to plateau.

Metric weighting. All MAEs are per cell over a concatenated split, so a 490-cell tissue counts 12 times as much as a 40-cell tissue.

## 6. Results on v1 (artifact-backed)

### 6.1 Accuracy and the seed study

Recomputed from outputs/{mlp,mpnn}_seed{42,137,256}/summary.json:

| Split | MLP (3 seeds) | MPNN (3 seeds) | Difference  |
|---|---|---|---|
| test_id | 0.797 ± 0.016 mV | 0.767 ± 0.012 mV | −0.031 mV (−3.8%) |
| test_ood | 1.190 ± 0.053 mV | 1.162 ± 0.057 mV | −0.029 mV; per-seed sign flips (−0.052, +0.088, −0.121) |

Both clear the roughly 8 mV threshold (10% of the BETSE Vmem range) by a factor of ten. R² on test_id is 0.980 to 0.981 for both architectures.

### 6.2 Nondeterminism and the noise floor

Six runs of the identical MPNN configuration and seed give test_id 0.780 ± 0.013 mV and test_ood 1.174 ± 0.032 mV, with early stopping between epochs 48 and 91. Two MLP runs of the same configuration agree to every printed digit. A controlled 10-epoch run shows the MPNN diverging on CPU and on CUDA while the MLP reproduces exactly, which points at index_add aggregation, whose reduction order is not fixed under parallel execution. Against this floor, the in-distribution MPNN advantage at seed 42 (0.032 mV) is 2.5 per-run standard deviations; the out-of-distribution advantage (0.042 mV) is 1.3.

### 6.3 Why the graph did not matter on v1

Within-tissue Vmem sd is 1.33 mV in train against 14.83 mV across tissues: about 99% of the target variance is between tissues. With identical cells, the junctional term vanishes in the bulk for any conductance, and coupling can act only at the rim. A per-cell model that predicts the tissue mean captures almost everything. This diagnostic, the ratio of within-graph to between-graph target variance, is the most transferable methodological result in the project.

### 6.4 OOD difficulty by family and perturbed channel

From outputs/eval_{mlp,mpnn}42.json (single runs):

| Family / channel | MLP MAE | MPNN MAE  |
|---|---|---|
| exogenous_expression / Nav | 3.692 | 3.304 |
| channel_blockade / K_leak | 1.377 | 1.212 |
| channel_blockade / Kir | 1.324 | 1.133 |
| exogenous_expression / Cl | 1.416 | 1.270 |
| channel_blockade / Nav | 1.321 | 1.302 |
| gj_blockade | 1.115 | 1.000 |
| exogenous_expression / Ca | 0.987 | 1.463 |
| spatial_gradient / Nav | 1.002 | 0.971 |
| channel_blockade / NaKATP | 0.892 | 0.857 |
| channel_blockade / Ca | 0.892 | 0.826 |
| channel_blockade / Cl | 0.710 | 0.667 |
| spatial_gradient / Cl | 0.577 | 0.590 |
| spatial_gradient / Ca | 0.532 | 0.572 |

Difficulty spans 6.5x. Cl and Ca gradients move Vmem by about 0.5 mV, so those two thirds of spatial_gradient are easier than in-distribution data and test nothing. exogenous_expression puts inputs at 4.0 in normalized units when training saw only [0, 1], so it mixes spatial and input-range extrapolation. The −48% MPNN regression on exogenous Ca is one run and remains unexplained.

### 6.5 Boundary versus interior cells

Interior cells (degree 6) carry about three times the error of boundary cells for both models, paired within tissue (MLP 0.949 vs 0.324 mV; MPNN 0.881 vs 0.276 mV). The MLP cannot see degree and is constrained to one output per uniform tissue, so the asymmetry is a property of the data. The previous report's leading explanation was cell-area variation across a disordered Voronoi mesh. With lattice disorder at 0.0 the interior cells are identical hexagons, which makes that explanation unlikely; the asymmetry is unexplained.

### 6.6 Speed

100 configurations, serial and unloaded (outputs/speed_benchmark.json): BETSE median 41.3 s (IQR 38.6 to 48.8 s), model median 22.3 ms on CPU and 7.4 ms on GPU, including graph construction and host-to-device transfer. BETSE's figure includes about 4 s of process startup, and BETSE computes a full time course where the model emits one field at one instant. The earlier 117.2 s denominator was a throughput figure under 12-way load and is withdrawn.

### 6.7 v1 ablations

Single runs at seed 42 (outputs/ablations.json). Only training-set size resolves above the noise floor: 0.886 → 0.817 → 0.787 → 0.767 mV from 1K to 8K tissues. Depth (K = 2, 4, 6, 8), the physics auxiliary loss and input normalization are within two per-run standard deviations of each other and are uninterpretable at n = 1.

## 7. v2: what was claimed and what the evidence supports

### 7.1 The claim

From the director log (log-only): MPNN test_id 1.070 mV (R² 0.983, 67 epochs) against MLP 3.008 mV (R² 0.915, 35 epochs), test_ood 1.816 vs 3.478 mV, and an 11-run ablation in which depth falls monotonically from 1.417 mV at K = 2 to 0.824 mV at K = 8. The previous report reads the 2.8x margin as confirmation that gap-junction topology carries signal a per-cell model cannot access.

### 7.2 What the generator actually simulated

Replaying ConfigSampler(seed=1000) (the v2 campaign seed; script v2_realization.py, delivered with this report) and group_cells_by_density for 3,000 configurations, then applying the realization rules from section 3:

| Quantity | Value  |
|---|---|
| Per-cell input variance (six mapped channels, normalized) never simulated, median | 73.9% (IQR 62.3% to 86.8%) |
| Tissues whose grouping column is Kir, K_leak or NaKATP | 35.3% |
| Grouping column frequency | Nav 63.8%, Kir 16.4%, NaKATP 16.3%, K_leak 2.7%, Cl 0.9% |
| corr(requested Nav, simulated Nav) per tissue, median (IQR) | 0.99 (0.66 to 0.99) |
| corr(requested Ca or Cl, simulated), median (IQR) | 0.46 (0.12 to 0.77) |

The grouping column is the one with the largest raw-unit spread. Because one amplitude is shared and Nav has the largest maximum, Nav usually wins; in about a third of tissues a channel that BETSE applies tissue-wide wins instead, and the groups are then sorted on a variable that has no per-cell effect. Each record still stores the requested per-cell values as its input.

The consequence for the comparison: a cell's simulated Vmem depends on tissue-mean Kir, K_leak and NaKATP and on group means of the other channels, while its input shows a sinusoid around those means with unknown phase and amplitude. A per-cell MLP cannot recover the tissue mean from one cell. A message-passing network can estimate it by averaging over a few hops, and a deeper network estimates it better. That predicts both a large MPNN margin and a monotonic depth curve with no junctional physics involved.

### 7.3 A control with zero coupling physics

To check whether the mismatch alone produces a margin of the observed size, a toy surrogate was built for this revision (script confound_experiment.py, delivered with this report). It reproduces the v2 sampler (shared amplitude, per-channel wavenumber and phase, x-ordered layout), the generator's realization rules, and hexagonal meshes of 50 to 300 cells. Labels come from a parallel-conductance resting-potential model in place of BETSE. Two label sets: pointwise, where each cell's voltage depends only on its own realized parameters and gap junctions do nothing, and coupled, where the same membranes are joined by a gap-junction Laplacian whose strength is sampled per tissue. 1,600 training and 400 test tissues, 30 epochs of cosine-annealed AdamW with no early stopping, mean aggregation with residual LayerNorm as in the repo; one seed per condition.

| Labels | Model input | Model | Test MAE (mV)  |
|---|---|---|---|
| Pointwise (no coupling) | requested, as the repo stores them | MLP | 3.665 |
| Pointwise (no coupling) | requested | MLP + pooled tissue mean | 1.173 |
| Pointwise (no coupling) | requested | MPNN, K = 6 | 1.881 |
| Pointwise (no coupling) | requested | MPNN, edges rewired within tissue | 1.149 |
| Pointwise (no coupling) | realized | MLP | 0.146 |
| Coupled | realized | MLP | 0.623 |
| Coupled | realized | MPNN, K = 6 | 0.346 |

With labels that contain no coupling at all, the MPNN still beats the per-cell MLP by 1.95x, the same order as the 2.81x reported on v2. Rewiring the edges at random within each tissue, keeping each cell's in-degree, makes the MPNN better, not worse (1.15 vs 1.88 mV): random senders reach the tissue mean in fewer hops than mesh neighbours do. A graph-blind MLP given one pooled tissue mean of its inputs matches the rewired network (1.17 mV). The quantity being learned is a tissue-level average, and the mesh topology is an obstacle to it. Giving the plain MLP the realized parameters instead of the requested ones removes nearly all of its error (0.15 mV). Only with coupled labels and realized inputs does the comparison measure coupling, and there the MPNN beats the MLP by 1.8x (0.35 vs 0.62 mV). A graph advantage is recoverable once the inputs describe the simulation.

This is a stand-in for BETSE, not BETSE: its voltages, ranges and nonlinearity differ. It establishes the mechanism and its scale, not the share of the real 2.8x margin it explains. That share needs the controls in R2 run on the real v2 data.

### 7.4 Replicate spread on v2

The v2 ablation grid repeated the headline configuration twice more (abl_depth_k6, abl_size_8000):

| Run | test_id MAE | test_ood MAE | Epochs  |
|---|---|---|---|
| headline MPNN | 1.070 | 1.816 | 67 |
| abl_depth_k6 | 0.934 | 1.868 | 200 |
| abl_size_8000 | 0.945 | 1.713 | 142 |
| mean ± sd | 0.983 ± 0.076 | 1.799 ± 0.079 |  |

The test_id spread is six times the v1 per-run sd. The MPNN-MLP gap (about 2 mV) is far larger than this spread, so the margin itself is not noise; its cause is the open question (section 7.3). The depth sweep is a different matter. K = 4 (1.071) sits inside the range of three K = 6 runs (0.934 to 1.070), and K = 8 (0.824) is 2.1 sd below the K = 6 mean. Only "K = 2 is worse than K ≥ 4" survives. Four of the depth and loss runs hit the 200-epoch cap while the rest stopped between epochs 35 and 196, so depth is also confounded with how far each run got through the learning-rate schedule. The MLP stopped at epoch 35, with the learning rate still at about 90% of its starting value.

### 7.5 Verdict on v2

The margin is real; its attribution to gap-junction coupling is not established. Any statement that v2 "confirms the graph hypothesis", that "the coupling length is 4 to 6 hops", or that the physics loss "over-regularizes OOD extrapolation" should be withdrawn until R1 and R2 are done.

## 8. Experimental validation

### 8.1 Protocol and results

Differential, matched-baseline protocol: select up to 40 training tissues whose BETSE mean Vmem lies within ±3 mV of a measured control potential, apply the published perturbation to each, and compare the median predicted shift with the measured shift. Thirty candidate records were curated with independent verification (29 confirmed, 0 fabricated); six map onto the model's channels. From outputs/experimental_validation.json (v1 MPNN, seed 42):

| Record | Operation | Measured ΔV | Predicted ΔV, median [IQR] | Error | Sign  |
|---|---|---|---|---|---|
| kcnh6 morpholino, Xenopus | zero K_leak | +20.0 | +22.2 [+10.9, +30.9] | 2.2 | correct |
| Ba²⁺, frog kidney | zero Kir | +13.0 | +4.5 [+2.7, +12.7] | 8.5 | correct |
| Ba²⁺, locust Malpighian tubule | zero Kir | −18.0 | +16.7 [+12.6, +23.7] | 34.7 | wrong |
| carbenoxolone 100 µM, hiPSC-CM | conductance ×0.5 | +3.1 | +0.000 | 3.1 | wrong |
| carbenoxolone 200 µM, hiPSC-CM | conductance ×0.2 | +7.5 | −0.012 | 7.5 | wrong |
| isolated vs monolayer, hiPSC-CM | conductance ×0 | +18.8 | +0.017 | 18.8 | correct (trivially) |

MAE 12.46 mV against a 5.78 mV threshold (10% of the 57.8 mV experimental range); 5.35 mV on the two representable channel blockades. The artifact records 3 of 6 sign-correct; the previous report's "2/6" is wrong, though the third is a 0.017 mV prediction that is correct in sign only. The locust record was curated in advance as a case the model must fail, because Malpighian tubules are driven by a V-ATPase held at zero, and it failed as written down.

The v2 validation ("both v2 models 2/6 sign-correct, gap-junction predictions still about zero") is log-only.

### 8.2 The gap-junction result, re-read

The previous report concludes that the model "learned that gap junctions do not affect Vmem" because uniform training tissues gave conductance no effect, and that v2 failed to fix this because complete uncoupling lies outside the training range. Both parts have problems.

- Conductance was sampled uniformly from 0 to 50 nS in training, so an edge weight of zero is inside the training range, not outside it.
- The simulator itself shows little dependence of Vmem on coupling. The MLP never sees conductance, yet it matches the MPNN to 0.03 mV on v1 test_id (0.797 vs 0.767 mV) and reaches 1.12 mV on near-uncoupled gj_blockade tissues. If coupling shifted tissue Vmem by anything near the 18.8 mV measured in cardiomyocytes, a model blind to it would carry errors of several millivolts. In a passive sheet of similar cells, junctional currents sum to zero across the tissue, so the mean potential moves only to the extent that cells differ.
- The experimental depolarization comes from biology the configuration cannot express. Isolated and monolayer hiPSC cardiomyocytes differ in electrotonic loading and in their own channel expression; a single homogeneous cell population cannot reproduce that.

So the gap-junction result is most likely a representability limit of the BETSE configuration, the same class as the locust record, rather than a learning failure. R3 tests this directly.

### 8.3 Coverage

The PRD's named validation sources (Beane et al. 2011, Pai et al. 2018, Adams and Levin 2012) perturb targets the parameterization does not contain: H+/K+-ATPase, nicotinic receptors, HCN2 and V-ATPase. Values were taken only from text and tables, so most published bioelectric data (figures and calibrated colour maps) was not used. Carbenoxolone scale factors (×0.5, ×0.2) are placeholders; only their sign and order mean anything. Morpholino knockdown is modelled as complete loss, so the close kcnh6 agreement may be partly fortuitous. Total error against biology includes BETSE's own error, whose published validation is one preparation.

## 9. Limitations

- The target is Vmem after 5 s, not a steady state.
- Two of eight inputs are constant; Nav means resting sodium permeability.
- v2 inputs do not match the simulated parameters (section 7.2). This invalidates v2 as a test of coupling until fixed.
- v2 results are single runs with no committed artifacts.
- Kir, K_leak and NaKATP cannot vary within a tissue in this generator, and spatial perturbations use only Nav, Ca and Cl.
- gj_blockade is a 1% near-blockade, and its stored edge weight (0) does not match the simulated residual.
- MPNN training is nondeterministic on CPU and GPU; v1 per-run sd is 0.013 mV and v2 per-run sd is about 0.076 mV.
- Early stopping and a fixed 200-epoch cosine schedule interact, so runs that stop early are under-annealed.
- Two thirds of spatial_gradient does not test generalization; exogenous_expression confounds spatial and input-range extrapolation.
- Experimental validation covers six records, two of them representable channel blockades.
- Per-cell weighting lets large tissues dominate every reported MAE.
- Meshes are regular hexagonal lattices, so results say nothing about irregular tissue geometry.

## 10. Required work

Ordered by what each item unblocks. Cost estimates assume the RTX 4050 rig and serial BETSE at about 42 s per simulation.

| # | Work | Specification | Decides | Cost  |
|---|---|---|---|---|
| R1 | Make v2 inputs match the simulation | Bin cells into profiles by x-position quantile instead of by one density column, so every Dm channel's sinusoid is realized per bin; modulate only Nav, Ca and Cl unless BETSE can set channel max Dm per profile (test this first with two profiles and a Kir difference); write the realized per-cell values, not the requested ones, as channel_densities; regenerate | Whether any v2-style dataset can test C2 | about 30 h generation |
| R2 | Controlled C2 test | On the R1 dataset: MLP; MLP with pooled tissue-mean inputs; MPNN; MPNN with edges rewired at random within each tissue, keeping in-degree; MPNN with conductance zeroed. Three seeds each, fixed 200 epochs, best-validation checkpoint, no early stopping | C2. It holds only if the real-graph MPNN beats both the pooled MLP and the rewired MPNN by more than the replicate sd | about 15 GPU-hours |
| R3 | BETSE-direct gap-junction control | Re-simulate the 40-member matched ensembles for the three gap-junction records with junction surface area scaled ×0.5, ×0.2 and at the floor; compare BETSE's own mean ΔV with the model's and the measured values | Whether the gap-junction failure belongs to the model or the simulator configuration | about 1.5 h |
| R4 | Commit and publish | All v2 run outputs, evaluation JSONs, ablation table, figures and validation output; both datasets on Zenodo or Hugging Face with checksums, linked from the README; a --spatial-heterogeneity flag in generate_dataset.py so each dataset card names the exact command | Whether any v2 number can be cited | under 1 day |
| R5 | Determinism | torch.use_deterministic_algorithms(True) with CUBLAS_WORKSPACE_CONFIG=:4096:8, or a sorted segment sum; report every result as mean ± sd over at least three runs | Whether sub-0.1 mV differences can be read | under 1 day, slower training |
| R6 | v2 OOD breakdown | The v1 family × channel table for the R1/R2 models | C3 on heterogeneous tissues | minutes |
| R7 | Validation coverage | Digitize figure data from the curated sources; check whether BETSE can enable H+/K+-ATPase and V-ATPase under another ion profile; add a two-population tissue (for example myocyte plus non-excitable cell) for coupling experiments | C1 against experiment with more than two representable records | several days |
| R8 | Interior-cell error | Regress within-tissue Vmem deviation on distance to the boundary and on tissue size, now that cell geometry is known to be uniform | The mechanism in section 6.5 | under 1 h |
| R9 | Repository consistency | Update the README (it still reports only v1); keep one copy each of DEVIATIONS.md and DATASET_CARD.md (the root and logs/ copies have diverged, and the root DEVIATIONS.md reads "complete in 46 simulated seconds" where "4 to 6" lost its dash in the ASCII pass); fix the dataset card's v2 MPNN test_ood (1.868, which is the abl_depth_k6 run, against 1.816 for the headline) | Whether a reader can trust the repository text | under 1 h |

R1 and R2 are the only route to C2. R3 is the cheapest experiment with the largest effect on how the project's main negative result is described.

## 11. Corrections to the previous report

| Location | Previous text | Correct | Source  |
|---|---|---|---|
| §1.2, §10.9, §24 | v2 "confirms the graph hypothesis" | Margin real, attribution confounded | §7 here |
| §3.3 | mesh from "a disordered lattice" | lattice disorder = 0.0, regular hexagons | betse_generator.py L66 |
| §5.2 | Kir, K_leak max Dm is a per-profile property | Set once for the whole tissue from the tissue mean | betse_generator.py L75 to 86, L113 to 128 |
| §10.9 | v2 modulation "applied to all six mapped channels, so every cell sees a distinct channel-density vector" | True of the stored input; false of the simulation | §7.2 here |
| §10.9 | MPNN "continues to epoch ~140" | Headline run stopped at epoch 67 | director log |
| §10.9 | MLP R² 0.9086 | 0.915 | director log |
| §10.9, §14, §24 | 2/6 sign-correct | 3/6 | experimental_validation.json |
| §10.9 | GJ blindness persists because complete uncoupling is outside the training distribution | Zero conductance is inside the sampled range | config_sampler.py L27 |
| §10.9 | "coupling length 4 to 6 hops" | Not resolvable at n = 1 per depth with v2 replicate spread | §7.4 here |
| §7.1 | depth sweep "not yet tested" | Run on v1 and v2 | stale text |
| §13 | Experiment G "unscoped" | Done as Milestone 7 (§10.8) | stale text |
| §2.2, §10.5 table, others | sentences broken by the punctuation pass ("The reasons are pragmatic and BETSE's internal state…", "gj_blockade /;") | Restored here | commit 8761569 |
| DATASET_CARD v2 | wavenumber drawn per configuration | Per channel, with per-channel phase | config_sampler.py L36 to 37 |
| DATASET_CARD v2 | v2 slower because tissues converge more slowly | Fixed 5 s runs; eight profiles per baseline is the likely cost | §4 here |
| commit d8b7fe8 | v2 generation took about 18 h | 13,800 × 95 s ÷ 12 workers ≈ 30 h | arithmetic |

## 12. The director-executor build (Part II, condensed)

Every line in nexus/ and scripts/ was written by qwen2.5-coder:7b through Ollama, directed by a Claude Opus process that wrote no code. The 92-test suite (87 without BETSE) was written before any implementation and treated as the objective function.

Result. At full scaffold (exact signatures, imports and line-level behaviour) the 7B produced every file, including the three files that compose several modules (mpnn.py, dataset.py, trainer.py). At partial scaffold it failed trainer.py with five simultaneous defects; re-issued at full scaffold it passed at once. The useful question about a small executor is the minimum specification density at which it can write a given file.

Failure taxonomy. The executor's errors were local and syntactic: lines deleted next to an insertion, one line dropped from an embedded block, an over-literal type inference, a destructive list operation, a type error in arithmetic, test code inside a source module, a stated convention replaced by an invented one, a block moved out of its loop. None involved the message-passing mathematics. Files without test coverage needed the most correction rounds.

Director errors. By the log's own count most correction rounds traced to the director's specifications (a forward-slash path BETSE rejects on Windows, a mapping onto an inert parameter, a convergence number measured in the wrong stage, a 36-hour campaign launched without profiling). The v2 realization gap in section 7.2 belongs to the same class. Each record stores the requested densities rather than the simulated ones, and the code passed every test because no test compares a record's stored densities with the parameters BETSE received.

Threats to validity. The partial-scaffold failure is one task (n = 1). There is no control run without the six protocol properties. Director-error counts are retrospective and self-reported. Fixed signatures make the task closer to transcription than to open-ended software design.

## 13. Changelog

2026-10-26, revision 3. Full audit of code, artifacts and git history against the report. Added the v2 realization measurement (§7.2), the zero-coupling control (§7.3) and the v2 replicate analysis (§7.4); re-read the gap-junction validation result (§8.2); corrected the sign-correct count, the mesh description and the scope of the Kir, K_leak and NaKATP parameters; marked every v2 number log-only; replaced the planned-experiments table with R1 to R9; condensed Part II.

2026-09-03. v2 dataset generated and trained; report declared Phase 1 complete.

2026-08-30. First training on BETSE data; v1 null; nondeterminism measured; speed benchmark re-run to protocol; experimental validation.

Earlier. BETSE integration, physics mapping by parameter sweep, 13,800-simulation campaign, steady-state criterion found unsatisfiable.
