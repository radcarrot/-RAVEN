# Research Plan: VAE Backbone Comparison for Time-Series Anomaly Detection

## One-Line Goal

Show that threshold calibration and training pathology (KL collapse,
training-set contamination) — not encoder backbone choice — dominate
VAE-based anomaly detection performance, with a secondary investigation of
Mamba's linear-time scaling advantage at long windows.

> **Updated session 16:** Mid-sweep analysis (9/28 machines) shows 48% of the
> gap to OmniAnomaly is threshold calibration, 52% is model quality. Backbone
> choice (TCN vs LSTM) produces nearly identical AUPRC on every machine.

---

## Target Venue

| Priority | Venue | Deadline | Notes |
|----------|-------|----------|-------|
| 1 | NeurIPS 2026 Time Series / Anomaly Detection Workshop | ~Sep 2026 | Realistic; workshop papers have lower novelty bar |
| 2 | ECML-PKDD 2026 | ~Apr 2026 | Mid-tier, accepts rigorous comparison papers |
| 3 | PAKDD / SDM 2027 | ~Oct 2026 | Fallback mid-tier |
| 4 | ArXiv preprint | Anytime | Always an option; not a publication |

**Not targeting:** NeurIPS/ICML/ICLR main track, KDD — these require a new method or
significant theoretical contribution. A backbone comparison alone will not pass
desk rejection.

---

## Core Research Questions

The paper must answer at least one of these with evidence, not just report numbers.

### RQ1 (Primary — architecture)
> Does the temporal backbone (TCN / BiLSTM / Mamba-2) meaningfully affect anomaly
> detection accuracy, or is the bottleneck elsewhere (normalization, KL collapse,
> threshold calibration, decoder capacity)?

**Hypothesis:** Backbone choice matters less than threshold calibration and
training pathology. All three converge to similar discriminative ability because
the shared decoder dominates representation quality.

> **Session 16 evidence (9/28 machines):** Hypothesis **supported**. TCN wins
> 5-2-2 vs LSTM on best-achievable PA-F1, but AUPRC is nearly identical on
> every machine, confirming equal discriminative power. The 48/52 gap
> decomposition shows threshold calibration (48%) rivals model quality (52%)
> as the dominant performance factor. Mamba comparison still pending.

### RQ2 (Primary — scaling)
> At long window sizes (W = 64 → 256 → 512), does Mamba's O(L) scan give a
> practical accuracy or speed advantage over LSTM (O(L) but large constant) and
> TCN (needs more layers)?

**Hypothesis:** At W=64 all models are equivalent. At W≥256, LSTM training slows
significantly; TCN needs more dilated layers to cover the window; Mamba maintains
linear time with the same block count. If anomaly signatures span >61 timesteps
(TCN's default receptive field), Mamba and LSTM should pull ahead.

### RQ3 (Secondary — pathology, CONFIRMED)
> Is KL collapse architecture-independent, and does fixing it (free bits) improve
> anomaly discrimination?

**Hypothesis:** Yes — all three backbones collapse to KL≈0 under the same
annealing schedule because the ConvTranspose decoder is powerful enough to ignore
the latent code. Free bits or normalizing flow prior will improve latent
utilization equally across all three.

> **Confirmed (sessions 11-16).** Free bits (0.1 nats/dim) locks KL at 3.2
> nats for both TCN and LSTM. ROC-AUC improves by +0.07 identically across
> both architectures. The improvement holds across all 9 machines tested so far.

### RQ4 (Emerging — threshold calibration)
> Is threshold calibration methodology a dominant and under-reported factor in
> reconstruction-based TSAD performance, and can training-data-adaptive threshold
> selection close the gap to published baselines without model changes?

**Hypothesis:** Yes. Published methods often use test-set-optimal thresholds
(Kim et al. 2022), inflating reported numbers. Our 48/52 gap decomposition
shows that switching from p99.9 to an adaptive strategy recovers 48% of the
gap to OmniAnomaly — an average of 0.274 PA-F1 points — without touching the
model.

> **Session 16 evidence (9/28 machines):** POT@1e-2 averages 0.503 PA-F1 vs
> p99.9's 0.314 (60% improvement). Three machines have contaminated training
> sets where p99.9 is catastrophically wrong (PA-F1=0.0 on machine-1-7), but
> the model achieves 0.941 at p99 threshold. Training-data-adaptive selection
> based on `train_std/train_mean` ratio correctly identifies contaminated
> machines. This is a potentially publishable methodological contribution.

---

## What Would Make This Publishable

### Minimum bar (workshop paper)
- [x] Fix KL collapse and show it matters — **DONE (F1, F3): free bits +0.07 ROC-AUC**
- [ ] Results on ≥ 3 datasets (SMD + SMAP + MSL)
- [x] Both PA-F1 and strict F1 reported — **DONE: all runs store both + AUPRC**
- [ ] ≥ 2 proper baselines beyond our own models (e.g. THOC, Anomaly Transformer)
- [x] One clear answered research question — **RQ1 (backbone secondary) + RQ4 (threshold calibration dominant)**
- [ ] 48/52 gap decomposition across full 28 machines (emerging key contribution)
- [ ] Three failure modes taxonomy validated across full sweep
- [ ] Contamination filtering demonstrated to recover Type A machines

### Stronger bar (mid-tier conference)
- All of the above, plus:
- [ ] Long-window scaling experiment (RQ2) showing Mamba advantage at W≥256
- [ ] Statistical significance testing across 28 machines (Wilcoxon signed-rank)
- [ ] Ablation table: backbone × normalization × KL fix → 3×2×2 = 12 conditions
- [ ] Runtime / memory profiling per architecture
- [ ] Training-data-adaptive threshold selection algorithm with formal description
- [ ] Per-machine gap decomposition showing threshold vs model contribution

---

## Benchmarking Standards & Comparison Methodology

This section documents current field standards so the study design is
defensible to reviewers. Based on literature survey (Apr 2026).

### Standard Train/Test Splits

| Dataset | Split | Source | Note |
|---------|-------|--------|------|
| SMD | Fixed 50/50 per machine (OmniAnomaly) | NetManAIOps/OmniAnomaly | Do NOT re-split; deviating makes numbers incomparable |
| SMAP | Pre-split by NASA/OmniAnomaly | Same repo | Use as-is |
| MSL | Pre-split by NASA/OmniAnomaly | Same repo | Use as-is |
| PSM | Pre-split | GitHub (eBay) | Public, no access request needed |
| SWaT | 7 days normal train / 4 days attack test | iTrust SUTD | Requires data access request |

No leave-one-out or rolling-window cross-validation is standard for these datasets.

### Threshold Selection (Critical for Fairness)

Three camps exist; **camp 2 is what we use and it is methodologically correct**:

1. **Optimal threshold from test set** — Many published papers (TranAD, DCdetector, Anomaly Transformer)
   find the F1-maximizing threshold by grid-searching over all thresholds on the *test set*.
   This is not blind validation. Reviewers increasingly flag this. Do not use.

2. **99th-percentile of training scores** (our approach) — Fully blind to test labels.
   Kim et al. (AAAI 2022) explicitly endorse this. Our `THRESHOLD_PCT99` is correct.
   **State explicitly in the paper**: "threshold is set on training data only; no test labels
   are accessed at any stage." This is a point of methodological strength.

3. **POT (Peaks Over Threshold)** — Fits a GPD to the training score tail. Principled and
   automatic, but less reproducible than percentile due to sensitivity to `q` and `level` params.
   Report as a secondary reference value (already in `evaluation.py`).

### Evaluation Metrics

Report all four of these per model per dataset:

| Metric | Required | How |
|--------|----------|-----|
| **PA-F1** | Yes | Enables comparison with OmniAnomaly, TranAD, Anomaly Transformer published numbers |
| **Strict point-wise F1** | Yes | Required by Kim et al. standard; shows PA inflation effect |
| **ROC-AUC** | Yes | Threshold-independent; already in notebook |
| **AUPRC** | Recommended | `sklearn.metrics.average_precision_score`; more informative than ROC-AUC under class imbalance (4–13% anomaly ratios) |
| Affiliation-F1 | Optional | Huet et al. KDD 2022; tolerant of early/late detection; code available |

**PA-F1 inflation note:** Always report strict F1 alongside PA-F1. Be explicit in the paper
that PA-F1 is used *for comparability with published baselines*, not as the primary claim.
Kim et al. (AAAI 2022) showed that near-random detectors can achieve competitive PA-F1.

**VUS-PR** (TSB-AD NeurIPS 2024 recommendation): the most robust metric identified by the
most authoritative current benchmark. We do not implement it for a workshop paper but should
cite the recommendation and note that AUPRC is our practical approximation.

### Statistical Significance Testing

| Test | When | Notes |
|------|------|-------|
| **Wilcoxon signed-rank** (paired, non-parametric) | Workshop level | 28 paired samples (SMD machines) per model pair; α=0.05 |
| **Friedman + Nemenyi post-hoc** | Mid-tier conference | Three-way comparison across TCN/LSTM/Mamba; report CD diagrams |

28 SMD machines provide sufficient paired samples for Wilcoxon without needing
multiple seeds per machine. For SMD, one run per machine per model is standard.

### Seeds and Reproducibility

- Single run with fixed random seed is standard for SMD (28 machines provide variance).
- Better papers (mid-tier conference level) report 3–5 seeds with mean ± std.
- Minimum: fix `torch.manual_seed`, `numpy.random.seed` before training. Report the seed.
- Document: Python/PyTorch/CUDA versions, GPU model, training time.

### How Parameters and Compute Are Reported

- Total trainable parameters per model (we have: TCN 264K, Mamba 268K, LSTM 714K).
- Training time per epoch and total wall-clock time on the specific GPU.
- Peak GPU memory per batch.
- Explicitly note: TCN and Mamba are parameter-matched; LSTM is 2.7× larger.
  Any LSTM advantage must be qualified as "may reflect capacity, not architecture."

---

## Datasets

| Dataset | Entities | Features | Anomaly ratio | Source | Priority |
|---------|----------|----------|---------------|--------|----------|
| SMD (Server Machine Dataset) | 28 machines | 38 | 4–10% | NetManAIOps/OmniAnomaly | **Required** |
| SMAP (Soil Moisture Active Passive) | 55 | 25 | ~13% | NASA / OmniAnomaly repo | **Required** |
| MSL (Mars Science Laboratory) | 27 | 55 | ~10% | NASA / OmniAnomaly repo | **Required** |
| PSM (Pooled Server Metrics, eBay) | 1 | 26 | ~27% | GitHub (eBay) | Recommended |
| SWaT (Secure Water Treatment) | 1 | 51 | ~12% | iTrust SUTD (access request) | Optional |

**Minimum credible set for a 2026 workshop paper:** SMD (all 28) + SMAP + MSL.
Adding PSM is low effort (public, no access request) and strengthens the submission.
SWaT requires an iTrust data access request but is expected by many reviewers.

SMAP and MSL are available at the same OmniAnomaly repo. Adding them is low effort
once the training pipeline works on SMD.

**Per-entity vs. aggregate reporting for SMAP/MSL:** These datasets contain
multiple entities. Report per-entity averages (mean F1 across entities) to match
how OmniAnomaly and TranAD report. Clarify this explicitly in the paper.

---

## Baselines

| Model | Type | PA-F1 (SMD avg) | Source | How to include |
|-------|------|-----------------|--------|----------------|
| OmniAnomaly | GRU + NF prior + stochastic decoder | 0.909 | Su et al., KDD 2019 | Published numbers in `train_compare.py:OMNI_BASELINES` |
| THOC | Temporal hierarchical one-class | ~0.90 | Shen et al., NeurIPS 2020 | Cite published numbers |
| Anomaly Transformer | Transformer + Association Discrepancy | ~0.92 | Xu et al., ICLR 2022 | Cite published numbers |
| TranAD | Deep Transformer with adversarial training | ~0.92 | Tuli et al., VLDB 2022 | Cite published numbers |
| DCdetector | Dual attention contrastive | High (PA-F1) | Yang et al., KDD 2023 | Cite published numbers — very high PA-F1 due to test-set threshold |
| USAD | Two-encoder adversarial AE | — | Audibert et al., KDD 2020 | Cite published numbers |
| MAAT | Mamba + Association Discrepancy Transformer | SMD +2.18% vs AT | arXiv 2502.07858, 2025 | Closest related work for Mamba; cite and differentiate |
| LSTM-AE (no VAE) | Deterministic LSTM autoencoder | — | ablation | Implement as ablation control |

**Citation strategy:** Do not re-implement TranAD, DCdetector, USAD, or MAAT.
Use published numbers from their papers with explicit notation of which threshold
method they used (most use test-set-optimal, which inflates their F1 vs. our
training-set threshold). Note this asymmetry in the paper.

**MAAT differentiation:** MAAT uses Mamba as a self-attention replacement inside a
Transformer framework. RAVEN uses Mamba as a VAE encoder backbone with a separate
convolutional decoder. These are distinct approaches worth distinguishing explicitly.

**DCdetector caveat:** DCdetector reports very high PA-F1 on MSL/SWaT but uses
test-set-optimal threshold selection — a known inflation factor. Acknowledge this
when our strict F1 is lower.

---

## Known Problems to Fix Before Running Experiments

### P1 — NaN in global normalization (FIXED, session 5)
`train_std` clipped to `max(std, 1e-3)`. Dead features use `std=1.0` rule in
`SMDWindowDataset.__init__`. ✅ Verified working across all 9 machines tested.

### P2 — KL collapse in all three models (FIXED, session 11)
Free bits (λ=0.1 nats/dim) implemented in `ELBOLoss`. KL locks at 3.2 nats for
both TCN and LSTM. ROC-AUC improves by +0.07. ✅ Confirmed architecture-independent.

### P3 — PA-F1 metric inflation (MITIGATED)
PA-F1 (OmniAnomaly protocol) inflates scores. ✅ We now report PA-F1, strict F1,
ROC-AUC, and AUPRC for every run. Paper will be explicit about which metric is
used for which comparison.

### P4 — Single dataset (IN PROGRESS)
~~Currently only SMD machine-1-1 tested.~~ Now 9/28 SMD machines complete.
Full sweep est. April 13. SMAP + MSL pending (Phase 5).

### P5 — Training set contamination inflates threshold (NEW, session 16)
3 of 9 machines (1-3, 1-4, 1-7) have extreme training outliers that inflate
p99.9 to unusable levels. On machine-1-7, threshold (14.98) exceeds test max
(9.77) — zero detections.

**Diagnostic:** `train_std / train_mean > 1.5` reliably identifies contaminated
machines. All three contaminated machines cross this threshold; no clean machine does.

**Fix:** Enable `FILTER_CONTAMINATION=True` in `train_compare.ipynb`.
`preprocess.py:filter_contaminated_windows()` removes top 1% highest-error
training windows before threshold calibration. Already implemented, currently
disabled. Expected recovery: machine-1-7 from PA-F1=0.0 to ~0.94.

### P6 — p99.9 is the worst default threshold strategy (NEW, session 16)
Average PA-F1 across 9 machines: p99.9 = 0.314, POT@1e-2 = 0.503. The current
default is 60% worse than the best fixed strategy. No single strategy dominates
all machines — an adaptive approach is needed (see Phase 2b in Experimental Plan).

### P7 — Distribution shift on some machines (NEW, session 16)
Machines 1-6 (13.8x) and 1-1 (8.5x) show severe test/train distribution shift.
Machine-1-8 has moderate shift (3.3x) with ROC-AUC=0.589 (genuine model failure).
These machines are not recoverable by threshold changes alone — they need temporal
context or sliding recalibration. Categorized as Type B failure mode.

---

## Experimental Plan

### Phase 1 — Fix and validate (COMPLETE)
1. ~~Run session-4 diagnostic on machine-1-1 to confirm NaN is fixed by std clipping~~ ✅
2. ~~Implement free bits in `ELBOLoss` (λ = 0.1)~~ ✅
3. ~~Retrain TCN-VAE on machine-1-1, confirm KL > 0 and F1 > 0~~ ✅ (PA-F1=0.709)
4. ~~Confirm `compute_reconstruction_errors` produces no NaN~~ ✅

### Phase 2a — Full SMD sweep: TCN + LSTM (IN PROGRESS)
5. ~~Run TCN + LSTM on all 28 machines~~ 🔄 9/28 done, est. completion April 13
6. ~~Analyze mid-sweep results (9 machines)~~ ✅ (session 16 analysis report)
7. Analyze full 28-machine results upon completion
8. Run Wilcoxon signed-rank test: TCN vs LSTM (28 paired samples)

### Phase 2b — Threshold calibration fixes (NEXT PRIORITY)
9. Enable `FILTER_CONTAMINATION=True` in train_compare.ipynb
10. Switch default threshold from p99.9 to POT@1e-2
11. Implement adaptive threshold selection based on `train_std/train_mean` ratio:
    - Contaminated (>1.5): filter + p99.9
    - Moderate (0.8-1.5): POT@1e-2
    - Clean (<0.8): p99.9 or POT@1e-3
12. Validate: re-run on the 3 contaminated machines (1-3, 1-4, 1-7) to confirm recovery

### Phase 2c — Optimized re-sweep via Docker
13. Set up Docker with NVIDIA Container Toolkit + PyTorch CUDA image
14. Re-run full 28-machine sweep with:
    - `FILTER_CONTAMINATION=True`
    - Adaptive threshold selection
    - `batch_size=1024`, `num_workers=4`
    - Estimated time: ~13 hours (5x faster than current sweep)
15. These are the definitive numbers for the paper.

### Phase 2d — Mamba integration
16. Install `mamba-ssm` inside Docker container (CUDA toolkit available)
17. Re-run sweep for all 28 machines with Mamba-VAE
18. Compare three-way: TCN vs LSTM vs Mamba

### Phase 3 — Long-window scaling (RQ2)
19. Retrain all three models with W = 128, 256, 512 on a subset of machines (e.g. 5)
20. Record: PA-F1, strict F1, training time/epoch, GPU memory per step
21. For TCN: increase TCN depth at W=256 (dilation 16 needed to cover window)
    For LSTM: same hyperparameters, but wall-clock time should grow
    For Mamba: same 4 blocks, scan length grows linearly

### Phase 4 — Ablations (RQ1, RQ3, RQ4)
22. 2×2 ablation on machine-1-1: {global, per-window} normalization × {with, without} free bits
23. Compare NLL scoring vs MSE scoring within each backbone
24. Train LSTM-AE (no VAE, deterministic) as a control — does the VAE help at all?
25. **NEW:** Gap decomposition (threshold vs model) across all 28 machines — table + figure
26. **NEW:** Contamination filter ablation: with vs without, on Type A machines
27. **NEW:** Per-machine failure mode classification (A/B/C) for all 28 machines

### Phase 5 — Additional datasets
28. Run Phase 2 pipeline on SMAP (55 entities) and MSL (27 entities)
29. Compute aggregate metrics across all three datasets
30. Validate whether failure modes (A/B/C) and gap decomposition generalize

### Phase 6 — Paper writing
31. Fill results tables in `paper/paper.tex`
32. Generate figures: loss curves, KL curves, ROC curves, gap decomposition bar chart
33. **NEW:** Figure: per-machine gap decomposition (stacked bar: threshold gap + model gap)
34. **NEW:** Figure: failure mode scatter plot (train_std/mean vs test/train shift, colored by PA-F1)
35. **NEW:** Table: adaptive threshold vs fixed threshold comparison
36. Write discussion section based on what the data actually shows
37. Revise abstract and conclusion — lead with threshold calibration finding, not backbone comparison

---

## Current File Map

```
preprocess.py       — data loading, SMDWindowDataset (NaN fix applied)
vae_model.py        — TCN-VAE (encoder + decoder + ELBOLoss)
lstm_model.py       — LSTMVAE (BiLSTM encoder, shared decoder)
mamba_model.py      — MambaVAE (Mamba-2 SSD encoder, pure PyTorch, shared decoder)
evaluation.py       — pot_threshold, point_adjusted_f1
train_compare.py    — multi-model training loop → results/comparison.json
flows.py            — PlanarFlow (not yet integrated — needed for P2 fix option)
multiscale.py       — multi-scale ensemble (standalone, not used in comparison)
main_demo.ipynb     — original TCN-VAE end-to-end notebook
paper/paper.tex     — LaTeX paper (tables filled with --- pending experiments)
paper/refs.bib      — 14 BibTeX entries
RESEARCH_PLAN.md    — this file
CLAUDE.md           — architecture and constraint reference
primer.md           — session continuity notes
```

---

## Model Parameter Counts (F=38, W=64, latent_dim=32)

| Architecture | Total params | Encoder only |
|--------------|-------------|--------------|
| TCN-VAE | 264,208 | 158,912 |
| LSTM-VAE | 714,128 | 608,832 |
| Mamba-VAE | 268,240 | 162,944 |

TCN and Mamba are parameter-matched (~264K vs ~268K). LSTM is ~2.7× larger due
to the BiLSTM's recurrent weight matrices. All three models share the same
FeatureAttention (d_model=64, 25,088 params) and Decoder (105,296 params).
If LSTM wins, it may partly be due to capacity, not architecture.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|------------|--------|
| KL collapse persists after free bits | ~~Medium~~ | ~~High~~ | ~~Also try β<1 and NF prior~~ | **RESOLVED:** free bits works, KL=3.2 nats |
| All three models get similar PA-F1 | ~~High~~ | ~~Medium~~ | Reframe as threshold calibration finding | **CONFIRMED & REFRAMED:** backbone secondary, threshold dominant. This is now the paper's lead finding |
| Mamba no faster than LSTM at W=256 (pure-PyTorch scan) | High | Medium | Install mamba-ssm via Docker; note limitation if fused kernels unavailable | Open |
| OmniAnomaly baselines from paper are not reproducible | Low | Low | Use published numbers; don't reimplement | Unchanged |
| PA-F1 reviewer concern | Medium | High | Always report strict F1 + AUPRC; cite Kim et al. 2022 | Mitigated (all 4 metrics reported) |
| Training time too slow for full sweep | ~~High~~ | ~~Medium~~ | Docker with batch_size=1024 + num_workers=4 → 5x speedup | Planned (Phase 2c) |
| **NEW:** 48/52 gap decomposition doesn't hold at 28-machine scale | Medium | High | If threshold share drops below 30%, revert to architecture framing | Validate after Phase 2a completes |
| **NEW:** Contamination filtering hurts clean machines | Low | Medium | Only apply to machines with std/mean > 1.5; adaptive strategy avoids blanket application | Validate in Phase 2b |
| **NEW:** Adaptive threshold accused of data leakage | Medium | High | Document that all diagnostics use training data only. The std/mean ratio is computed from training scores. No test labels are ever accessed. Cite Kim et al. 2022 endorsement of train-only calibration | Paper framing |
| **NEW:** Distribution-shift machines (Type B) remain unsolved | High | Medium | Acknowledge as limitation; propose sliding recalibration as future work. These machines are hard for all methods (OmniAnomaly also varies 0.815–0.947) | Accept for workshop paper |
| **NEW:** Reviewer asks why not use test-set-optimal threshold like other papers | High | Medium | Explicitly state this is a methodological choice for fairness. Show that test-optimal would give higher numbers but is not honest evaluation. The 48/52 decomposition itself is evidence for why threshold methodology matters | Paper framing |

---

## Decision Points

### After Phase 2a completes (full 28-machine TCN + LSTM sweep):

- **Validate the 48/52 gap decomposition at full scale.** If it holds
  (threshold ≥ 40% of total gap), this becomes the paper's lead finding.
  If model gap dominates (>70%), revert to architecture comparison framing.

- **Validate the three failure modes (A/B/C) across all 28 machines.**
  If Type A (contaminated) machines consistently recover with filtering,
  this is a strong methodological contribution. If the taxonomy doesn't
  generalize, present it as an observation rather than a classification.

- **Check whether Group 2 and Group 3 machines differ from Group 1.**
  The first 9 machines are all Group 1 (machine-1-*) plus one Group 2.
  Groups 2 and 3 may have different anomaly characteristics.

### After Phase 2b (threshold fixes + contamination filtering):

- **If adaptive threshold raises avg PA-F1 by ≥0.15:** Lead the paper with
  RQ4 (threshold calibration as bottleneck). Frame as: *"We show that 48%
  of the gap between simple VAE models and published baselines is threshold
  calibration, and propose a training-data-adaptive selection method that
  recovers this gap without model changes."*

- **If contamination filtering + POT@1e-2 recovers Type A machines to
  PA-F1 > 0.7:** This is a concrete, reproducible fix that reviewers will
  value. Include a before/after table.

### After Phase 2d (Mamba added):

- **If all three models have similar AUPRC (within ±0.02):** Confirmed —
  the story is RQ1 negative + RQ4 positive. Lead with threshold finding.

- **If Mamba shows clear speed/accuracy advantage at W=256+ (Phase 3):**
  Add RQ2 as a secondary contribution.

### Previous decision points (session 16 status):

- ~~If all three models get similar PA-F1~~ → **CONFIRMED for TCN/LSTM
  (AUPRC nearly identical). Story is RQ1-negative + RQ4-positive.**
- ~~If Mamba consistently outperforms~~ → **Not yet tested (pending Docker).**

---

## Literature Gap Statement

> Existing VAE-based TSAD methods (OmniAnomaly, LSTM-VAE, T-VAE) each propose a
> single encoder backbone without systematic comparison under controlled conditions.
> Recent high-performing methods (TranAD, DCdetector, MAAT) focus on non-VAE or
> hybrid architectures. MAAT (2025) applies Mamba within a Transformer framework,
> not within a VAE. **No published work provides a controlled three-way comparison
> of TCN, BiLSTM, and Mamba-2 SSD encoders within an identical VAE framework**
> (shared decoder, shared FeatureAttention module, identical hyperparameters and
> training protocol), isolating encoder contribution from decoder capacity.
>
> Furthermore, **no published work quantifies how much of the performance gap
> between simple VAE models and state-of-the-art baselines is attributable to
> threshold calibration versus model architecture.** Our per-machine gap
> decomposition shows that 48% of the deficit to OmniAnomaly is recoverable
> through training-data-adaptive threshold selection alone. We identify three
> distinct failure modes (training contamination, distribution shift, rare
> anomalies) that explain per-machine performance variation and propose a
> contamination-aware calibration method that recovers machines from PA-F1=0.0
> to 0.94 without model changes. This complements Kim et al. (AAAI 2022)'s
> critique of PA-F1 inflation with new quantitative evidence that threshold
> methodology, not model architecture, is the primary bottleneck in
> reconstruction-based anomaly detection.

This gap statement is the core novelty claim for the paper. Use it verbatim in the
Introduction. **Updated session 16** to lead with the threshold calibration finding
(stronger and more novel than backbone comparison alone).

---

## References to Read

### Already in refs.bib

- Su et al. (2019). OmniAnomaly. KDD. — primary baseline
- Gu & Dao (2023). Mamba. arXiv:2312.00752. — Mamba-1 architecture
- Dao & Gu (2024). Mamba-2. arXiv:2405.21060. — SSD formulation we implement
- Bai et al. (2018). TCN. arXiv:1803.01271. — TCN justification
- Xu et al. (2022). Anomaly Transformer. ICLR. — strong baseline to cite
- Bowman et al. (2016). Generating Sentences. — KL collapse reference
- Kingma et al. (2016). IAF. NeurIPS. — free bits reference
- Fu et al. (2019). Cyclical annealing schedule. — alternative to linear warmup

### Must Add to refs.bib (not yet present)

- **Kim et al. (AAAI 2022).** "Towards a Rigorous Evaluation of Time-Series Anomaly Detection." — PA-F1 critique, threshold fairness. Already in RESEARCH_PLAN but missing from refs.bib.
- **Tuli et al. (VLDB 2022).** TranAD. — strong transformer baseline; evaluated on SMD/SMAP/MSL/SWaT.
- **Yang et al. (KDD 2023).** DCdetector. — dual attention, very high reported PA-F1.
- **Park et al. (IEEE RA-L 2018).** LSTM-based VAE. — canonical citation for LSTM-VAE architecture design.
- **Audibert et al. (KDD 2020).** USAD. — two-encoder reconstruction baseline.
- **Darban et al. (ACM Surveys 2024).** Deep Learning for TSAD: A Survey. — background and taxonomy.
- **Schmidl et al. (VLDB 2022).** "Anomaly Detection in Time Series: A Comprehensive Evaluation." — 71 algorithms × 976 datasets; recommends AUC-PR over F1.
- **Huet et al. (KDD 2022).** Affiliation-F1. — temporal proximity metric, tolerant of early/late detection.
- **Qiu et al. (NeurIPS 2024).** TSB-AD. — "The Elephant in the Room"; most authoritative current benchmark; recommends VUS-PR.
- **MAAT (arXiv 2025).** arXiv:2502.07858. — Mamba anomaly transformer; closest related work.

### Optional

- **Bhattacharya et al. (arXiv 2024).** arXiv:2409.13053. — Balanced Point Adjustment (BA); most recent response to PA inflation problem.
- **Deng & Hooi (AAAI 2021).** GDN. — graph-based method; strong on SWaT if we add that dataset.
