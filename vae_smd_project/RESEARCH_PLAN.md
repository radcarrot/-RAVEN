# Research Plan: VAE Backbone Comparison for Time-Series Anomaly Detection

## One-Line Goal

Determine whether the temporal encoder backbone (TCN vs BiLSTM vs Mamba-2) or
the training pathology (KL collapse, normalisation strategy) is the dominant
factor in VAE-based multivariate anomaly detection — with a secondary focus on
whether Mamba's linear-time scaling gives a practical advantage at long windows.

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
> decoder capacity)?

**Hypothesis:** Backbone choice matters less than normalization strategy and
KL regularization. All three converge to similar PA-F1 because KL collapses
identically regardless of encoder.

### RQ2 (Primary — scaling)
> At long window sizes (W = 64 → 256 → 512), does Mamba's O(L) scan give a
> practical accuracy or speed advantage over LSTM (O(L) but large constant) and
> TCN (needs more layers)?

**Hypothesis:** At W=64 all models are equivalent. At W≥256, LSTM training slows
significantly; TCN needs more dilated layers to cover the window; Mamba maintains
linear time with the same block count. If anomaly signatures span >61 timesteps
(TCN's default receptive field), Mamba and LSTM should pull ahead.

### RQ3 (Secondary — pathology)
> Is KL collapse architecture-independent, and does fixing it (free bits) improve
> anomaly discrimination?

**Hypothesis:** Yes — all three backbones collapse to KL≈0 under the same
annealing schedule because the ConvTranspose decoder is powerful enough to ignore
the latent code. Free bits or normalizing flow prior will improve latent
utilization equally across all three.

---

## What Would Make This Publishable

### Minimum bar (workshop paper)
- [ ] Results on ≥ 3 datasets (SMD + SMAP + MSL)
- [ ] Fix KL collapse and show it matters, OR frame collapse as the key finding
- [ ] Both PA-F1 and strict F1 reported (PA-F1 inflation is a known issue)
- [ ] ≥ 2 proper baselines beyond our own models (e.g. THOC, Anomaly Transformer)
- [ ] One clear answered research question (not just "we compare X Y Z")

### Stronger bar (mid-tier conference)
- All of the above, plus:
- [ ] Long-window scaling experiment (RQ2) showing Mamba advantage at W≥256
- [ ] Statistical significance testing across 28 machines (Wilcoxon signed-rank)
- [ ] Ablation table: backbone × normalization × KL fix → 3×2×2 = 12 conditions
- [ ] Runtime / memory profiling per architecture

---

## Datasets

| Dataset | Machines | Features | Anomaly ratio | Source |
|---------|----------|----------|---------------|--------|
| SMD (Server Machine Dataset) | 28 | 38 | 4–10% | NetManAIOps/OmniAnomaly |
| SMAP (Soil Moisture Active Passive) | 55 | 25 | ~13% | NASA |
| MSL (Mars Science Laboratory) | 27 | 55 | ~10% | NASA |

SMAP and MSL are available at the same OmniAnomaly repo. Adding them is low effort
once the training pipeline works on SMD.

---

## Baselines

| Model | Type | PA-F1 (SMD avg) | Source |
|-------|------|-----------------|--------|
| OmniAnomaly | GRU + NF prior + stochastic decoder | 0.909 | Su et al., KDD 2019 |
| THOC | Temporal hierarchical one-class | ~0.90 | Shen et al., NeurIPS 2020 |
| Anomaly Transformer | Transformer + Association Discrepancy | ~0.92 | Xu et al., ICLR 2022 |
| LSTM-AE (no VAE) | Deterministic LSTM autoencoder | — | ablation baseline |

OmniAnomaly numbers are embedded in `train_compare.py:OMNI_BASELINES`.
THOC and Anomaly Transformer: use published numbers from their papers (no need
to reimplement unless a reviewer asks).

---

## Known Problems to Fix Before Running Experiments

### P1 — NaN in global normalization (FIXED in preprocess.py)
`train_std` now clipped to `max(std, 1e-3)`. Eliminates NaN/Inf for near-constant
features. **Must verify by running the diagnostic from session 4 before trusting results.**

### P2 — KL collapse in all three models
All backbones collapse to KL ≈ 0.002–0.03 within 20 epochs. The latent code
carries no information; models are effectively deterministic autoencoders.

**Fix options (pick one to implement):**
- **Free bits** (easiest): add `max(KL_j, λ)` floor per latent dim (λ = 0.1 nats)
  in `ELBOLoss.forward()`. Prevents collapse without changing architecture.
- **Normalizing flow prior** (already scaffolded): wire `flows.py:PlanarFlow`
  between `reparameterize()` and `decoder()`. Makes posterior more expressive.
- **β-VAE with β < 1**: set `MAX_BETA = 0.1` — reconstruction dominates, KL
  stays nonzero. Simple but may reduce anomaly discrimination.

### P3 — PA-F1 metric inflation
PA-F1 (OmniAnomaly protocol) inflates scores: if *any* prediction in an anomaly
segment is positive, the *entire* segment counts as detected. This can make
near-random detectors look good (Kim et al., 2022).

**Fix:** always report both PA-F1 and strict point-wise F1. In the paper, be
explicit about which metric is used for each comparison. When comparing to
OmniAnomaly, use PA-F1 (their metric). For internal ablations, use strict F1.

### P4 — Single dataset in current experiments
Currently only SMD machine-1-1 tested. Need all 28 SMD machines + SMAP + MSL
before the paper is credible.

---

## Experimental Plan

### Phase 1 — Fix and validate (do first)
1. Run session-4 diagnostic on machine-1-1 to confirm NaN is fixed by std clipping
2. Implement free bits in `ELBOLoss` (λ = 0.1)
3. Retrain TCN-VAE on machine-1-1, confirm KL > 0 and F1 > 0
4. Confirm `compute_reconstruction_errors` produces no NaN

### Phase 2 — Full SMD sweep
5. Run `train_compare.py` — all 3 models × 28 machines → `results/comparison.json`
6. Compute per-machine PA-F1, strict F1, ROC-AUC for all three
7. Run Wilcoxon signed-rank test: TCN vs LSTM, TCN vs Mamba, LSTM vs Mamba (28 paired samples)

### Phase 3 — Long-window scaling (RQ2)
8. Retrain all three models with W = 128, 256, 512 on a subset of machines (e.g. 5)
9. Record: PA-F1, strict F1, training time/epoch, GPU memory per step
10. For TCN: increase TCN depth at W=256 (dilation 16 needed to cover window)
    For LSTM: same hyperparameters, but wall-clock time should grow
    For Mamba: same 4 blocks, scan length grows linearly

### Phase 4 — Ablations (RQ3)
11. 2×2 ablation on machine-1-1: {global, per-window} normalization × {with, without} free bits
12. Compare NLL scoring vs MSE scoring within each backbone
13. Train LSTM-AE (no VAE, deterministic) as a control — does the VAE help at all?

### Phase 5 — Additional datasets
14. Run Phase 2 on SMAP (55 entities) and MSL (27 entities)
15. Compute aggregate metrics across all three datasets

### Phase 6 — Paper writing
16. Fill results tables in `paper/paper.tex`
17. Generate figures: loss curves, KL curves, ROC curves, scaling plot
18. Write discussion section based on what the data actually shows
19. Revise abstract and conclusion to match actual findings

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

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| KL collapse persists after free bits | Medium | High | Also try β<1 and NF prior |
| All three models get similar PA-F1 | High | Medium | Reframe as "backbone doesn't matter, collapse does" — still publishable |
| Mamba no faster than LSTM at W=256 (pure-PyTorch scan) | High | Medium | Note limitation; compare with mamba-ssm extension if available |
| OmniAnomaly baselines from paper are not reproducible | Low | Low | Use published numbers; don't reimplement |
| PA-F1 reviewer concern | Medium | High | Always report strict F1 alongside; cite Kim et al. 2022 |
| Training time on CPU too slow for 28×3 models | High | Medium | Use GPU; or reduce to 10 representative machines for initial submission |

---

## Decision Points

After Phase 2 (full SMD results), decide:

- **If Mamba consistently outperforms TCN/LSTM on specific machine types:**
  Investigate why — likely long-range dependency or sparse anomaly signatures.
  This supports RQ1 with a positive finding.

- **If all three models have similar PA-F1 (within ±0.02):**
  The story becomes RQ3 — the bottleneck is the training pathology, not the
  backbone. This is still publishable: *"We show that KL collapse and normalization
  strategy account for most of the variance in VAE anomaly detection; backbone
  choice is secondary."*

- **If Mamba shows clear speed/accuracy advantage at W=256+:**
  Lead with RQ2. This is the strongest contribution because it's a positive,
  actionable finding with practical implications.

---

## References to Read

- Su et al. (2019). OmniAnomaly. KDD. — our primary baseline
- Gu & Dao (2023). Mamba. arXiv:2312.00752. — Mamba-1 architecture
- Dao & Gu (2024). Mamba-2. arXiv:2405.21060. — SSD formulation we implement
- Bai et al. (2018). TCN. arXiv:1803.01271. — TCN justification
- Kim et al. (2022). Towards a Rigorous Evaluation of Time-Series Anomaly Detection. — PA-F1 critique
- Xu et al. (2022). Anomaly Transformer. ICLR. — strong baseline to cite
- Bowman et al. (2016). Generating Sentences. — KL collapse reference
- Kingma et al. (2016). IAF. NeurIPS. — free bits reference
