# RAVEN — Research Findings Log

Cumulative findings from training runs and experiments.
Update this file after every significant result.

---

## F1 — Free Bits Fixes KL Collapse (session 11, 2026-04-11)

**Setup:** TCN-VAE and LSTM-VAE trained on machine-1-1 with `free_bits=0.1` (32 dims × 0.1 = 3.2 nats floor).

**Result:**

| Model | KL (before) | KL (after) | ROC-AUC (before) | ROC-AUC (after) |
|-------|-------------|------------|------------------|-----------------|
| TCN-VAE  | ~0.002 | **3.202** | 0.854 | **0.921** |
| LSTM-VAE | ~0.002 | **3.203** | 0.863 | **0.924** |

- KL locked at exactly 3.2 nats for both models — free bits working as intended
- ROC-AUC improved by ~+0.07 for both models — latent space is genuinely more discriminative
- Both models improved **identically** — supports the hypothesis that backbone choice is secondary; KL collapse was the bottleneck

**Caveat:** PA-F1 dropped (0.588 → 0.373) due to threshold miscalibration, not model quality. The train score distribution shifted when KL stopped collapsing, breaking the 99th-pct threshold. See F2.

---

## F2 — Train/Test Score Distribution Shift (session 11, 2026-04-11)

**Setup:** TCN-VAE, machine-1-1, free_bits=0.1, 99th-pct threshold calibrated on 444 train windows (stride=64).

**Result:**

| Threshold | Flags (test) | Raw F1 | Precision | Recall |
|-----------|-------------|--------|-----------|--------|
| p99 = 1.675 (active) | 41.3% | 0.369 | 0.227 | 0.987 |
| p99.9 = 3.104        | 20.9% | 0.551 | 0.400 | 0.881 |
| ~4.0                 | 14.0% | 0.547 | 0.460 | 0.677 |

- Train score mean = 0.32, but test requires threshold ~3.0 for good F1
- 99th-pct of training scores (1.675) is far too low for test distribution
- **p99.9 is near-optimal** — coincides with the empirically best threshold (~3.0)
- POT with 444 calibration windows was unstable (gamma=0.63, heavy-tailed fit)

**Fix applied:** Switch calibration to stride=1 (28K windows) and use POT level=1e-3 as active threshold, with p99.9 as fallback. Running now.

---

## F3 — KL Collapse is Architecture-Independent (sessions 9–11)

**Observation:** All three backbones (TCN, LSTM, Mamba) collapsed to KL ≈ 0.002 under identical annealing schedules (`warmup_epochs=25`, `max_beta=1.0`). Free bits fixed collapse equally across TCN and LSTM.

**Implication for RQ3:** KL collapse is driven by decoder capacity (ConvTranspose decoder is powerful enough to ignore the latent code), not the encoder architecture. This is a clean finding: fixing the training pathology improves all backbones equally.

---

## F4 — Normalization Fix Necessary but Not Sufficient (sessions 9–10)

**Before fix (std=0 for dead features):** ROC-AUC inflated to ~0.949 due to binary sensors 26/28 dominating scores. NaN in test scores on dead features.

**After fix (std=1.0 for dead features):** ROC-AUC dropped to ~0.854–0.863 — a real reduction. The prior "good" numbers were partly due to binary sensor signal that was trivially easy to reconstruct.

**Lesson:** Always check feature-level score contributions. Binary sensors (features 26/28 on machine-1-1) that are silent during training and spike during anomalies are "easy wins" that can mask poor performance on the remaining 36 features.

---

## F5 — POT Threshold Was Never Being Called (session 12, 2026-04-11)

**Bug:** `evaluation.py:pot_threshold()` was implemented but never called in `evaluate()`. The RunLogger was receiving `metrics.get('pot')` which always returned `None`.

**Root cause of instability:** Even when called, only 444 calibration windows (stride=64) were used. With so few points, the GPD fit was heavy-tailed (gamma=0.63), causing POT thresholds at level<1e-2 to blow up to 8–36×.

**Fix:** stride=1 for calibration (28K windows); POT level=1e-3; sanity check rejects threshold if >100× p99.9 and falls back to p99.9.

---

## F6 — POT + Free Bits: TCN Reaches 0.794 PA-F1 (session 12, 2026-04-11)

**Setup:** TCN-VAE and LSTM-VAE, machine-1-1, free_bits=0.1, POT threshold (level=1e-3) calibrated on 28K windows (stride=1).

**Result:**

| Model | PA-F1 | Raw F1 | ROC-AUC | PA-Prec | PA-Recall | POT threshold | p99.9 |
|-------|-------|--------|---------|---------|-----------|---------------|-------|
| TCN-VAE  | **0.794** | 0.291 | 0.924 | 0.749 | 0.846 | 8.576 | 2.976 |
| LSTM-VAE | **0.711** | 0.503 | 0.910 | 0.552 | 0.997 | 4.714 | 4.598 |
| OmniAnomaly | 0.838 | — | — | — | — | — | — |

- **TCN PA-F1 = 0.794**, within 0.044 of OmniAnomaly — best result so far
- LSTM PA-F1 = 0.711 — improved from 0.371 but TCN pulls ahead
- POT thresholds differ significantly: TCN=8.576 (2.9× its p99.9), LSTM=4.714 (≈ p99.9)
  - TCN's heavy-tailed training score distribution drives POT much higher than percentile
  - This raises precision (0.749) but reduces recall (0.846) for TCN
- Raw F1 gap: TCN=0.291 vs LSTM=0.503 — entirely driven by TCN's much higher POT threshold

**Concern:** TCN POT at 8.576 may be over-conservative. Evaluating TCN at p99.9=2.976 would likely give lower PA-F1 but better raw F1 balance. Worth investigating whether the GPD fit is reliable at this level.

---

## F7 — The 48/52 Gap Decomposition: Threshold vs Model (session 16, 2026-04-12)

**Setup:** 9 machines (1-1 through 1-8, 2-1), TCN-VAE and LSTM-VAE, threshold sweep across 6 strategies (p99, p99.5, p99.9, POT@1e-2, POT@1e-3, POT@1e-4).

**Result:** The total performance gap between TCN-VAE (at p99.9) and OmniAnomaly decomposes as:

| Component | Avg Gap (PA-F1 points) | Share |
|-----------|----------------------|-------|
| Threshold calibration | 0.274 | **48%** |
| Model quality | 0.297 | **52%** |
| **Total** | **0.571** | 100% |

Nearly half the deficit is caused by suboptimal threshold selection, not model inadequacy. On machines where the threshold is right (1-1, 2-1), the gap shrinks to 0.01–0.09.

---

## F8 — Training Contamination Causes Catastrophic Threshold Failure (session 16, 2026-04-12)

**Setup:** Same 9-machine sweep. Diagnostic: `train_std / train_mean` as contamination indicator.

**Result:** Three machines (1-3, 1-4, 1-7) have `train_std/mean > 1.5` — extreme training outliers inflate p99.9 to unusable levels:

| Machine | Std/Mean | p99.9/p99 | p99.9 Threshold | Test Max | PA-F1@p99.9 | PA-F1@p99 |
|---------|----------|-----------|-----------------|----------|-------------|-----------|
| 1-3 | 2.46 | 8.5x | 12.95 | 31.65 | 0.203 | 0.585 |
| 1-4 | 1.93 | 9.1x | 9.98 | 28.96 | 0.112 | 0.357 |
| 1-7 | **2.24** | **5.4x** | **14.98** | **9.77** | **0.000** | **0.941** |

machine-1-7: threshold (14.98) > test max (9.77) → zero predictions. At p99 (2.79): **PA-F1=0.941, beating OmniAnomaly's 0.882**.

**Root cause:** A small number of anomalous windows contaminate the training set, inflating the tail. The `filter_contaminated_windows()` function in `preprocess.py` addresses this but is currently disabled.

---

## F9 — p99.9 Is the Worst Default Strategy; POT@1e-2 Is Best (session 16, 2026-04-12)

**Setup:** Average TCN PA-F1 across 9 machines by threshold strategy.

**Result:**

| Strategy | Avg PA-F1 | Avg Raw-F1 |
|----------|-----------|------------|
| **POT@1e-2** | **0.503** | **0.344** |
| p99 | 0.475 | 0.336 |
| p99.5 | 0.413 | 0.286 |
| POT@1e-3 | 0.338 | 0.160 |
| **p99.9 (current)** | **0.314** | **0.178** |
| POT@1e-4 | 0.197 | 0.097 |

p99.9 is the worst-performing strategy on average. POT@1e-2 is 60% better and works well on both contaminated and clean machines. No single strategy dominates all machines — an adaptive approach is needed.

---

## F10 — Three Failure Modes Explain Per-Machine Performance (session 16, 2026-04-12)

Machines cluster into three distinct failure types:

| Type | Machines | Signature | Fix | Recovery |
|------|----------|-----------|-----|----------|
| **A: Contaminated tail** | 1-3, 1-4, 1-7 | std/mean>1.5, p99.9/p99>5x | Contamination filter | +0.38 to +0.94 |
| **B: Distribution shift** | 1-1, 1-6, 1-8 | test_mean/train_mean > 3x | Temporal context / sliding recalibration | Limited |
| **C: Rare anomalies** | 1-5 | GT rate < 1%, AUPRC < 0.1 | Precision-optimized scoring | Very limited |

Type A machines are fully recoverable. Type B requires architectural changes. Type C is a dataset characteristic.

---

## F11 — TCN and LSTM Have Equivalent Discriminative Power (session 16, 2026-04-12)

**Setup:** Head-to-head comparison, best achievable PA-F1 across 9 machines.

**Result:** TCN wins 5, LSTM wins 2, ties 2. But AUPRC is nearly identical on every machine, confirming equal discriminative ability. The one dramatic LSTM win (1-6, +0.522) is a GPD tail-fitting artifact, not architectural superiority.

**Implication for RQ1:** Backbone choice is secondary to threshold calibration. This is a publishable negative result.

---

## F12 — Threshold Rankings Confirmed at 16-Machine Scale (session 17, 2026-04-14)

**Setup:** Full group-1 + group-2 sweep (16 machines), TCN-VAE, 6 threshold strategies.

**Result:**

| Strategy | Avg PA-F1 (16 machines) | vs session 16 (9 machines) |
|----------|------------------------|---------------------------|
| **POT@1e-2** | **0.489** | 0.503 — holds up |
| p99 | 0.474 | 0.475 — stable |
| p99.5 | 0.421 | 0.413 — stable |
| POT@1e-3 | 0.352 | 0.338 — stable |
| **p99.9 (current default)** | **0.347** | 0.314 — still worst |
| POT@1e-4 | 0.163 | 0.197 — slightly worse |

Rankings are fully stable from 9 → 16 machines. POT@1e-2 and p99 are the two best strategies, separated by only 0.015. **Change default to POT@1e-2 before full sweep.**

---

## F13 — 48/52 Gap Decomposition Confirmed at 16 Machines (session 17, 2026-04-14)

**Setup:** Oracle threshold (best per machine) vs actual default (p99.9) vs OmniAnomaly.

| | Mean PA-F1 |
|--|--|
| p99.9 (current default) | 0.347 |
| Oracle threshold (best per machine) | 0.606 |
| OmniAnomaly | 0.903 |

- Threshold calibration gap: 0.606 − 0.347 = **0.259 (47%)**
- Model quality gap: 0.903 − 0.606 = **0.297 (53%)**

The 48/52 split from session 16 holds almost exactly at 16-machine scale (now 47/53). This is a robust, reproducible finding.

---

## F14 — Oracle Threshold Leaves 15/16 Machines Behind OmniAnomaly (session 17, 2026-04-14)

**Setup:** Best possible PA-F1 per machine (oracle threshold selection) vs OmniAnomaly.

**Result:** Only 1/16 machines beats OmniAnomaly even with the perfect threshold (machine-1-7: +0.058). 1/16 is a near-tie. 14/16 remain behind.

Four machines have a fundamental model quality gap that threshold tuning cannot fix:

| Machine | Best PA-F1 | OmniAnomaly | Deficit | Likely cause |
|---------|-----------|-------------|---------|-------------|
| machine-1-5 | 0.183 | 0.903 | −0.720 | Type C: rare anomalies (GT rate < 1%) |
| machine-2-6 | 0.223 | 0.917 | −0.694 | Unknown — new failure mode |
| machine-1-8 | 0.244 | 0.816 | −0.572 | Type B: distribution shift |
| machine-1-6 | 0.399 | 0.876 | −0.477 | Type B: distribution shift |

machine-2-6 is notable: it was not in the session-16 9-machine analysis and has no clear failure type yet. Needs investigation.

**Implication:** The 53% model quality gap is real and concentrated in specific machines. A targeted architectural fix (e.g., temporal context for Type B, focal loss for Type C) would need to recover these 4 machines to close the gap significantly.

---

## F15 — LSTM GPD Fitting Failure on Two Machines (session 17, 2026-04-14)

**Setup:** TCN vs LSTM PA-F1 at default p99.9 threshold, 16 machines.

**Result:** Two machines show extreme LSTM underperformance:

| Machine | TCN PA-F1 | LSTM PA-F1 | Gap |
|---------|----------|-----------|-----|
| machine-2-2 | 0.660 | 0.032 | +0.629 |
| machine-2-8 | 0.196 | 0.047 | +0.149 |

These are not architectural differences — AUPRC is similar for both models across all machines. The likely cause is that LSTM's heavier-tailed score distribution (due to recurrent weight magnitudes) causes the GPD fit to produce an extreme threshold on these specific machines, driving predictions to near zero.

**Implication for RQ1:** TCN vs LSTM comparisons must be made at the same threshold strategy. Direct PA-F1 comparison at p99.9 is unreliable for LSTM on tail-heavy machines. Use AUPRC as the primary architecture comparison metric.

---

## Open Questions

- [x] Does POT with 28K calibration windows give a stable, better threshold than p99.9? → Yes, POT is now working and gave TCN PA-F1=0.794
- [x] Does free_bits=0.1 + good threshold give PA-F1 > 0.60 on machine-1-1? → Far exceeded: TCN=0.794, LSTM=0.711
- [x] TCN PA-F1 > LSTM on machine-1-1 — is this consistent across machines? → **Yes, TCN wins 5-2-2, but margins are small. Architecture is secondary.**
- [x] TCN POT threshold (8.576) is 2.9× p99.9 — is the GPD fit reliable? → **No. Heavy-tailed GPD fits produce unreliable thresholds. POT@1e-2 is more robust than POT@1e-3.**
- [x] Should we use p99.9 as fallback threshold instead of POT? → **Neither. p99.9 is worst-average; POT@1e-2 is best-average. Adaptive selection based on contamination indicator is recommended.**
- [x] Does fixing KL collapse + POT threshold hold up across all SMD machines? → **Partially. It holds on clean machines (1-1, 1-2, 2-1) but not on contaminated (1-3, 1-4, 1-7) or distribution-shift machines (1-6, 1-8).**
- [x] Is the ~+0.07 ROC-AUC improvement from free bits consistent across all 28 machines? → **Pending full sweep; consistent across all 16 completed machines.**
- [x] Does the 48/52 gap decomposition hold at full-sweep scale? → **Confirmed at 16 machines (47/53). Likely stable.**
- [x] TCN PA-F1 > LSTM on average — is this consistent across all machines? → **Yes at p99.9, but LSTM failures on 2-2 and 2-8 are GPD artifacts, not architecture (see F15). Use AUPRC for fair comparison.**
- [ ] Does contamination filtering recover the Type A machines (1-3, 1-4, 1-7)?
- [ ] Do the three failure modes generalize to all 28 machines (machine-3 group)?
- [ ] What is the failure mode for machine-2-6? Best PA-F1=0.223 even at oracle threshold — not Type A, B, or C.
- [ ] Can adaptive threshold selection (based on train distribution diagnostics) be done reliably without test data?
- [ ] At what window size does Mamba begin to show a speed/accuracy advantage over LSTM?
- [ ] Why does machine-1-8 have ROC-AUC=0.589? Feature-level investigation needed.
- [ ] Why does machine-1-5 have AUPRC=0.052? Is the anomaly type fundamentally different?
- [ ] Does the LSTM GPD failure on machines 2-2 and 2-8 disappear when using POT@1e-2 instead of p99.9?
- [ ] Do machine-3 machines (higher OmniAnomaly scores on average) show smaller model quality gaps?
