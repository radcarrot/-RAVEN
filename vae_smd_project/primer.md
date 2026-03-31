# Session Primer

> **Instructions for Claude:** At the **start** of every session, read this file to restore context.
> At the **end** of every session (or after any significant change), update this file:
> move "This Session" content into "Previous Sessions", then fill in a new "This Session" block.
> Keep "Previous Sessions" concise — bullet points only, no need to preserve full detail.

---

## Project in One Line

VAE with three interchangeable temporal encoder backbones (TCN / BiLSTM / Mamba-2 SSD) for unsupervised anomaly detection on the 38-channel Server Machine Dataset (SMD). Train on normal data only; anomalous windows produce high reconstruction error.

---

## File Map

```
preprocess.py       — data loading, SMDWindowDataset (NaN fix: std clipped to 1e-3)
vae_model.py        — VAE v3 (TCN + FeatureAttention + Learned Variance), ELBOLoss
lstm_model.py       — LSTMVAE (BiLSTM encoder, shared decoder), compute_reconstruction_errors
mamba_model.py      — MambaVAE (Mamba-2 SSD encoder, pure-PyTorch), compute_reconstruction_errors
evaluation.py       — pot_threshold, point_adjusted_f1
train_compare.py    — trains all 3 models on all 28 machines → results/ (via RunLogger)
results_logger.py   — RunLogger: structured experiment persistence (JSON + CSV + .npy)
plot_results.py     — generates comparison figures from saved RunLogger outputs
multiscale.py       — multi-scale ensemble (standalone)
flows.py            — PlanarFlow normalizing flow prior (standalone, not yet integrated)
main_demo.ipynb     — end-to-end TCN-VAE orchestration (now integrates RunLogger)
paper/paper.tex     — LaTeX research paper (tables filled with "---", fill after experiments)
paper/refs.bib      — 14 BibTeX references
RESEARCH_PLAN.md    — research questions, venue targets, experimental phases, risk register
foragent/           — research papers for reference
CLAUDE.md           — authoritative architecture/constraint reference
primer.md           — this file
```

---

## This Session

### 2026-03-28 (session 5) — LSTM-VAE, Mamba-VAE, research paper, results infrastructure

**New files created:**
- `lstm_model.py` — LSTMVAE: BiLSTM encoder (2-layer, hidden=128/dir) + shared Decoder
- `mamba_model.py` — MambaVAE: Mamba-2 SSD encoder (4 blocks, d_model=64, d_state=16, pure PyTorch)
- `train_compare.py` — Trains all 3 models × 28 machines; integrates RunLogger for structured output
- `results_logger.py` — RunLogger: saves JSON summary + CSV history + .npy arrays per run
- `plot_results.py` — 6 plot functions for comparison figures from saved RunLogger outputs
- `paper/paper.tex` — Full LaTeX paper (tables with "---" placeholders pending experiments)
- `paper/refs.bib` — 14 BibTeX entries
- `RESEARCH_PLAN.md` — Research questions, venue targets, experimental phases, risk register

**Bug fixes:**
- `preprocess.py`: clips `train_std` to `max(std, 1e-3)` → fixes NaN from near-constant features (session 4 root cause)
- `lstm_model.py`: added `attn_d_model=64` to decouple FeatureAttention from lstm_hidden (was 128, giving LSTM unfair advantage)
- `train_compare.py`: `point_adjusted_f1()` returns dict, not float — fixed to extract `['f1']`
- `train_compare.py`: `evaluate()` now returns raw arrays (prefixed `_`) alongside JSON-safe metrics for RunLogger
- `plot_results.py`: replaced Unicode `→` with `->` for Windows cp1252 compatibility
- `paper/paper.tex`: fixed tabular column spec `{lrr}` → `{lr}` (mismatch with 2-column table)
- `main_demo.ipynb`: fixed `auc` variable check from `'auc' in dir()` to try/except

**Notebook integration (main_demo.ipynb):**
- Cell 2: imports RunLogger + time
- Cell 4: creates RunLogger, logs hyperparameters
- Cell 9: times training, logs history
- Cell 14: logs train scores + thresholds
- Cell 17: logs test results + saves

**Verified all three models (11 tests, 9 validation checks):**
- Forward/backward pass, ELBOLoss, MSE scoring, evaluation, JSON serialization ✓
- Param counts: TCN=264,208, Mamba=268,240, LSTM=714,128
- TCN and Mamba parameter-matched (~264K vs ~268K); LSTM ~2.7× larger (BiLSTM recurrent weights)
- All share identical FeatureAttention (25,088 params) and Decoder (105,296 params)

**Not yet done (next session):**
- [ ] Run `train_compare.py --machines machine-1-1` as a smoke-test
- [ ] Fill paper tables with actual results after running experiments
- [ ] Consider free bits / normalizing flow prior to fix KL collapse
- [ ] Run on SMAP + MSL datasets (low effort once SMD pipeline works)

---

## Previous Sessions

### 2026-03-28 (session 4) — Run 3: F1=0, NaN bug diagnosed
- Training (global norm + MSE scoring + clamp -20, machine-1-1): NLL reached -0.84, KL collapsed to 0.028
- F1=0 because NaN in test scores from near-constant features (train_std ≈ 0 → Inf normalization)
- Root cause confirmed: need to clip train_std before normalization → **fixed in session 5**

### 2026-03-26 (session 3) — Three robustness fixes implemented
- Reverted x_log_var clamp to [-20, 2], added global normalization, switched to MSE scoring

### 2026-03-25 (session 2) — Two training runs, core issues diagnosed
- Run 1 (NLL scoring): PA-F1=0.76, ROC-AUC=0.59. Run 2 (clamp -4): PA-F1=0.74, ROC-AUC=0.60
- Diagnosis: (1) NLL scoring counterproductive (variance absorbs error), (2) clamp -4 crippled gradients, (3) per-window Z-score erases anomaly signals

---

## Proposed Changes for Next Session

**Priority: retrain with current changes, then tackle flow/prior improvements.**

### Step 1: Retrain with current changes (required)

All three changes from session 3 (global norm, MSE scoring, clamp -20) are implemented but untested. Run the notebook top-to-bottom and compare ROC-AUC / F1 against session 2 baselines (ROC-AUC 0.60, PA-F1 0.74).

### Step 2: Integrate flows.py into VAE

Wire `PlanarFlow` between `reparameterize()` and `decoder()`. Subtract `log_det.mean()` from KL loss. Include `flow.parameters()` in optimizer. This makes the posterior more expressive and may break KL collapse.

### Step 3: Add free bits to ELBOLoss

Set a minimum KL per latent dimension (e.g., λ=0.1 nats). Below this floor, KL is not penalized — prevents collapse while still regularizing. Simple change to `ELBOLoss.forward()`.

### Step 4 (exploratory): Better prior

Options: VampPrior (mixture of learned pseudo-inputs), GMM prior, or autoregressive prior (requires cross-window state — hard with current windowed architecture).

---

## Accuracy Improvement Roadmap

| # | Improvement | Status | Location |
|---|---|---|---|
| 1 | Feature-wise top-k scoring | Done | `vae_model.py` |
| 2 | Learned decoder variance (Gaussian NLL) | Done | `vae_model.py`, notebook |
| 3 | Temporal score smoothing | Done | notebook Section 6 |
| 4 | Combined NLL + KL scoring | Done | `vae_model.py` |
| 5 | Hyperparameter tuning presets | Done | notebook Section 2 |
| 6 | Training contamination filtering | Done | `preprocess.py`, notebook |
| 7 | Multi-scale ensemble | Done | `multiscale.py` |
| 8 | Normalizing flow prior | Done (not integrated) | `flows.py` |
| 9 | Global normalization | Done | `preprocess.py`, notebook |
| 10 | MSE scoring | Done | notebook Sections 5+6 |
| 11 | Clamp revert to [-20, 2] | Done | `vae_model.py` |

---

## Planned / Open Items

- [x] **Run the notebook** — two runs completed (session 2). Raw F1 ~0.28, PA-F1 ~0.74, ROC-AUC ~0.60.
- [x] **Switch to MSE scoring + global norm + clamp [-20,2]** — done (session 3)
- [x] **Fix NaN in global normalization** — clipped train_std to max(std, 1e-3) (session 5)
- [x] **Implement LSTM-VAE and Mamba-VAE** — done (session 5), verified end-to-end
- [x] **Create batch training script** — train_compare.py with RunLogger integration (session 5)
- [x] **Create results persistence** — results_logger.py + plot_results.py (session 5)
- [x] **Draft research paper** — paper/paper.tex + refs.bib (session 5)
- [x] **Integrate RunLogger into notebook** — main_demo.ipynb now saves structured results (session 5)
- [ ] **Run smoke-test** — `train_compare.py --machines machine-1-1` to verify end-to-end
- [ ] **Full SMD sweep** — all 3 models × 28 machines, fill paper tables
- [ ] **Add free bits to ELBOLoss** — prevent KL collapse (λ=0.1 nats per latent dim)
- [ ] **Integrate `flows.py`** — wire PlanarFlow into VAE forward pass
- [ ] **Run on SMAP + MSL** — extend beyond SMD for publication credibility
- [ ] **Long-window scaling** — W=128/256/512 to test Mamba's linear-time advantage
- [ ] **Statistical significance** — Wilcoxon signed-rank across 28 machines
