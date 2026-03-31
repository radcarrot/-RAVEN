# RAVEN

**Reconstruction-based Anomaly detection with Variational ENcoders**

Unsupervised time-series anomaly detection using three interchangeable VAE architectures evaluated on the Server Machine Dataset (SMD). Models train on normal server telemetry only — anomalous windows are detected via abnormally high reconstruction error at inference time.

---

## Architecture

Three encoder backbones, all sharing the same FeatureAttention module and Decoder:

```
Input (B, F=38, W=64)
        │
        ▼
FeatureAttention        ← sensor-wise self-attention (shared across all three)
        │
   ┌────┴────────────────────┐
   │                         │                        │
TCN Encoder          BiLSTM Encoder          Mamba-2 Encoder
4× TemporalBlock     2-layer BiLSTM          4× MambaBlock
(dilations 1,2,4,8)  (hidden=128/dir)        (d_model=64, d_state=16)
~264K params         ~714K params            ~268K params
   │                         │                        │
   └────────────┬────────────┘
                │
         Latent space z (dim=32)
                │
            Decoder             ← shared ConvTranspose1d + learned variance head
                │
         (x_mu, x_log_var)
```

**Scoring:** per-window MSE between input and reconstruction mean. Threshold at 99th percentile of training scores.

**Evaluation:** OmniAnomaly point-adjusted F1 protocol.

---

## Installation

```bash
pip install -r requirements.txt
```

For CUDA support (NVIDIA GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

The Mamba encoder uses a pure-PyTorch selective scan — no `mamba-ssm` CUDA extension required.

---

## Data Setup

1. Clone [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)
2. Place its `ServerMachineDataset/` folder at `./data/ServerMachineDataset/`

The dataset has 28 machines: `machine-1-1` through `machine-1-8`, `machine-2-1` through `machine-2-9`, `machine-3-1` through `machine-3-11`. Each machine has 38 sensor channels.

Data and model checkpoints are not tracked in this repository.

---

## Usage

### Interactive (single model)

Open `main_demo.ipynb` in Jupyter and run cell-by-cell. Set `MACHINE_NAME` in Section 2.

```python
MACHINE_NAME = "machine-1-1"   # pick any of the 28 machines
WINDOW_SIZE  = 64
LATENT_DIM   = 32
EPOCHS       = 100
```

Outputs saved to `results/{machine}_{arch}_{timestamp}/`.

### Batch comparison (all 3 models × 28 machines)

```bash
python train_compare.py                           # full sweep
python train_compare.py --machines machine-1-1    # smoke-test on one machine
python train_compare.py --archs TCN LSTM          # subset of architectures
```

### Visualize results

```bash
python plot_results.py
```

Generates figures to `results/figures/`: loss curves, score distributions, anomaly timelines, cross-architecture comparison bar chart, and a markdown summary table.

---

## Project Structure

```
raven/
├── vae_model.py          # TCN-VAE v3 (TCN + FeatureAttention + learned variance)
├── lstm_model.py         # BiLSTM-VAE
├── mamba_model.py        # Mamba-2 SSD VAE (pure-PyTorch selective scan)
├── preprocess.py         # SMDWindowDataset, data loaders, contamination filter
├── evaluation.py         # POT threshold, point-adjusted F1
├── train_compare.py      # batch training script (all models × all machines)
├── results_logger.py     # RunLogger: JSON + CSV + .npy output per run
├── plot_results.py       # figures and comparison tables from saved runs
├── multiscale.py         # multi-scale ensemble (standalone, window sizes 32/64/128)
├── flows.py              # PlanarFlow normalizing flow prior (standalone)
├── main_demo.ipynb       # interactive end-to-end notebook
├── train_compare.ipynb   # batch training notebook
├── requirements.txt
├── RESEARCH_PLAN.md      # research questions, venue targets, experimental phases
└── paper/
    ├── paper.tex         # LaTeX manuscript
    └── refs.bib          # BibTeX references
```

---

## Key Design Decisions

- **Global normalization** — training set mean/std applied across all windows. Preserves level-shift anomalies that per-window Z-score normalization erases.
- **MSE scoring over NLL** — the learned variance head can absorb reconstruction error and mask anomalies; MSE ignores it and scores directly on reconstruction fidelity.
- **Parameter matching** — TCN (~264K) and Mamba (~268K) are parameter-matched for controlled comparison. LSTM is larger (~714K) due to BiLSTM recurrent weights.
- **Shared components** — all three architectures use identical FeatureAttention (25K params) and Decoder (105K params), isolating the encoder as the only variable.

---

## Results

Experiments in progress. Results will be reported as PA-F1 (point-adjusted F1, OmniAnomaly protocol) across all 28 SMD machines.

---

## Citation

Paper in preparation.
