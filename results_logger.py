"""
results_logger.py — Capture and persist all important outputs from a training run
----------------------------------------------------------------------------------
Drop-in logger for main_demo.ipynb and train_compare.py. Collects training
history, scores, thresholds, metrics, and hyperparameters, then writes them
to a structured directory for later analysis and paper table generation.

Usage in notebook (add one line per section):
    from results_logger import RunLogger
    logger = RunLogger('machine-1-1', 'TCN-VAE')

    # After Section 2 (hyperparameters):
    logger.log_hyperparameters(WINDOW_SIZE=64, LATENT_DIM=32, ...)

    # After Section 4 (training loop):
    logger.log_training(history, train_time_s=elapsed)

    # After Section 5 (threshold calibration):
    logger.log_train_scores(train_scores_np, THRESHOLD, THRESHOLD_POT, THRESHOLD_PCT99)

    # After Section 6 (inference + metrics):
    logger.log_test_results(
        test_scores_raw, test_scores_smoothed, predictions,
        aligned_labels, raw_f1, pa_f1_dict, roc_auc
    )

    # Save everything to disk:
    logger.save()          # → results/machine-1-1_TCN-VAE_20260328_143022/

Loading saved results:
    run = RunLogger.load('results/machine-1-1_TCN-VAE_20260328_143022')
    run.summary          # dict with all metadata + metrics
    run.history           # dict with 'total', 'recon', 'kl', 'beta' lists
    run.train_scores      # np.ndarray (N_train,)
    run.test_scores_raw   # np.ndarray (N_test,)
    run.test_scores       # np.ndarray (N_test,) — smoothed
    run.predictions       # np.ndarray (N_test,) — binary
    run.labels            # np.ndarray (N_test,) — aligned ground truth
"""

import os
import json
import csv
import numpy as np
from datetime import datetime
from typing import Optional, Union


class RunLogger:
    """Accumulates outputs from a single training run and saves to disk."""

    def __init__(
        self,
        machine:    str,
        arch:       str,
        output_dir: str = 'results',
    ):
        """
        Args:
            machine    : Machine name, e.g. 'machine-1-1'.
            arch       : Architecture name, e.g. 'TCN-VAE', 'LSTM-VAE', 'Mamba-VAE'.
            output_dir : Parent directory for all runs.
        """
        self.machine    = machine
        self.arch       = arch
        self.output_dir = output_dir
        self.timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Run directory: results/machine-1-1_TCN-VAE_20260328_143022/
        self.run_dir = os.path.join(
            output_dir,
            f'{machine}_{arch}_{self.timestamp}'
        )

        # Accumulated data (filled by log_* methods)
        self.hyperparameters = {}
        self.history         = None      # dict: total, recon, kl, beta
        self.train_time_s    = None
        self.train_scores      = None      # np.ndarray (N_train,)
        self.threshold         = None
        self.threshold_pot     = None
        self.threshold_pct99   = None
        self.threshold_pct995  = None
        self.threshold_pct999  = None
        self.threshold_pot_1e2 = None
        self.threshold_pot_1e3 = None
        self.threshold_pot_1e4 = None
        self.threshold_sweep   = None    # dict: per-strategy raw_f1/pa_f1
        self.train_score_stats = None    # dict: min, mean, std, max, p99

        self.test_scores_raw = None      # np.ndarray (N_test,)
        self.test_scores     = None      # np.ndarray (N_test,) — smoothed
        self.predictions     = None      # np.ndarray (N_test,) — binary
        self.labels          = None      # np.ndarray (N_test,) — aligned GT

        self.metrics         = {}        # f1, pa_f1, pa_precision, pa_recall, roc_auc, auprc
        self.test_score_stats = None

    # ── Logging methods (call these from the notebook) ────────────────────────

    def log_hyperparameters(self, **kwargs):
        """Log hyperparameters as keyword arguments.

        Example:
            logger.log_hyperparameters(
                WINDOW_SIZE=64, LATENT_DIM=32, TCN_HIDDEN=64,
                N_HEADS=4, EPOCHS=100, LR=1e-3, WARMUP_EPOCHS=25,
                MAX_BETA=1.0, BATCH_SIZE=256, MACHINE_NAME='machine-1-1',
            )
        """
        self.hyperparameters.update(kwargs)

    def log_training(self, history: dict, train_time_s: float = None):
        """Log training history and optional wall-clock time.

        Args:
            history      : Dict with keys 'total', 'recon', 'kl', 'beta',
                           each a list of per-epoch floats.
            train_time_s : Total training wall-clock time in seconds.
        """
        self.history      = {k: list(v) for k, v in history.items()}
        self.train_time_s = train_time_s

    def log_train_scores(
        self,
        train_scores:      np.ndarray,
        threshold:         float,
        threshold_pot:     float = None,
        threshold_pct99:   float = None,
        threshold_pct995:  float = None,
        threshold_pct999:  float = None,
        threshold_pot_1e2: float = None,
        threshold_pot_1e3: float = None,
        threshold_pot_1e4: float = None,
        threshold_sweep:   dict  = None,
    ):
        """Log training-set anomaly scores and calibrated thresholds."""
        self.train_scores      = np.asarray(train_scores, dtype=np.float32)
        self.threshold         = float(threshold)
        self.threshold_pot     = float(threshold_pot)     if threshold_pot     is not None else None
        self.threshold_pct99   = float(threshold_pct99)   if threshold_pct99   is not None else None
        self.threshold_pct995  = float(threshold_pct995)  if threshold_pct995  is not None else None
        self.threshold_pct999  = float(threshold_pct999)  if threshold_pct999  is not None else None
        self.threshold_pot_1e2 = float(threshold_pot_1e2) if threshold_pot_1e2 is not None else None
        self.threshold_pot_1e3 = float(threshold_pot_1e3) if threshold_pot_1e3 is not None else None
        self.threshold_pot_1e4 = float(threshold_pot_1e4) if threshold_pot_1e4 is not None else None
        self.threshold_sweep   = threshold_sweep or {}
        self.train_score_stats = {
            'min':  float(np.nanmin(train_scores)),
            'mean': float(np.nanmean(train_scores)),
            'std':  float(np.nanstd(train_scores)),
            'max':  float(np.nanmax(train_scores)),
            'p99':  float(np.nanpercentile(train_scores, 99)),
            'nan_count': int(np.isnan(train_scores).sum()),
        }

    def log_test_results(
        self,
        test_scores_raw: np.ndarray,
        test_scores:     np.ndarray,
        predictions:     np.ndarray,
        aligned_labels:  np.ndarray,
        raw_f1:          float,
        pa_f1_result:    Union[dict, float],
        roc_auc:         float = None,
        auprc:           float = None,
    ):
        """Log test-set scores, predictions, labels, and computed metrics."""
        self.test_scores_raw = np.asarray(test_scores_raw, dtype=np.float32)
        self.test_scores     = np.asarray(test_scores, dtype=np.float32)
        self.predictions     = np.asarray(predictions, dtype=np.int32)
        self.labels          = np.asarray(aligned_labels, dtype=np.int32)

        # Handle pa_f1_result being either a dict or a float
        if isinstance(pa_f1_result, dict):
            self.metrics['pa_f1']       = float(pa_f1_result.get('f1', 0))
            self.metrics['pa_precision'] = float(pa_f1_result.get('precision', 0))
            self.metrics['pa_recall']    = float(pa_f1_result.get('recall', 0))
        else:
            self.metrics['pa_f1'] = float(pa_f1_result)

        self.metrics['raw_f1']  = float(raw_f1)
        self.metrics['roc_auc'] = float(roc_auc) if roc_auc is not None else None
        self.metrics['auprc']   = float(auprc)   if auprc   is not None else None

        n = len(predictions)
        self.metrics['n_windows']          = n
        self.metrics['n_anomalous']        = int(predictions.sum())
        self.metrics['anomaly_rate_pct']   = float(100 * predictions.mean()) if n > 0 else 0
        self.metrics['gt_anomaly_rate_pct'] = float(100 * aligned_labels.mean()) if n > 0 else 0

        self.test_score_stats = {
            'min':  float(np.nanmin(test_scores)),
            'mean': float(np.nanmean(test_scores)),
            'std':  float(np.nanstd(test_scores)),
            'max':  float(np.nanmax(test_scores)),
            'nan_count': int(np.isnan(test_scores).sum()),
        }

    # ── Save to disk ──────────────────────────────────────────────────────────

    def save(self) -> str:
        """Write all logged data to self.run_dir. Returns the run directory path."""
        os.makedirs(self.run_dir, exist_ok=True)

        # 1. Summary JSON — all metadata, hyperparams, metrics, stats
        summary = {
            'machine':          self.machine,
            'arch':             self.arch,
            'timestamp':        self.timestamp,
            'hyperparameters':  self.hyperparameters,
            'train_time_s':     self.train_time_s,
            'thresholds': {
                'active':    self.threshold,
                'pot':       self.threshold_pot,
                'pct99':     self.threshold_pct99,
                'pct99.5':   self.threshold_pct995,
                'pct99.9':   self.threshold_pct999,
                'pot_1e-2':  self.threshold_pot_1e2,
                'pot_1e-3':  self.threshold_pot_1e3,
                'pot_1e-4':  self.threshold_pot_1e4,
            },
            'threshold_sweep': self.threshold_sweep,
            'metrics':            self.metrics,
            'train_score_stats':  self.train_score_stats,
            'test_score_stats':   self.test_score_stats,
        }
        with open(os.path.join(self.run_dir, 'run_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=_json_default)
        self.summary = summary

        # 2. Training history CSV — epoch, total, recon, kl, beta
        if self.history is not None:
            csv_path = os.path.join(self.run_dir, 'training_history.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'total_loss', 'recon_loss', 'kl_loss', 'beta'])
                n_epochs = len(self.history.get('total', []))
                for i in range(n_epochs):
                    writer.writerow([
                        i + 1,
                        self.history['total'][i],
                        self.history['recon'][i],
                        self.history['kl'][i],
                        self.history['beta'][i] if i < len(self.history.get('beta', [])) else '',
                    ])

        # 3. NumPy arrays — scores, predictions, labels
        if self.train_scores is not None:
            np.save(os.path.join(self.run_dir, 'train_scores.npy'), self.train_scores)

        if self.test_scores_raw is not None:
            np.save(os.path.join(self.run_dir, 'test_scores_raw.npy'), self.test_scores_raw)

        if self.test_scores is not None:
            np.save(os.path.join(self.run_dir, 'test_scores.npy'), self.test_scores)

        if self.predictions is not None:
            np.save(os.path.join(self.run_dir, 'predictions.npy'), self.predictions)

        if self.labels is not None:
            np.save(os.path.join(self.run_dir, 'aligned_labels.npy'), self.labels)

        print(f'[RunLogger] Saved to: {self.run_dir}/')
        _print_file_list(self.run_dir)
        return self.run_dir

    # ── Load from disk ────────────────────────────────────────────────────────

    @classmethod
    def load(cls, run_dir: str) -> 'RunLogger':
        """Reconstruct a RunLogger from a saved run directory.

        Args:
            run_dir : Path to a previously saved run directory.
        Returns:
            RunLogger with all fields populated from disk.
        """
        with open(os.path.join(run_dir, 'run_summary.json')) as f:
            summary = json.load(f)

        logger = cls(
            machine    = summary['machine'],
            arch       = summary['arch'],
            output_dir = os.path.dirname(run_dir),
        )
        logger.run_dir         = run_dir
        logger.timestamp       = summary.get('timestamp', '')
        logger.hyperparameters = summary.get('hyperparameters', {})
        logger.train_time_s    = summary.get('train_time_s')
        logger.metrics         = summary.get('metrics', {})
        logger.train_score_stats = summary.get('train_score_stats')
        logger.test_score_stats  = summary.get('test_score_stats')
        logger.summary         = summary

        thresholds = summary.get('thresholds', {})
        logger.threshold         = thresholds.get('active')
        logger.threshold_pot     = thresholds.get('pot')
        logger.threshold_pct99   = thresholds.get('pct99')
        logger.threshold_pct995  = thresholds.get('pct99.5')
        logger.threshold_pct999  = thresholds.get('pct99.9')
        logger.threshold_pot_1e2 = thresholds.get('pot_1e-2')
        logger.threshold_pot_1e3 = thresholds.get('pot_1e-3')
        logger.threshold_pot_1e4 = thresholds.get('pot_1e-4')
        logger.threshold_sweep   = summary.get('threshold_sweep', {})

        # Load training history from CSV
        csv_path = os.path.join(run_dir, 'training_history.csv')
        if os.path.exists(csv_path):
            logger.history = {'total': [], 'recon': [], 'kl': [], 'beta': []}
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    logger.history['total'].append(float(row['total_loss']))
                    logger.history['recon'].append(float(row['recon_loss']))
                    logger.history['kl'].append(float(row['kl_loss']))
                    beta = row.get('beta', '')
                    logger.history['beta'].append(float(beta) if beta else 0.0)

        # Load numpy arrays
        for attr, fname in [
            ('train_scores',    'train_scores.npy'),
            ('test_scores_raw', 'test_scores_raw.npy'),
            ('test_scores',     'test_scores.npy'),
            ('predictions',     'predictions.npy'),
            ('labels',          'aligned_labels.npy'),
        ]:
            fpath = os.path.join(run_dir, fname)
            if os.path.exists(fpath):
                setattr(logger, attr, np.load(fpath))

        return logger

    def __repr__(self):
        pa = self.metrics.get('pa_f1', '?')
        f1 = self.metrics.get('raw_f1', '?')
        return f'RunLogger({self.machine}, {self.arch}, PA-F1={pa}, F1={f1})'


# ── Utilities ─────────────────────────────────────────────────────────────────

def _json_default(obj):
    """Handle numpy types in json.dump."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _print_file_list(directory: str):
    """Print files in a directory with sizes."""
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        size  = os.path.getsize(fpath)
        if size > 1024 * 1024:
            size_str = f'{size / 1024 / 1024:.1f} MB'
        elif size > 1024:
            size_str = f'{size / 1024:.1f} KB'
        else:
            size_str = f'{size} B'
        print(f'  {fname:<30s} {size_str:>10s}')


def collect_all_runs(results_dir: str = 'results') -> list:
    """Scan results_dir for saved runs and return a list of RunLogger objects.

    Useful for loading multiple runs for comparison plotting:
        runs = collect_all_runs()
        for r in runs:
            print(r.machine, r.arch, r.metrics.get('pa_f1'))
    """
    runs = []
    if not os.path.isdir(results_dir):
        return runs
    for entry in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, entry)
        summary_path = os.path.join(run_dir, 'run_summary.json')
        if os.path.isdir(run_dir) and os.path.exists(summary_path):
            try:
                runs.append(RunLogger.load(run_dir))
            except Exception as e:
                print(f'Warning: could not load {run_dir}: {e}')
    return runs
