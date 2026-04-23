"""
evaluation.py
-------------
Evaluation utilities for VAE-based anomaly detection.

Functions:
  pot_threshold     : POT (Peaks Over Threshold) threshold via Generalized Pareto fit.
  point_adjusted_f1 : Point-adjusted F1/precision/recall (OmniAnomaly protocol).
"""

import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import f1_score, precision_score, recall_score


def pot_threshold(
    train_scores: np.ndarray,
    q:            float = 0.80,
    level:        float = 1e-4,
) -> float:
    """
    Peaks Over Threshold (POT) anomaly threshold using a Generalized Pareto Distribution.

    Algorithm:
        1. Set initial threshold t0 = q-th percentile of training scores.
        2. Extract exceedances: peaks = scores[scores > t0] - t0.
        3. Fit a GPD to the peaks: F(y; γ, σ) = 1 − (1 + γy/σ)^(−1/γ).
        4. Solve for T such that P(score > T) = level (desired false alarm rate).

    This is more statistically principled than a fixed percentile: it models the
    tail of the error distribution using extreme value theory, so the threshold
    adapts to the heaviness of the tail rather than a fixed quantile.

    Args:
        train_scores : 1D numpy array of training anomaly scores (normal data only).
        q            : Percentile (0–1) for the initial tail cut-off t0. Default 0.80.
        level        : Desired false alarm probability on training data. Default 1e-4
                       (0.01% of training windows flagged). Tune this to trade off
                       precision vs recall: lower → fewer false alarms, higher threshold.
    Returns:
        threshold : Float — scores above this value are flagged as anomalous.

    Fallback:
        If fewer than 10 exceedances exist (sparse tail), returns the 99th percentile.
    """
    t0    = float(np.quantile(train_scores, q))
    peaks = train_scores[train_scores > t0] - t0

    if len(peaks) < 10:
        return float(np.quantile(train_scores, 0.99))

    # Fit GPD with loc fixed at 0 → returns (shape=γ, loc=0, scale=σ)
    gamma, _, sigma = genpareto.fit(peaks, floc=0)

    n       = len(train_scores)
    n_peaks = len(peaks)

    # Solve: (n_peaks / n) * (1 + γ(T - t0) / σ)^(−1/γ) = level  →  T = ?
    if abs(gamma) < 1e-8:
        # Limiting case γ→0: GPD degenerates to Exponential
        # T = t0 − σ · ln(level · n / n_peaks)
        threshold = t0 - sigma * np.log(level * n / n_peaks)
    else:
        threshold = t0 + (sigma / gamma) * ((level * n / n_peaks) ** (-gamma) - 1)

    return float(threshold)


def point_adjusted_f1(
    predictions: np.ndarray,
    labels:      np.ndarray,
) -> dict:
    """
    Point-adjusted precision, recall, and F1 (OmniAnomaly / MTAD-GAT protocol).

    Standard evaluation protocol for multivariate time-series anomaly detection:
        For each contiguous ground-truth anomaly segment, if ANY timestep in that
        segment is predicted anomalous, mark ALL timesteps in the segment as TP.

    Rationale:
        Detecting even one timestep of an anomaly event is operationally useful.
        Delayed or partial detection within an event should not be penalised as FN,
        since the alert was still raised. This protocol matches how OmniAnomaly,
        MTAD-GAT, and most published SMD benchmarks report their numbers.

    IMPORTANT: Point-adjusted F1 is always higher than raw F1. Report both for
    transparency — raw F1 for strict evaluation, point-adjusted for fair comparison
    with published results.

    Args:
        predictions : Binary 1D array of model predictions (0/1), shape (N,).
        labels      : Binary 1D array of ground truth labels (0/1), shape (N,).
    Returns:
        dict with keys:
            'f1'                   : point-adjusted F1 score
            'precision'            : point-adjusted precision
            'recall'               : point-adjusted recall
            'adjusted_predictions' : adjusted binary prediction array, shape (N,)
    """
    adjusted = predictions.copy().astype(int)
    labels   = labels.astype(int)

    i = 0
    while i < len(labels):
        if labels[i] == 1:
            # Find the end of this contiguous anomaly segment
            j = i
            while j < len(labels) and labels[j] == 1:
                j += 1
            # If any prediction in [i, j) is positive, mark the whole segment TP
            if adjusted[i:j].any():
                adjusted[i:j] = 1
            i = j
        else:
            i += 1

    return {
        'f1':                   f1_score(labels, adjusted, zero_division=0),
        'precision':            precision_score(labels, adjusted, zero_division=0),
        'recall':               recall_score(labels, adjusted, zero_division=0),
        'adjusted_predictions': adjusted,
    }
