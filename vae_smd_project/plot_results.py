"""
plot_results.py — Generate comparison plots from saved RunLogger outputs
-------------------------------------------------------------------------
Reads results/ directory and produces publication-quality figures.

Usage:
    python plot_results.py                          # all runs in results/
    python plot_results.py --run_dir results/machine-1-1_TCN-VAE_*   # single run
    python plot_results.py --machine machine-1-1    # all archs for one machine

Outputs saved to results/figures/:
    loss_curves_{machine}.png       — training loss curves (all archs overlaid)
    kl_curves_{machine}.png         — KL divergence over epochs
    score_dist_{machine}.png        — train/test score distributions
    anomaly_timeline_{machine}.png  — test scores vs labels with threshold
    comparison_bar.png              — PA-F1 bar chart across machines/archs
"""

import os
import sys
import argparse
import numpy as np

from results_logger import RunLogger, collect_all_runs


def _import_matplotlib():
    """Import matplotlib with Agg backend for headless environments."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_loss_curves(runs: list, output_dir: str):
    """Training loss curves: total, recon, KL — one plot per machine, archs overlaid."""
    plt = _import_matplotlib()

    machines = sorted(set(r.machine for r in runs))
    for machine in machines:
        machine_runs = [r for r in runs if r.machine == machine and r.history]
        if not machine_runs:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(f'Training Curves — {machine}', fontsize=13)

        for run in machine_runs:
            epochs = range(1, len(run.history['total']) + 1)
            axes[0].plot(epochs, run.history['total'], label=run.arch)
            axes[1].plot(epochs, run.history['recon'], label=run.arch)
            axes[2].plot(epochs, run.history['kl'],    label=run.arch)

        axes[0].set_title('Total Loss (ELBO)')
        axes[1].set_title('Reconstruction Loss (NLL)')
        axes[2].set_title('KL Divergence')

        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(output_dir, f'loss_curves_{machine}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')


def plot_kl_curves(runs: list, output_dir: str):
    """KL divergence over epochs — one plot per machine, archs overlaid."""
    plt = _import_matplotlib()

    machines = sorted(set(r.machine for r in runs))
    for machine in machines:
        machine_runs = [r for r in runs if r.machine == machine and r.history]
        if not machine_runs:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f'KL Divergence — {machine}')

        for run in machine_runs:
            epochs = range(1, len(run.history['kl']) + 1)
            ax.plot(epochs, run.history['kl'], label=run.arch, linewidth=1.5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        fig.tight_layout()
        path = os.path.join(output_dir, f'kl_curves_{machine}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')


def plot_score_distributions(runs: list, output_dir: str):
    """Train and test score histograms with thresholds marked."""
    plt = _import_matplotlib()

    for run in runs:
        if run.train_scores is None or run.test_scores is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(f'Score Distribution — {run.machine} / {run.arch}', fontsize=12)

        # Train scores
        axes[0].hist(run.train_scores, bins=80, color='steelblue', alpha=0.7, density=True)
        if run.threshold is not None:
            axes[0].axvline(run.threshold, color='red', linestyle='--',
                            label=f'Threshold = {run.threshold:.4f}')
        axes[0].set_title('Training Scores (MSE)')
        axes[0].set_xlabel('MSE')
        axes[0].legend(fontsize=8)

        # Test scores
        if run.labels is not None:
            n = min(len(run.test_scores), len(run.labels))
            normal  = run.test_scores[:n][run.labels[:n] == 0]
            anomaly = run.test_scores[:n][run.labels[:n] == 1]
            axes[1].hist(normal,  bins=80, color='steelblue', alpha=0.6, density=True, label='Normal')
            axes[1].hist(anomaly, bins=80, color='crimson',   alpha=0.6, density=True, label='Anomaly')
        else:
            axes[1].hist(run.test_scores, bins=80, color='steelblue', alpha=0.7, density=True)
        if run.threshold is not None:
            axes[1].axvline(run.threshold, color='red', linestyle='--',
                            label=f'Threshold = {run.threshold:.4f}')
        axes[1].set_title('Test Scores (MSE, smoothed)')
        axes[1].set_xlabel('MSE')
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        path = os.path.join(output_dir, f'score_dist_{run.machine}_{run.arch}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')


def plot_anomaly_timeline(runs: list, output_dir: str):
    """Test scores over time with ground truth shading and threshold line."""
    plt = _import_matplotlib()

    for run in runs:
        if run.test_scores is None:
            continue

        fig, ax = plt.subplots(figsize=(16, 4))
        n = len(run.test_scores)
        timesteps = np.arange(n)

        ax.plot(timesteps, run.test_scores, color='steelblue', linewidth=0.5, alpha=0.8,
                label='MSE Score')

        if run.threshold is not None:
            ax.axhline(run.threshold, color='red', linestyle='--', linewidth=1,
                       label=f'Threshold = {run.threshold:.4f}')

        # Shade ground truth anomaly regions
        if run.labels is not None:
            labels = run.labels[:n]
            in_anomaly = False
            start = 0
            for i in range(len(labels)):
                if labels[i] == 1 and not in_anomaly:
                    start = i
                    in_anomaly = True
                elif labels[i] == 0 and in_anomaly:
                    ax.axvspan(start, i, color='red', alpha=0.15)
                    in_anomaly = False
            if in_anomaly:
                ax.axvspan(start, len(labels), color='red', alpha=0.15)

        ax.set_title(f'Anomaly Detection — {run.machine} / {run.arch}  '
                     f'(PA-F1={run.metrics.get("pa_f1", "?"):.4f})', fontsize=11)
        ax.set_xlabel('Window Index')
        ax.set_ylabel('MSE Score')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        path = os.path.join(output_dir, f'anomaly_timeline_{run.machine}_{run.arch}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {path}')


def plot_comparison_bar(runs: list, output_dir: str):
    """Grouped bar chart: PA-F1 per machine × architecture."""
    plt = _import_matplotlib()

    # Group by machine, then by arch
    machines = sorted(set(r.machine for r in runs))
    archs    = sorted(set(r.arch for r in runs))

    if len(machines) == 0 or len(archs) == 0:
        return

    # Build matrix: machines × archs
    matrix = {}
    for r in runs:
        matrix[(r.machine, r.arch)] = r.metrics.get('pa_f1', float('nan'))

    fig, ax = plt.subplots(figsize=(max(8, len(machines) * 1.2), 5))

    x = np.arange(len(machines))
    width = 0.8 / len(archs)
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

    for i, arch in enumerate(archs):
        values = [matrix.get((m, arch), float('nan')) for m in machines]
        offset = (i - len(archs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9, label=arch,
                       color=colors[i % len(colors)], alpha=0.85)
        # Value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=6)

    ax.set_xlabel('Machine')
    ax.set_ylabel('Point-Adjusted F1')
    ax.set_title('PA-F1 Comparison Across Architectures')
    ax.set_xticks(x)
    ax.set_xticklabels(machines, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'comparison_bar.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_comparison_table(runs: list, output_dir: str):
    """Write a markdown summary table to results/figures/summary_table.md."""
    machines = sorted(set(r.machine for r in runs))
    archs    = sorted(set(r.arch for r in runs))

    lines = []
    header = '| Machine |' + '|'.join(f' {a} PA-F1 ' for a in archs) + '|'
    sep    = '|---------|' + '|'.join('---------' for _ in archs) + '|'
    lines.append(header)
    lines.append(sep)

    for m in machines:
        row = f'| {m} |'
        for a in archs:
            match = [r for r in runs if r.machine == m and r.arch == a]
            if match:
                val = match[0].metrics.get('pa_f1', float('nan'))
                row += f' {val:.4f} |' if not np.isnan(val) else ' --- |'
            else:
                row += ' --- |'
        lines.append(row)

    # Averages
    lines.append(sep)
    row = '| **Mean** |'
    for a in archs:
        vals = [r.metrics.get('pa_f1', float('nan')) for r in runs if r.arch == a]
        vals = [v for v in vals if not np.isnan(v)]
        row += f' **{np.mean(vals):.4f}** |' if vals else ' --- |'
    lines.append(row)

    table_str = '\n'.join(lines)
    path = os.path.join(output_dir, 'summary_table.md')
    with open(path, 'w') as f:
        f.write(f'# Results Summary\n\nGenerated: {np.datetime64("today")}\n\n')
        f.write(table_str + '\n')
    print(f'  Saved: {path}')


def main():
    parser = argparse.ArgumentParser(description='Plot results from saved RunLogger outputs')
    parser.add_argument('--results_dir', default='results',
                        help='Parent directory containing run directories')
    parser.add_argument('--run_dir', default=None,
                        help='Single run directory to plot (overrides --results_dir)')
    parser.add_argument('--machine', default=None,
                        help='Filter to a single machine name')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for figures (default: results/figures)')
    args = parser.parse_args()

    # Collect runs
    if args.run_dir:
        runs = [RunLogger.load(args.run_dir)]
    else:
        runs = collect_all_runs(args.results_dir)

    if args.machine:
        runs = [r for r in runs if r.machine == args.machine]

    if not runs:
        print('No runs found. Run train_compare.py first, or check --results_dir.')
        sys.exit(1)

    print(f'Found {len(runs)} run(s):')
    for r in runs:
        print(f'  {r}')

    # Output directory
    fig_dir = args.output_dir or os.path.join(args.results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print(f'\nGenerating plots -> {fig_dir}/')
    plot_loss_curves(runs, fig_dir)
    plot_kl_curves(runs, fig_dir)
    plot_score_distributions(runs, fig_dir)
    plot_anomaly_timeline(runs, fig_dir)
    plot_comparison_bar(runs, fig_dir)
    plot_comparison_table(runs, fig_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
