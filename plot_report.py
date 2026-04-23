"""Generate report-ready figures from the most recent run per (machine, arch)."""
import os
import numpy as np
from results_logger import collect_all_runs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
OUT_DIR = 'results/report_figures'
os.makedirs(OUT_DIR, exist_ok=True)

ARCH_COLORS = {'TCN-VAE': '#4C72B0', 'LSTM-VAE': '#DD8452', 'Mamba-VAE': '#55A868'}


def latest_runs(runs):
    """Keep only the most recent run per (machine, arch)."""
    best = {}
    for r in runs:
        key = (r.machine, r.arch)
        if key not in best or r.timestamp > best[key].timestamp:
            best[key] = r
    return sorted(best.values(), key=lambda r: (r.machine, r.arch))


def fig_comparison_bar(runs, path):
    machines = sorted({r.machine for r in runs})
    archs = sorted({r.arch for r in runs})
    matrix = {(r.machine, r.arch): r.metrics.get('pa_f1', np.nan) for r in runs}

    fig, ax = plt.subplots(figsize=(max(8, len(machines) * 0.9), 4.5))
    x = np.arange(len(machines))
    w = 0.8 / len(archs)
    for i, arch in enumerate(archs):
        vals = [matrix.get((m, arch), np.nan) for m in machines]
        off = (i - len(archs) / 2 + 0.5) * w
        ax.bar(x + off, vals, w * 0.9, label=arch,
               color=ARCH_COLORS.get(arch, None), alpha=0.85)
    ax.set_ylabel('Point-Adjusted F1')
    ax.set_title('PA-F1 Across Machines and Architectures')
    ax.set_xticks(x)
    ax.set_xticklabels(machines, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def fig_threshold_comparison(runs, path):
    """PA-F1 averaged across machines for each threshold strategy × arch."""
    strategies = ['threshold_pct99', 'threshold_pct995', 'threshold_pct999',
                  'threshold_pot_1e2', 'threshold_pot_1e3', 'threshold_pot_1e4']
    labels = ['p99', 'p99.5', 'p99.9', 'POT@1e-2', 'POT@1e-3', 'POT@1e-4']
    archs = sorted({r.arch for r in runs})

    data = {arch: {lab: [] for lab in labels} for arch in archs}
    for r in runs:
        sweep = r.threshold_sweep or {}
        for strat, lab in zip(strategies, labels):
            key = strat.replace('threshold_', '')
            v = sweep.get(key, {}).get('pa_f1') if isinstance(sweep.get(key), dict) else None
            if v is not None and not np.isnan(v):
                data[r.arch][lab].append(v)

    # Fallback: if threshold_sweep empty, skip
    if not any(any(vs) for d in data.values() for vs in d.values()):
        print(f'  skipped {path} (no threshold_sweep data)')
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    w = 0.8 / len(archs)
    for i, arch in enumerate(archs):
        means = [np.mean(data[arch][lab]) if data[arch][lab] else np.nan for lab in labels]
        off = (i - len(archs) / 2 + 0.5) * w
        ax.bar(x + off, means, w * 0.9, label=arch,
               color=ARCH_COLORS.get(arch, None), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Mean PA-F1 (across machines)')
    ax.set_title('Threshold Strategy Comparison')
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def fig_score_dist(run, path):
    if run.train_scores is None or run.test_scores is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    fig.suptitle(f'Score Distribution — {run.machine} / {run.arch}', fontsize=12)

    axes[0].hist(run.train_scores, bins=80, color='steelblue', alpha=0.75, density=True)
    if run.threshold is not None:
        axes[0].axvline(run.threshold, color='red', linestyle='--',
                        label=f'threshold = {run.threshold:.3f}')
    axes[0].set_title('Training (normal only)'); axes[0].set_xlabel('MSE')
    axes[0].legend(fontsize=8)

    if run.labels is not None:
        n = min(len(run.test_scores), len(run.labels))
        normal = run.test_scores[:n][run.labels[:n] == 0]
        anom = run.test_scores[:n][run.labels[:n] == 1]
        axes[1].hist(normal, bins=80, color='steelblue', alpha=0.6, density=True, label='normal')
        axes[1].hist(anom, bins=80, color='crimson', alpha=0.6, density=True, label='anomaly')
    if run.threshold is not None:
        axes[1].axvline(run.threshold, color='red', linestyle='--',
                        label=f'threshold = {run.threshold:.3f}')
    axes[1].set_title('Test (normal vs anomaly)'); axes[1].set_xlabel('MSE')
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def fig_timeline(run, path):
    if run.test_scores is None:
        return
    fig, ax = plt.subplots(figsize=(14, 3.5))
    n = len(run.test_scores)
    ax.plot(np.arange(n), run.test_scores, color='steelblue', linewidth=0.5, alpha=0.85,
            label='MSE')
    if run.threshold is not None:
        ax.axhline(run.threshold, color='red', linestyle='--', linewidth=1,
                   label=f'threshold = {run.threshold:.3f}')
    if run.labels is not None:
        labels = run.labels[:n]; in_a = False; s = 0
        for i in range(len(labels)):
            if labels[i] == 1 and not in_a:
                s = i; in_a = True
            elif labels[i] == 0 and in_a:
                ax.axvspan(s, i, color='red', alpha=0.15); in_a = False
        if in_a:
            ax.axvspan(s, len(labels), color='red', alpha=0.15)
    pa = run.metrics.get('pa_f1', float('nan'))
    ax.set_title(f'Anomaly Timeline — {run.machine} / {run.arch}  (PA-F1={pa:.3f})')
    ax.set_xlabel('Window Index'); ax.set_ylabel('MSE')
    ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def fig_loss_curves(runs_for_machine, machine, path):
    runs = [r for r in runs_for_machine if r.history]
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    fig.suptitle(f'Training Curves — {machine}', fontsize=12)
    for r in runs:
        epochs = range(1, len(r.history['total']) + 1)
        c = ARCH_COLORS.get(r.arch)
        axes[0].plot(epochs, r.history['total'], label=r.arch, color=c)
        axes[1].plot(epochs, r.history['recon'], label=r.arch, color=c)
        axes[2].plot(epochs, r.history['kl'], label=r.arch, color=c)
    axes[0].set_title('Total Loss (ELBO)')
    axes[1].set_title('Reconstruction Loss')
    axes[2].set_title('KL Divergence'); axes[2].set_yscale('log')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path}')


def summary_table(runs, path):
    machines = sorted({r.machine for r in runs})
    archs = sorted({r.arch for r in runs})
    lines = ['# Results Summary (most recent run per machine × arch)', '']
    lines.append('| Machine | ' + ' | '.join(f'{a} PA-F1' for a in archs) + ' |')
    lines.append('|---' * (len(archs) + 1) + '|')
    for m in machines:
        row = [m]
        for a in archs:
            match = [r for r in runs if r.machine == m and r.arch == a]
            v = match[0].metrics.get('pa_f1', float('nan')) if match else float('nan')
            row.append(f'{v:.4f}' if not np.isnan(v) else '---')
        lines.append('| ' + ' | '.join(row) + ' |')
    # Mean row
    lines.append('|---' * (len(archs) + 1) + '|')
    mean_row = ['**Mean**']
    for a in archs:
        vals = [r.metrics.get('pa_f1', float('nan')) for r in runs if r.arch == a]
        vals = [v for v in vals if not np.isnan(v)]
        mean_row.append(f'**{np.mean(vals):.4f}**' if vals else '---')
    lines.append('| ' + ' | '.join(mean_row) + ' |')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  saved {path}')


def main():
    all_runs = collect_all_runs(RESULTS_DIR)
    runs = latest_runs(all_runs)
    print(f'Filtered {len(all_runs)} -> {len(runs)} most-recent runs')
    for r in runs:
        print(f'  {r.machine:<14} {r.arch:<10} {r.timestamp}  PA-F1={r.metrics.get("pa_f1", float("nan")):.3f}')

    fig_comparison_bar(runs, os.path.join(OUT_DIR, 'comparison_bar.png'))
    fig_threshold_comparison(runs, os.path.join(OUT_DIR, 'threshold_comparison.png'))
    summary_table(runs, os.path.join(OUT_DIR, 'summary_table.md'))

    # Per-machine deep dives: score dist + timeline + loss curves
    for machine in sorted({r.machine for r in runs}):
        mruns = [r for r in runs if r.machine == machine]
        fig_loss_curves(mruns, machine, os.path.join(OUT_DIR, f'loss_curves_{machine}.png'))
        for r in mruns:
            fig_score_dist(r, os.path.join(OUT_DIR, f'score_dist_{machine}_{r.arch}.png'))
            fig_timeline(r, os.path.join(OUT_DIR, f'timeline_{machine}_{r.arch}.png'))

    print(f'\nDone. Figures in {OUT_DIR}/')


if __name__ == '__main__':
    main()
