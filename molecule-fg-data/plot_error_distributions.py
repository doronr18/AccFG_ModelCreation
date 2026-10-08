"""Plot every sampled error for every model, coloured by FG cluster.

Reads error_distributions.py's molecule_model_error_table.csv and writes to --out-dir:

  error_points.png              one panel per model, all on the same axes: x = molecules
                                ordered by FG cluster then pattern (the same order in every
                                panel), y = sampled error, one dot per molecule
  error_points_<model>.png      the same panel alone, on the same axes
  error_histograms.png          one histogram per model, same bins and axes, stacked by cluster
  error_plot_summary.csv        the numbers behind the figures: n, mean, std, min, max per
                                model, cluster and domain

A molecule's group is its pattern's primary cluster (pattern_clusters.csv). The largest
--top-groups clusters (by molecule count) get their own colour; the rest are grey "Other".
Clusters sit side by side along x, so each colour is also its own x band, labelled under
the axis. Dashed lines mark each model's in-domain and out-of-domain mean.

    python3 molecule-fg-data/plot_error_distributions.py
    python3 molecule-fg-data/plot_error_distributions.py --max-points 500000
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = DEFAULT_ROOT / 'molecule-fg-data'
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / 'csv_outputs'

# Categorical slots in fixed order (CVD-checked as neighbours, which is how the bands sit).
GROUP_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']
OTHER_COLOR = '#c3c2b7'
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'


def load_errors(errors_path: str | Path, chunksize: int):
    """Per model (in file order): pattern_index, in-domain flag and sampled error arrays.

    Also returns {(model, domain): error_mean}. Only these columns are kept, so the table
    is read in chunks and never held as a DataFrame.
    """
    columns = ['pattern_index', 'model_name', 'domain_flag', 'error_mean', 'sampled_error']
    parts: dict[str, dict[str, list]] = {}
    means: dict[tuple, float] = {}
    reader = pd.read_csv(errors_path, usecols=columns, dtype={'model_name': str, 'domain_flag': str},
                         chunksize=chunksize)
    for chunk in reader:
        for model_name, group in chunk.groupby('model_name', sort=False):
            p = parts.setdefault(model_name, {'pattern_index': [], 'in_domain': [], 'error': []})
            p['pattern_index'].append(group['pattern_index'].to_numpy(dtype=np.int64))
            p['in_domain'].append(group['domain_flag'].to_numpy() == 'in')
            p['error'].append(group['sampled_error'].to_numpy(dtype=np.float32))
            for domain, rows in group.groupby('domain_flag'):
                means.setdefault((model_name, domain), float(rows['error_mean'].iloc[0]))
    if not parts:
        raise ValueError(f'{errors_path} has no rows')
    models = {name: {k: np.concatenate(v) for k, v in p.items()} for name, p in parts.items()}
    sizes = {name: len(m['error']) for name, m in models.items()}
    if len(set(sizes.values())) != 1:
        print(f'WARNING: models have different molecule counts {sizes}; x positions will not line up')
    return models, means


def load_groups(pattern_clusters_path: str | Path, cluster_summary_path: str | Path | None):
    """pattern_index -> primary cluster, and cluster -> short FG label."""
    clusters = pd.read_csv(pattern_clusters_path)
    primary = 'cluster_primary' if 'cluster_primary' in clusters.columns else 'cluster_id'
    if not {'pattern_index', primary}.issubset(clusters.columns):
        raise ValueError(f'{pattern_clusters_path} must include pattern_index and {primary}')
    first = clusters.drop_duplicates('pattern_index')
    cluster_of_pattern = pd.Series(first[primary].astype(np.int64).to_numpy(),
                                   index=first['pattern_index'].astype(np.int64).to_numpy())

    labels = {}
    if cluster_summary_path is not None and Path(cluster_summary_path).exists():
        summary = pd.read_csv(cluster_summary_path)
        fg_column = next((c for c in ('common_fgs', 'centroid_fgs') if c in summary.columns), None)
        if fg_column is not None:
            for cid, fgs in zip(summary['cluster_id'].astype(np.int64), summary[fg_column]):
                names = [] if pd.isna(fgs) else [f for f in str(fgs).split(',') if f]
                shown = ', '.join(names[:3]) + (f' +{len(names) - 3}' if len(names) > 3 else '')
                labels[int(cid)] = shown or 'no common FG'
    return cluster_of_pattern, labels


def assign_groups(pattern_index: np.ndarray, cluster_of_pattern: pd.Series, top_groups: int):
    """Cluster per molecule, and the clusters that get a colour (largest first)."""
    cluster = cluster_of_pattern.reindex(pattern_index).fillna(-1).to_numpy(dtype=np.int64)
    ids, counts = np.unique(cluster, return_counts=True)
    order = np.lexsort((ids, -counts))
    top = [int(i) for i in ids[order] if i >= 0][:top_groups]
    return cluster, top


def x_positions(pattern_index: np.ndarray, cluster: np.ndarray, top: list[int]):
    """Molecule x positions: coloured clusters (largest first), then Other; within, by pattern.

    Returns (x, rank) with rank = colour slot, len(top) for Other. Every model holds the same
    molecules, so each pattern gets the same x range in every panel.
    """
    rank = np.full(len(cluster), len(top), dtype=np.int64)
    for slot, cid in enumerate(top):
        rank[cluster == cid] = slot
    order = np.lexsort((pattern_index, cluster, rank))
    x = np.empty(len(order), dtype=np.int64)
    x[order] = np.arange(len(order))
    return x, rank


def _slug(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_') or 'model'


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _mean_lines(ax, model_name, means, vertical=False):
    for domain, label in (('in', 'in-domain mean'), ('out', 'out-of-domain mean')):
        value = means.get((model_name, domain))
        if value is None:
            continue
        style = dict(color=INK_SECONDARY, linewidth=0.9, linestyle='--' if domain == 'in' else ':')
        (ax.axvline if vertical else ax.axhline)(value, **style)


def _legend_handles(top, labels, group_counts, n_other):
    handles = []
    for slot, cid in enumerate(top):
        text = f'Cluster {cid}: {labels.get(cid, "")}'.rstrip(': ') + f'  (n={group_counts[slot]:,})'
        handles.append(Line2D([], [], linestyle='', marker='o', markersize=7,
                              markerfacecolor=GROUP_COLORS[slot], markeredgecolor=SURFACE, label=text))
    if n_other:
        handles.append(Line2D([], [], linestyle='', marker='o', markersize=7, markerfacecolor=OTHER_COLOR,
                              markeredgecolor=SURFACE, label=f'Other clusters  (n={n_other:,})'))
    handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=0.9, linestyle='--', label='in-domain mean'))
    handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=0.9, linestyle=':', label='out-of-domain mean'))
    return handles


def plot_points(ax, model_name, data, means, top, band_ticks, xlim, ylim, point_size):
    _style_axes(ax)
    colors = np.array(GROUP_COLORS[:len(top)] + [OTHER_COLOR])
    # Other first so the coloured clusters draw on top where bands touch.
    draw = np.argsort(data['rank'] != len(top), kind='stable')
    ax.scatter(data['x'][draw], data['error'][draw], c=colors[data['rank'][draw]], s=point_size,
               linewidths=0, rasterized=True)
    _mean_lines(ax, model_name, means)
    n_in = int(data['in_domain'].sum())
    ax.set_title(f'{model_name}   in-domain {n_in:,} · out-of-domain {len(data["error"]) - n_in:,}',
                 loc='left', fontsize=10, color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_ylabel('sampled error', fontsize=9, color=INK_SECONDARY)
    centers, names, edges = band_ticks
    ax.set_xticks(centers)
    ax.set_xticklabels(names, fontsize=8)
    for edge in edges:
        ax.axvline(edge, color=GRID, linewidth=0.6)


def build_figures(
    errors_path: str | Path = DEFAULT_OUTPUT_DIR / 'molecule_model_error_table.csv',
    pattern_clusters_path: str | Path = DEFAULT_OUTPUT_DIR / 'pattern_clusters.csv',
    cluster_summary_path: str | Path | None = DEFAULT_OUTPUT_DIR / 'cluster_summary.csv',
    out_dir: str | Path = DEFAULT_OUTPUT_DIR / 'figures',
    top_groups: int = 7,
    max_points: int | None = None,
    bins: int = 80,
    seed: int = 0,
    chunksize: int = 500_000,
    dpi: int = 200,
) -> pd.DataFrame:
    """Write the figures and the summary table; return the summary table."""
    if not 1 <= top_groups <= len(GROUP_COLORS):
        raise ValueError(f'top_groups must be 1-{len(GROUP_COLORS)}; more colours cannot be told apart')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models, means = load_errors(errors_path, chunksize)
    cluster_of_pattern, labels = load_groups(pattern_clusters_path, cluster_summary_path)
    reference = next(iter(models.values()))
    _, top = assign_groups(reference['pattern_index'], cluster_of_pattern, top_groups)

    for data in models.values():
        data['cluster'], _ = assign_groups(data['pattern_index'], cluster_of_pattern, top_groups)
        data['x'], data['rank'] = x_positions(data['pattern_index'], data['cluster'], top)

    summary_rows = []
    for model_name, data in models.items():
        frame = pd.DataFrame({'cluster': data['cluster'], 'in_domain': data['in_domain'], 'error': data['error']})
        frame['domain_flag'] = np.where(frame['in_domain'], 'in', 'out')
        stats = frame.groupby(['cluster', 'domain_flag'])['error'].agg(['count', 'mean', 'std', 'min', 'max'])
        for (cid, domain), row in stats.iterrows():
            summary_rows.append({'model_name': model_name, 'cluster_id': int(cid),
                                 'colour_group': f'Cluster {cid}' if cid in top else 'Other',
                                 'cluster_fgs': labels.get(int(cid), ''), 'domain_flag': domain,
                                 'n_molecules': int(row['count']), 'mean_error': row['mean'],
                                 'std_error': row['std'], 'min_error': row['min'], 'max_error': row['max']})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / 'error_plot_summary.csv', index=False)

    # Same molecules (by row position) in every panel when sampling.
    shown = {}
    for model_name, data in models.items():
        n = len(data['error'])
        if max_points is not None and n > max_points:
            keep = np.sort(np.random.default_rng(seed).choice(n, size=max_points, replace=False))
            shown[model_name] = {k: v[keep] for k, v in data.items()}
        else:
            shown[model_name] = data

    all_errors = np.concatenate([d['error'] for d in models.values()])
    lo, hi = float(all_errors.min()), float(all_errors.max())
    pad = 0.04 * (hi - lo or 1.0)
    ylim = (lo - pad, hi + pad)
    n_molecules = max(len(d['error']) for d in models.values())
    xlim = (-0.005 * n_molecules, 1.005 * n_molecules)

    ref_rank, ref_x = models[next(iter(models))]['rank'], models[next(iter(models))]['x']
    centers, names, edges = [], [], []
    group_counts = []
    for slot in range(len(top) + 1):
        xs = ref_x[ref_rank == slot]
        if slot < len(top):
            group_counts.append(len(xs))
        if xs.size == 0:
            continue
        centers.append((xs.min() + xs.max()) / 2.0)
        names.append(f'C{top[slot]}' if slot < len(top) else 'Other')
        if slot:
            edges.append(xs.min() - 0.5)
    n_other = int((ref_rank == len(top)).sum())
    band_ticks = (centers, names, edges)
    handles = _legend_handles(top, labels, group_counts, n_other)
    point_size = 1.5 if max(len(d['error']) for d in shown.values()) > 200_000 else 4.0

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 2.4 * n_models + 0.8), sharex=True, sharey=True,
                             squeeze=False, facecolor=SURFACE)
    for ax, (model_name, data) in zip(axes[:, 0], shown.items()):
        plot_points(ax, model_name, data, means, top, band_ticks, xlim, ylim, point_size)
    axes[-1, 0].set_xlabel('molecules, grouped by FG cluster (same order in every panel)', fontsize=9,
                           color=INK_SECONDARY)
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8,
               labelcolor=INK_SECONDARY)
    fig.suptitle('Sampled error per molecule, all models on the same axes', x=0.01, ha='left', fontsize=12,
                 color=INK)
    fig.tight_layout()
    fig.savefig(out_dir / 'error_points.png', dpi=dpi, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)

    for model_name, data in shown.items():
        fig, ax = plt.subplots(figsize=(12, 3.4), facecolor=SURFACE)
        plot_points(ax, model_name, data, means, top, band_ticks, xlim, ylim, point_size)
        ax.set_xlabel('molecules, grouped by FG cluster (same order in every model figure)', fontsize=9,
                      color=INK_SECONDARY)
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8,
                   labelcolor=INK_SECONDARY)
        fig.tight_layout()
        fig.savefig(out_dir / f'error_points_{_slug(model_name)}.png', dpi=dpi, bbox_inches='tight',
                    facecolor=SURFACE)
        plt.close(fig)

    edges_hist = np.linspace(ylim[0], ylim[1], bins + 1)
    colors = GROUP_COLORS[:len(top)] + [OTHER_COLOR]
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 2.2 * n_models + 0.8), sharex=True, sharey=True,
                             squeeze=False, facecolor=SURFACE)
    for ax, (model_name, data) in zip(axes[:, 0], models.items()):
        _style_axes(ax)
        stacks = [data['error'][data['rank'] == slot] for slot in range(len(top) + 1)]
        ax.hist(stacks, bins=edges_hist, stacked=True, color=colors, edgecolor=SURFACE, linewidth=0.3)
        _mean_lines(ax, model_name, means, vertical=True)
        ax.set_title(model_name, loc='left', fontsize=10, color=INK)
        ax.set_ylabel('molecules', fontsize=9, color=INK_SECONDARY)
    axes[-1, 0].set_xlabel('sampled error', fontsize=9, color=INK_SECONDARY)
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8,
               labelcolor=INK_SECONDARY)
    fig.suptitle('Error distribution per model, same bins and axes, stacked by FG cluster', x=0.01, ha='left',
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out_dir / 'error_histograms.png', dpi=dpi, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--errors', type=str, default=str(DEFAULT_OUTPUT_DIR / 'molecule_model_error_table.csv'))
    parser.add_argument('--pattern-clusters', type=str, default=str(DEFAULT_OUTPUT_DIR / 'pattern_clusters.csv'))
    parser.add_argument('--cluster-summary', type=str, default=str(DEFAULT_OUTPUT_DIR / 'cluster_summary.csv'),
                        help='label_clusters.py output, for the FG names in the legend (optional)')
    parser.add_argument('--out-dir', type=str, default=str(DEFAULT_OUTPUT_DIR / 'figures'))
    parser.add_argument('--top-groups', type=int, default=7,
                        help=f'clusters that get their own colour (1-{len(GROUP_COLORS)}); the rest are Other')
    parser.add_argument('--max-points', type=int, default=None,
                        help='plot at most this many molecules per model (the same ones in every panel); '
                             'default: all. Histograms and the summary always use every molecule')
    parser.add_argument('--bins', type=int, default=80)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunksize', type=int, default=500_000)
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    build_figures(
        errors_path=args.errors,
        pattern_clusters_path=args.pattern_clusters,
        cluster_summary_path=args.cluster_summary,
        out_dir=args.out_dir,
        top_groups=args.top_groups,
        max_points=args.max_points,
        bins=args.bins,
        seed=args.seed,
        chunksize=args.chunksize,
        dpi=args.dpi,
    )
    print(f'Wrote figures and error_plot_summary.csv to {args.out_dir}')


if __name__ == '__main__':
    main()
