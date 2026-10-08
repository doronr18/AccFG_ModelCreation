"""Plot every sampled error for every model, coloured by FG cluster.

Reads error_distributions.py's molecule_model_error_table.csv and writes to --out-dir:

  error_points.png              one row per model, in-domain and out-of-domain side by side,
                                every panel on the same y axis (the sampled error itself):
                                x = that panel's molecules ordered by FG cluster then pattern,
                                one dot per molecule
  error_points_<model>.png      one model's two panels alone, on the same axes
  error_cluster_boxes.png       per model and domain, one box per cluster (5-95% whiskers),
                                to show whether the FG bunches sit in different ranges
  error_histograms.png          per model and domain, same bins, stacked by cluster
  error_plot_summary.csv        the numbers behind the figures: n, mean, std, min, max per
                                model, cluster and domain

A molecule's group is its pattern's primary cluster (pattern_clusters.csv). The largest
--top-groups clusters (by molecule count) get their own colour; the rest are grey "Other".
Clusters sit side by side along x, so each colour is also its own x band, labelled under
the axis. A dashed line marks the domain's mean (error_mean in the table).

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
    return models, means


def check_table(models, means):
    """Print what in the table would make the figures misleading."""
    for name, data in models.items():
        n_in = int(data['in_domain'].sum())
        print(f'{name}: {n_in:,} in-domain, {len(data["in_domain"]) - n_in:,} out-of-domain molecules')
        if n_in == len(data['in_domain']):
            print(f'WARNING: {name} has no out-of-domain molecules: its clusters cover every pattern '
                  '(check the targets in model_specs.csv and pattern_cluster_model_map.csv)')
        mean_in, mean_out = means.get((name, 'in')), means.get((name, 'out'))
        if mean_in is not None and mean_out is not None and mean_out <= mean_in:
            print(f'WARNING: {name}: out-of-domain mean {mean_out:g} is not above in-domain mean {mean_in:g}')
    if means and all(v == 0 for v in means.values()):
        print('WARNING: error_mean is 0 in every row. The table was made without avg_error_in (the original '
              'error_distributions.py defaulted it to 0); re-run the current script with model_specs.csv '
              'holding avg_error_in for every model')


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


def panel_layout(pattern_index: np.ndarray, rank: np.ndarray, cluster: np.ndarray, top: list[int]):
    """x in [0, 1] for one panel's molecules: coloured clusters (largest first), then Other;
    within a cluster, by pattern. Returns (x, (band centres, band names, band edges))."""
    n = len(rank)
    order = np.lexsort((pattern_index, cluster, rank))
    x = np.empty(n)
    x[order] = (np.arange(n) + 0.5) / max(n, 1)
    centers, names, edges = [], [], []
    sorted_rank = rank[order]
    for slot in range(len(top) + 1):
        lo, hi = np.searchsorted(sorted_rank, slot, 'left'), np.searchsorted(sorted_rank, slot, 'right')
        if hi == lo:
            continue
        centers.append((lo + hi) / 2.0 / n)
        names.append(f'C{top[slot]}' if slot < len(top) else 'Other')
        if lo:
            edges.append(lo / n)
    return x, (centers, names, edges)


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


def _empty(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center', fontsize=9, color=INK_MUTED)


def _legend_handles(top, labels, group_counts, n_other):
    handles = []
    for slot, cid in enumerate(top):
        text = f'Cluster {cid}: {labels.get(cid, "")}'.rstrip(': ') + f'  (n={group_counts[slot]:,})'
        handles.append(Line2D([], [], linestyle='', marker='o', markersize=7,
                              markerfacecolor=GROUP_COLORS[slot], markeredgecolor=SURFACE, label=text))
    if n_other:
        handles.append(Line2D([], [], linestyle='', marker='o', markersize=7, markerfacecolor=OTHER_COLOR,
                              markeredgecolor=SURFACE, label=f'Other clusters  (n={n_other:,})'))
    handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=0.9, linestyle='--', label='domain mean'))
    return handles


DOMAINS = (('in', True, 'in-domain'), ('out', False, 'out-of-domain'))


def plot_points(ax, model_name, data, domain, means, top, ylim, point_size):
    """One model's molecules in one domain: x = cluster bands, y = sampled error."""
    _style_axes(ax)
    key, flag, label = domain
    mask = data['in_domain'] == flag
    ax.set_title(f'{model_name}  ·  {label}  ({int(mask.sum()):,})', loc='left', fontsize=10, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    if not mask.any():
        _empty(ax, f'no {label} molecules for this model')
        ax.set_xticks([])
        return
    rank, error = data['rank'][mask], data['error'][mask]
    x, (centers, names, edges) = panel_layout(data['pattern_index'][mask], rank, data['cluster'][mask], top)
    colors = np.array(GROUP_COLORS[:len(top)] + [OTHER_COLOR])
    # Other first so the coloured clusters draw on top where bands touch.
    draw = np.argsort(rank != len(top), kind='stable')
    ax.scatter(x[draw], error[draw], c=colors[rank[draw]], s=point_size, linewidths=0, rasterized=True)
    if (model_name, key) in means:
        ax.axhline(means[(model_name, key)], color=INK_SECONDARY, linewidth=0.9, linestyle='--')
    ax.set_xticks(centers)
    ax.set_xticklabels(names, fontsize=8)
    for edge in edges:
        ax.axvline(edge, color=GRID, linewidth=0.6)


def plot_boxes(ax, model_name, data, domain, means, top, ylim):
    """One box per cluster: where each FG bunch sits in this model's distribution."""
    _style_axes(ax)
    key, flag, label = domain
    mask = data['in_domain'] == flag
    ax.set_title(f'{model_name}  ·  {label}', loc='left', fontsize=10, color=INK)
    ax.set_ylim(*ylim)
    slots = [s for s in range(len(top) + 1) if (mask & (data['rank'] == s)).any()]
    if not slots:
        _empty(ax, f'no {label} molecules for this model')
        ax.set_xticks([])
        return
    boxes = ax.boxplot([data['error'][mask & (data['rank'] == s)] for s in slots], positions=range(len(slots)),
                       whis=(5, 95), showfliers=False, widths=0.6, patch_artist=True,
                       medianprops=dict(color=INK, linewidth=1.0),
                       whiskerprops=dict(color=INK_SECONDARY), capprops=dict(color=INK_SECONDARY))
    for patch, slot in zip(boxes['boxes'], slots):
        patch.set_facecolor(GROUP_COLORS[slot] if slot < len(top) else OTHER_COLOR)
        patch.set_edgecolor(SURFACE)
    if (model_name, key) in means:
        ax.axhline(means[(model_name, key)], color=INK_SECONDARY, linewidth=0.9, linestyle='--')
    ax.set_xticks(range(len(slots)))
    ax.set_xticklabels([f'C{top[s]}' if s < len(top) else 'Other' for s in slots], fontsize=8)


def _grid(n_models, height):
    return plt.subplots(n_models, 2, figsize=(14, height * n_models + 0.8), sharey=True, squeeze=False,
                        facecolor=SURFACE)


def _finish(fig, handles, title, path, dpi):
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8,
               labelcolor=INK_SECONDARY)
    if title:
        fig.suptitle(title, x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


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
    check_table(models, means)
    cluster_of_pattern, labels = load_groups(pattern_clusters_path, cluster_summary_path)
    reference = next(iter(models.values()))
    _, top = assign_groups(reference['pattern_index'], cluster_of_pattern, top_groups)

    for data in models.values():
        data['cluster'], _ = assign_groups(data['pattern_index'], cluster_of_pattern, top_groups)
        data['rank'] = np.full(len(data['cluster']), len(top), dtype=np.int64)
        for slot, cid in enumerate(top):
            data['rank'][data['cluster'] == cid] = slot

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

    # One y range for every panel, from the errors themselves, so values read off directly.
    all_errors = np.concatenate([d['error'] for d in models.values()])
    lo, hi = float(all_errors.min()), float(all_errors.max())
    pad = 0.04 * (hi - lo or 1.0)
    ylim = (lo - pad, hi + pad)

    group_counts = [int((reference['rank'] == slot).sum()) for slot in range(len(top))]
    n_other = int((reference['rank'] == len(top)).sum())
    handles = _legend_handles(top, labels, group_counts, n_other)
    point_size = 1.5 if max(len(d['error']) for d in shown.values()) > 200_000 else 4.0
    n_models = len(models)
    xlabel = 'molecules in this panel, grouped by FG cluster'

    fig, axes = _grid(n_models, 2.4)
    for row, (model_name, data) in zip(axes, shown.items()):
        for ax, domain in zip(row, DOMAINS):
            plot_points(ax, model_name, data, domain, means, top, ylim, point_size)
        row[0].set_ylabel('sampled error', fontsize=9, color=INK_SECONDARY)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    _finish(fig, handles, 'Sampled error per molecule: in-domain | out-of-domain, all panels on the same y axis',
            out_dir / 'error_points.png', dpi)

    for model_name, data in shown.items():
        fig, row = plt.subplots(1, 2, figsize=(14, 3.4), sharey=True, facecolor=SURFACE)
        for ax, domain in zip(row, DOMAINS):
            plot_points(ax, model_name, data, domain, means, top, ylim, point_size)
            ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
        row[0].set_ylabel('sampled error', fontsize=9, color=INK_SECONDARY)
        _finish(fig, handles, None, out_dir / f'error_points_{_slug(model_name)}.png', dpi)

    fig, axes = _grid(n_models, 2.2)
    for row, (model_name, data) in zip(axes, models.items()):
        for ax, domain in zip(row, DOMAINS):
            plot_boxes(ax, model_name, data, domain, means, top, ylim)
        row[0].set_ylabel('sampled error', fontsize=9, color=INK_SECONDARY)
    _finish(fig, handles, 'Error range of each FG cluster (box 25-75%, whiskers 5-95%), same y axis',
            out_dir / 'error_cluster_boxes.png', dpi)

    edges_hist = np.linspace(ylim[0], ylim[1], bins + 1)
    colors = GROUP_COLORS[:len(top)] + [OTHER_COLOR]
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 2.2 * n_models + 0.8), sharex=True, squeeze=False,
                             facecolor=SURFACE)
    for row, (model_name, data) in zip(axes, models.items()):
        for ax, (key, flag, label) in zip(row, DOMAINS):
            _style_axes(ax)
            mask = data['in_domain'] == flag
            ax.set_title(f'{model_name}  ·  {label}', loc='left', fontsize=10, color=INK)
            if not mask.any():
                _empty(ax, f'no {label} molecules for this model')
                continue
            stacks = [data['error'][mask & (data['rank'] == slot)] for slot in range(len(top) + 1)]
            ax.hist(stacks, bins=edges_hist, stacked=True, color=colors, edgecolor=SURFACE, linewidth=0.3)
            if (model_name, key) in means:
                ax.axvline(means[(model_name, key)], color=INK_SECONDARY, linewidth=0.9, linestyle='--')
        row[0].set_ylabel('molecules', fontsize=9, color=INK_SECONDARY)
    for ax in axes[-1]:
        ax.set_xlabel('sampled error', fontsize=9, color=INK_SECONDARY)
    _finish(fig, handles, 'Error distribution per model and domain, same bins, stacked by FG cluster',
            out_dir / 'error_histograms.png', dpi)
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
