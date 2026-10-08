"""Build the molecule-by-model ground-truth error table.

Every model has an in-domain and an out-of-domain normal distribution:

- in-domain:     mean = avg_error_in,                    std = std_error_in
- out-of-domain: mean = avg_error_in + 3 * std_error_in, std = std_error_out (defaults to std_error_in)

Each molecule gets one sampled error per model. Where in the distribution it lands is
decided per FG pattern, so molecules with the same FGs come from the same range:

- each model gives every FG cluster (the pattern's primary cluster) and every FG a random
  effect. A pattern's score is its cluster's effect plus the sum of its FGs' effects plus
  some per-pattern noise, so each FG bunch sits in its own part of the distribution,
  patterns sharing FGs get correlated scores, and the match is not 100%.
- in-domain: patterns are ranked by score.
- out-of-domain: patterns are ranked by their minimum Hamming distance to any of the
  model's in-domain FG vectors (closest first, score breaks ties), so closer patterns
  sample from the lower end of the out-of-domain distribution.

The rank is molecule-weighted and mapped through the normal quantile function, so each
distribution keeps its normal shape. A small per-molecule jitter is added on top.

Reads stage 2's outputs (build_pattern_count_dictionary.py), so it works the same whether
stage 1 wrote fg_presence.csv or fg_matrix.py Parquet:

  pubchem_like_pattern_counts.csv   pattern_index, pattern, count   the unique FG vectors
  molecule_patterns.csv             cid, pattern_index              streamed, one row per molecule

Molecules with no FGs (stage 2's `rejected_no_fg`) are left out, as they are from clustering.
Nearest-neighbour distances are computed once per unique pattern.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = DEFAULT_ROOT / 'molecule-fg-data'
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / 'csv_outputs'

OUT_OF_DOMAIN_SHIFT_STDS = 3.0


def _normal_quantile(u: np.ndarray) -> np.ndarray:
    inv_cdf = NormalDist().inv_cdf
    return np.fromiter((inv_cdf(float(x)) for x in u), dtype=float, count=len(u))


def _rank_positions(order: np.ndarray, counts: np.ndarray, n_patterns: int):
    """Molecule-weighted mid-rank of the patterns in `order` (best first).

    Returns (u, z) arrays of length n_patterns: u in (0, 1) and z = Phi^-1(u). Patterns
    not in `order` stay NaN.
    """
    u = np.full(n_patterns, np.nan)
    z = np.full(n_patterns, np.nan)
    if order.size == 0:
        return u, z
    c = counts[order].astype(float)
    u_order = (np.cumsum(c) - c / 2.0) / c.sum()
    u[order] = u_order
    z[order] = _normal_quantile(u_order)
    return u, z


def load_model_specs(model_specs_path: str | Path, default_std: float) -> pd.DataFrame:
    specs = pd.read_csv(model_specs_path)
    if not {'model_name', 'avg_error_in'}.issubset(specs.columns):
        raise ValueError('model_specs.csv must include model_name and avg_error_in columns')
    if specs['model_name'].duplicated().any():
        raise ValueError('model_specs.csv has duplicate model_name rows')
    specs = specs.copy()
    specs['model_name'] = specs['model_name'].astype(str).str.strip()
    if 'std_error_in' not in specs.columns:
        specs['std_error_in'] = default_std
    specs['std_error_in'] = specs['std_error_in'].fillna(default_std)
    if 'std_error_out' not in specs.columns:
        specs['std_error_out'] = specs['std_error_in']
    specs['std_error_out'] = specs['std_error_out'].fillna(specs['std_error_in'])
    if specs['avg_error_in'].isna().any() or (specs[['std_error_in', 'std_error_out']] <= 0).any().any():
        raise ValueError('model_specs.csv needs an avg_error_in for every model and positive std values')
    specs['avg_error_out'] = specs['avg_error_in'] + OUT_OF_DOMAIN_SHIFT_STDS * specs['std_error_in']
    return specs


def load_patterns(pattern_counts_path: str | Path):
    """Read the unique FG vectors and their molecule counts from stage 2.

    Returns (pattern_ids, bits, counts, no_fg_ids). Patterns with no FG are returned
    separately in no_fg_ids and left out of the rest.
    """
    patterns = pd.read_csv(pattern_counts_path, dtype={'pattern': str})
    if not {'pattern_index', 'pattern', 'count'}.issubset(patterns.columns):
        raise ValueError(f'{pattern_counts_path} must include pattern_index, pattern and count columns')
    if patterns['pattern_index'].duplicated().any():
        raise ValueError(f'{pattern_counts_path} lists a pattern_index twice')
    lengths = patterns['pattern'].str.len()
    if lengths.nunique() != 1:
        raise ValueError(f'{pattern_counts_path}: patterns have different lengths {sorted(lengths.unique())}')

    patterns = patterns.sort_values('pattern_index')
    n_fgs = int(lengths.iloc[0])
    bits = (np.frombuffer(''.join(patterns['pattern']).encode('ascii'), dtype=np.uint8)
            .reshape(len(patterns), n_fgs) - ord('0'))
    if bits.max() > 1:
        raise ValueError(f'{pattern_counts_path}: patterns must be 0/1 strings')
    has_fg = bits.any(axis=1)
    pattern_ids = patterns['pattern_index'].to_numpy(dtype=np.int64)
    counts = patterns['count'].to_numpy(dtype=np.int64)
    return pattern_ids[has_fg], bits[has_fg], counts[has_fg], pattern_ids[~has_fg]


def load_domains(pattern_model_map_path: str | Path, pattern_clusters: pd.DataFrame, model_names: list[str]):
    """Return {model_name: set of in-domain pattern_index}."""
    pattern_model_map = pd.read_csv(pattern_model_map_path)
    if 'model_name' not in pattern_model_map.columns:
        raise ValueError('pattern_cluster_model_map.csv must include a model_name column')
    pattern_model_map['model_name'] = pattern_model_map['model_name'].astype(str).str.strip()

    if 'pattern_index' not in pattern_model_map.columns:
        # Cluster-level map: expand every cluster to all of its member patterns.
        if 'cluster_id' not in pattern_model_map.columns or not {'cluster_id', 'pattern_index'}.issubset(pattern_clusters.columns):
            raise ValueError('pattern_cluster_model_map.csv needs pattern_index, or cluster_id plus pattern_clusters.csv')
        pattern_model_map = pattern_model_map.merge(pattern_clusters[['cluster_id', 'pattern_index']], on='cluster_id')

    domains = {
        model_name: set(group['pattern_index'].astype(np.int64).tolist())
        for model_name, group in pattern_model_map.groupby('model_name')
    }
    unknown = sorted(set(domains) - set(model_names))
    if unknown:
        print(f'WARNING: models in the pattern-model map but not in model_specs.csv (ignored): {unknown}')
    for model_name in model_names:
        if not domains.get(model_name):
            print(f'WARNING: model {model_name!r} has no in-domain patterns; every molecule is out-of-domain for it')
    return {model_name: domains.get(model_name, set()) for model_name in model_names}


def nearest_in_domain(bits: np.ndarray, in_mask: np.ndarray, block_out: int = 1024, block_in: int = 16384):
    """Minimum Hamming distance from every out-of-domain pattern to the in-domain patterns.

    Uses d(a, b) = |a| + |b| - 2 a.b on float32 blocks (exact for 0/1 vectors this size).
    Returns (distance, nearest pattern position) arrays over all patterns; in-domain
    patterns get distance 0 and themselves, and every pattern gets inf / -1 when the
    model has no in-domain patterns.
    """
    n_patterns = len(bits)
    distance = np.full(n_patterns, np.inf)
    nearest = np.full(n_patterns, -1, dtype=np.int64)
    ref_idx = np.flatnonzero(in_mask)
    distance[ref_idx] = 0.0
    nearest[ref_idx] = ref_idx
    out_idx = np.flatnonzero(~in_mask)
    if ref_idx.size == 0 or out_idx.size == 0:
        return distance, nearest

    ref = bits[ref_idx].astype(np.float32)
    ref_norm = ref.sum(axis=1)
    for start in range(0, out_idx.size, block_out):
        rows = out_idx[start:start + block_out]
        x = bits[rows].astype(np.float32)
        x_norm = x.sum(axis=1)
        best = np.full(rows.size, np.inf)
        best_pos = np.full(rows.size, -1, dtype=np.int64)
        for ref_start in range(0, ref_idx.size, block_in):
            ref_block = ref[ref_start:ref_start + block_in]
            d = x_norm[:, None] + ref_norm[None, ref_start:ref_start + block_in] - 2.0 * (x @ ref_block.T)
            j = d.argmin(axis=1)
            dj = d[np.arange(rows.size), j]
            better = dj < best
            best[better] = dj[better]
            best_pos[better] = ref_idx[ref_start + j[better]]
        distance[rows] = np.rint(best)
        nearest[rows] = best_pos
    return distance, nearest


def fg_score(bits: np.ndarray, rng: np.random.Generator, block: int = 65536) -> np.ndarray:
    """Sum of random per-FG effects, scaled so every pattern's score has unit variance."""
    effects = rng.standard_normal(bits.shape[1]).astype(np.float32)
    score = np.empty(len(bits))
    for start in range(0, len(bits), block):
        score[start:start + block] = bits[start:start + block].astype(np.float32) @ effects
    return score / np.sqrt(np.maximum(bits.sum(axis=1), 1))


def cluster_score(pattern_ids: np.ndarray, cluster_of_pattern: dict[int, int], rng: np.random.Generator) -> np.ndarray:
    """One random effect per primary cluster; a pattern with no cluster gets one of its own."""
    cluster = np.array([cluster_of_pattern.get(int(i), -1) for i in pattern_ids], dtype=np.int64)
    ids, inverse = np.unique(cluster, return_inverse=True)
    score = rng.standard_normal(len(ids))[inverse]
    unclustered = cluster < 0
    score[unclustered] = rng.standard_normal(int(unclustered.sum()))
    return score


def build_pattern_positions(
    specs: pd.DataFrame,
    domains: dict[str, set],
    pattern_ids: np.ndarray,
    bits: np.ndarray,
    counts: np.ndarray,
    cluster_of_pattern: dict[int, int],
    pattern_rngs: list[np.random.Generator],
    cluster_weight: float,
    fg_weight: float,
    jitter: float,
) -> pd.DataFrame:
    """Place every (pattern, model) pair in that model's in- or out-of-domain distribution."""
    n_patterns = len(pattern_ids)
    n_fgs = bits.sum(axis=1)
    noise_weight = 1.0 - cluster_weight - fg_weight
    frames = []
    for spec, rng in zip(specs.itertuples(index=False), pattern_rngs):
        in_mask = np.isin(pattern_ids, np.fromiter(domains[spec.model_name], dtype=np.int64))
        in_idx = np.flatnonzero(in_mask)
        out_idx = np.flatnonzero(~in_mask)

        # Same cluster -> same range; shared FGs -> correlated score; the noise keeps it below 100%.
        score = (np.sqrt(cluster_weight) * cluster_score(pattern_ids, cluster_of_pattern, rng)
                 + np.sqrt(fg_weight) * fg_score(bits, rng)
                 + np.sqrt(noise_weight) * rng.standard_normal(n_patterns))

        distance, nearest = nearest_in_domain(bits, in_mask)

        _, z_in = _rank_positions(in_idx[np.argsort(score[in_idx], kind='stable')], counts, n_patterns)
        # lexsort: last key is primary, so closest distance first, FG score breaks ties.
        u_out, z_out = _rank_positions(out_idx[np.lexsort((score[out_idx], distance[out_idx]))], counts, n_patterns)

        z = np.where(in_mask, z_in, z_out)
        mean = np.where(in_mask, spec.avg_error_in, spec.avg_error_out)
        std = np.where(in_mask, spec.std_error_in, spec.std_error_out)
        nearest_pattern = np.where(nearest >= 0, pattern_ids[np.maximum(nearest, 0)], -1)

        frame = pd.DataFrame({
            'pattern_index': pattern_ids,
            'model_name': spec.model_name,
            'domain_flag': np.where(in_mask, 'in', 'out'),
            'n_molecules': counts,
            'n_fgs': n_fgs,
            'nearest_distance': np.where(in_mask, np.nan, distance),
            'nearest_reference_pattern': np.where(in_mask, -1, nearest_pattern),
            'distance_rank': u_out,
            'error_position': z,
            'error_mean': mean,
            'error_std': std,
            'pattern_error': mean + std * np.sqrt(1.0 - jitter ** 2) * z,
        })
        frame['nearest_reference_cluster'] = frame['nearest_reference_pattern'].map(cluster_of_pattern)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_model_error_table(
    pattern_counts_path: str | Path = DEFAULT_OUTPUT_DIR / 'pubchem_like_pattern_counts.csv',
    molecule_patterns_path: str | Path = DEFAULT_OUTPUT_DIR / 'molecule_patterns.csv',
    model_specs_path: str | Path = DEFAULT_OUTPUT_DIR / 'model_specs.csv',
    pattern_clusters_path: str | Path = DEFAULT_OUTPUT_DIR / 'pattern_clusters.csv',
    pattern_model_map_path: str | Path = DEFAULT_OUTPUT_DIR / 'pattern_cluster_model_map.csv',
    output_path: str | Path = DEFAULT_OUTPUT_DIR / 'molecule_model_error_table.csv',
    pattern_output_path: str | Path | None = DEFAULT_OUTPUT_DIR / 'pattern_model_error_table.csv',
    seed: int = 0,
    default_std: float = 0.05,
    cluster_weight: float = 0.6,
    fg_weight: float = 0.3,
    jitter: float = 0.2,
    min_error: float | None = 0.0,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """Write one sampled error per (molecule, model) and return a per-model summary.

    cluster_weight: share of a pattern's position that comes from its primary FG cluster,
        so each cluster occupies its own part of the distribution.
    fg_weight: share that comes from its FGs (patterns sharing FGs land close together).
        The rest (1 - cluster_weight - fg_weight) is per-pattern noise.
    jitter: share (in std units) of each molecule's error that is its own noise, so
        molecules with the same FGs land in the same range but not on the same value.
    min_error: errors below this are clipped (None = no clipping).
    """
    if min(cluster_weight, fg_weight) < 0.0 or cluster_weight + fg_weight > 1.0 or not 0.0 <= jitter < 1.0:
        raise ValueError('cluster_weight and fg_weight must be >= 0 and sum to at most 1, and jitter in [0, 1)')

    specs = load_model_specs(model_specs_path, default_std)
    model_names = specs['model_name'].tolist()
    pattern_clusters = pd.read_csv(pattern_clusters_path) if Path(pattern_clusters_path).exists() else pd.DataFrame()
    domains = load_domains(pattern_model_map_path, pattern_clusters, model_names)

    cluster_of_pattern = {}
    if {'pattern_index', 'cluster_id'}.issubset(pattern_clusters.columns):
        first = pattern_clusters.drop_duplicates('pattern_index')
        cluster_of_pattern = dict(zip(first['pattern_index'].astype(np.int64), first['cluster_id'].astype(np.int64)))

    pattern_ids, bits, counts, no_fg_ids = load_patterns(pattern_counts_path)
    print(f'{counts.sum()} molecules, {len(pattern_ids)} unique FG patterns, {bits.shape[1]} FGs'
          f' ({len(no_fg_ids)} no-FG pattern(s) left out)')

    seeds = np.random.SeedSequence(seed).spawn(2 * len(specs))
    pattern_rngs = [np.random.default_rng(s) for s in seeds[0::2]]
    molecule_rngs = [np.random.default_rng(s) for s in seeds[1::2]]

    patterns = build_pattern_positions(
        specs, domains, pattern_ids, bits, counts, cluster_of_pattern, pattern_rngs, cluster_weight, fg_weight, jitter,
    )
    if pattern_output_path is not None:
        Path(pattern_output_path).parent.mkdir(parents=True, exist_ok=True)
        patterns.to_csv(pattern_output_path, index=False)

    # Per model: arrays aligned with pattern_ids, for fast lookup while streaming molecules.
    per_model = {
        model_name: group.set_index('pattern_index').loc[pattern_ids]
        for model_name, group in patterns.groupby('model_name', sort=False)
    }
    stats = {(m, d): [0, 0.0, 0.0] for m in model_names for d in ('in', 'out')}
    clipped = dict.fromkeys(model_names, 0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        'cid', 'pattern_index', 'model_name', 'domain_flag', 'nearest_distance',
        'nearest_reference_cluster', 'distance_rank', 'error_mean', 'error_std', 'sampled_error',
    ]
    pd.DataFrame(columns=columns).to_csv(output_path, index=False)

    reader = pd.read_csv(molecule_patterns_path, usecols=['cid', 'pattern_index'], dtype={'cid': str}, chunksize=chunksize)
    for chunk in reader:
        chunk = chunk[~chunk['pattern_index'].isin(no_fg_ids)]
        if chunk.empty:
            continue
        chunk_ids = chunk['pattern_index'].to_numpy(dtype=np.int64)
        positions = np.minimum(np.searchsorted(pattern_ids, chunk_ids), len(pattern_ids) - 1)
        if not np.array_equal(pattern_ids[positions], chunk_ids):
            raise ValueError(f'{molecule_patterns_path} has pattern_index values missing from {pattern_counts_path}')
        for model_name, rng in zip(model_names, molecule_rngs):
            p = per_model[model_name]
            domain = p['domain_flag'].to_numpy()[positions]
            std = p['error_std'].to_numpy()[positions]
            error = p['pattern_error'].to_numpy()[positions] + std * jitter * rng.standard_normal(len(chunk))
            if min_error is not None:
                clipped[model_name] += int((error < min_error).sum())
                error = np.maximum(error, min_error)
            for d in ('in', 'out'):
                e = error[domain == d]
                s = stats[(model_name, d)]
                s[0] += e.size
                s[1] += float(e.sum())
                s[2] += float((e ** 2).sum())
            pd.DataFrame({
                'cid': chunk['cid'].to_numpy(),
                'pattern_index': chunk['pattern_index'].to_numpy(),
                'model_name': model_name,
                'domain_flag': domain,
                'nearest_distance': p['nearest_distance'].to_numpy()[positions],
                'nearest_reference_cluster': p['nearest_reference_cluster'].to_numpy()[positions],
                'distance_rank': p['distance_rank'].to_numpy()[positions],
                'error_mean': p['error_mean'].to_numpy()[positions],
                'error_std': std,
                'sampled_error': error,
            }, columns=columns).to_csv(output_path, mode='a', header=False, index=False)

    summary_rows = []
    for spec in specs.itertuples(index=False):
        for d, target_mean, target_std in (
            ('in', spec.avg_error_in, spec.std_error_in),
            ('out', spec.avg_error_out, spec.std_error_out),
        ):
            n, total, total_sq = stats[(spec.model_name, d)]
            mean = total / n if n else np.nan
            std = np.sqrt(max(total_sq / n - mean ** 2, 0.0)) if n else np.nan
            summary_rows.append({
                'model_name': spec.model_name, 'domain_flag': d, 'n_molecules': n,
                'target_mean': target_mean, 'sampled_mean': mean,
                'target_std': target_std, 'sampled_std': std,
            })
        if clipped[spec.model_name]:
            print(f'{spec.model_name}: clipped {clipped[spec.model_name]} errors below {min_error}')
    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description='Build a molecule-model error table from FG patterns and model coverage rules.')
    parser.add_argument('--pattern-counts', type=str, default=str(DEFAULT_OUTPUT_DIR / 'pubchem_like_pattern_counts.csv'))
    parser.add_argument('--molecule-patterns', type=str, default=str(DEFAULT_OUTPUT_DIR / 'molecule_patterns.csv'))
    parser.add_argument('--model-specs', type=str, default=str(DEFAULT_OUTPUT_DIR / 'model_specs.csv'))
    parser.add_argument('--pattern-clusters', type=str, default=str(DEFAULT_OUTPUT_DIR / 'pattern_clusters.csv'))
    parser.add_argument('--pattern-model-map', type=str, default=str(DEFAULT_OUTPUT_DIR / 'pattern_cluster_model_map.csv'))
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_DIR / 'molecule_model_error_table.csv'))
    parser.add_argument('--pattern-output', type=str, default=str(DEFAULT_OUTPUT_DIR / 'pattern_model_error_table.csv'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--std', type=float, default=0.05, help='std used when model_specs.csv has no std_error_in')
    parser.add_argument('--cluster-weight', type=float, default=0.6,
                        help='share of the error position driven by the FG cluster (0-1)')
    parser.add_argument('--fg-weight', type=float, default=0.3,
                        help='share driven by the individual FGs (0-1); the rest is per-pattern noise')
    parser.add_argument('--jitter', type=float, default=0.2, help='per-molecule noise, in std units (0-1)')
    parser.add_argument('--min-error', type=float, default=0.0, help='clip errors below this value')
    parser.add_argument('--no-clip', action='store_true', help='allow negative errors')
    parser.add_argument('--chunksize', type=int, default=50_000)
    args = parser.parse_args()

    summary = build_model_error_table(
        pattern_counts_path=args.pattern_counts,
        molecule_patterns_path=args.molecule_patterns,
        model_specs_path=args.model_specs,
        pattern_clusters_path=args.pattern_clusters,
        pattern_model_map_path=args.pattern_model_map,
        output_path=args.output,
        pattern_output_path=args.pattern_output,
        seed=args.seed,
        default_std=args.std,
        cluster_weight=args.cluster_weight,
        fg_weight=args.fg_weight,
        jitter=args.jitter,
        min_error=None if args.no_clip else args.min_error,
        chunksize=args.chunksize,
    )
    print(f'Wrote molecule-model errors to {args.output}')
    print(f'Wrote pattern-model errors to {args.pattern_output}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
