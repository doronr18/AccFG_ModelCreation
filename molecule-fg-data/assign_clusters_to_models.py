"""Stage 5: assign whole clusters to models so each model's coverage is as close as possible
to its target.

Rules:
  * every cluster is assigned to at least one model;
  * a cluster is assigned whole or not at all -- never part of its molecules;
  * a cluster may be assigned to several models (the targets in model_specs.csv may sum to
    more than 100%, in which case some must be);
  * a model's coverage = unique molecules in the union of its clusters / all clustered
    molecules. A molecule that belongs to two clusters given to the same model counts once;
  * min_fgs / max_fgs in model_specs.csv are ignored: any cluster may go to any model;
  * every model keeps at least --min-exclusive (default 0.01 = 1%) of the clustered molecules
    that no other model covers; a `min_exclusive` column in model_specs.csv sets it per model.

Subject to these rules the sum over models of |coverage - target| is minimised exactly, as a
small integer program (scipy.optimize.milp, i.e. HiGHS). Overlapping memberships are handled
by grouping patterns into "atoms" that share one membership set; an atom is covered by a
model if any of its clusters is.

If no whole-cluster assignment can give every model its exclusive share, or any model ends
further than --tolerance (default 0.02 = 2 percentage points) from its
target, the clusters are too coarse for the targets: every output is still written, a message
saying to re-run build_pattern_clusters.py with more clusters is printed and saved to
csv_outputs/assignment_warnings.txt, and the script exits with status 3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'molecule-fg-data'
OUTPUT_DIR = DATA_DIR / 'csv_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

CLUSTER_SUMMARY = OUTPUT_DIR / 'cluster_summary.csv'
MODEL_SPECS = OUTPUT_DIR / 'model_specs.csv'
MODEL_ASSIGNMENTS = OUTPUT_DIR / 'model_assignments.csv'
MODEL_REPORT = OUTPUT_DIR / 'model_coverage_report.csv'
PATTERN_MODEL_MAP = OUTPUT_DIR / 'pattern_cluster_model_map.csv'
MODEL_MOLECULE_COUNTS = OUTPUT_DIR / 'model_molecule_counts.csv'
ASSIGNMENT_WARNINGS = OUTPUT_DIR / 'assignment_warnings.txt'

EXIT_NEEDS_MORE_CLUSTERS = 3

# Bit-string and list columns must stay text; see label_clusters.TEXT_COLUMNS.
TEXT_COLUMNS = {
    'pattern': str,
    'cluster_representative': str,
    'cluster_memberships': str,
    'cluster_probabilities': str,
    'member_cids': str,
}

# Per-assignment cost in the objective, in units of coverage fraction. It only breaks ties
# toward fewer assignments: one assignment costs as much as a millionth of the molecules.
TIE_BREAK = 1e-6


def parse_memberships(value, fallback) -> list:
    """'0,3' -> [0, 3]. Falls back to [fallback] when the memberships cell is empty."""
    if pd.isna(value) or not str(value).strip():
        return [int(fallback)]
    return [int(part) for part in str(value).split(',') if part.strip()]


def build_atoms(pattern_clusters: pd.DataFrame) -> tuple[list, np.ndarray]:
    """Group patterns by membership set: returns (sorted cluster-id tuples, molecules per tuple).

    Every molecule in an atom belongs to exactly the clusters in its tuple, so a model covers
    the whole atom as soon as it is given any one of those clusters.
    """
    memberships = (pattern_clusters['cluster_memberships']
                   if 'cluster_memberships' in pattern_clusters.columns
                   else pattern_clusters['cluster_id'])
    totals: dict = {}
    for value, fallback, n in zip(memberships, pattern_clusters['cluster_id'], pattern_clusters['count']):
        key = tuple(sorted(set(parse_memberships(value, fallback))))
        totals[key] = totals.get(key, 0) + int(n)
    keys = sorted(totals)
    return keys, np.array([totals[k] for k in keys], dtype=float)


def coverage(assign: np.ndarray, atom_sets: list, atom_counts: np.ndarray) -> np.ndarray:
    """Fraction of molecules each model covers.

    assign is (n_models, n_clusters) bool; atom_sets hold column indices into it.
    """
    total = atom_counts.sum()
    out = np.zeros(assign.shape[0])
    for a, cols in enumerate(atom_sets):
        out += assign[:, list(cols)].any(axis=1) * atom_counts[a]
    return out / total


def solve_assignment(atom_sets: list, atom_counts: np.ndarray, n_clusters: int,
                     targets: np.ndarray, time_limit: float = 600.0,
                     min_exclusive: np.ndarray | None = None):
    """Return (assign (n_models, n_clusters) bool, proven_optimal, solver message).
    assign is None when there is no feasible assignment.

    Variables: x[m,c] in {0,1} (cluster c given to model m); y[m,a] in [0,1] (atom a covered
    by model m), forced to OR(x[m,c] for c in atom a); d[m] >= |coverage[m] - target[m]|.
    Minimise sum(d) + TIE_BREAK * sum(x) subject to every cluster in >= 1 model.

    With min_exclusive (a fraction per model), also z[m,a] in [0,1] with z[m,a] <= y[m,a] and
    z[m,a] <= 1 - y[m',a] for every other model m', so z can only be 1 for atoms model m
    covers alone, and sum_a share[a] * z[m,a] >= min_exclusive[m].
    """
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as err:
        raise SystemExit(f'assign_clusters_to_models.py needs scipy >= 1.9 '
                         f'(scipy.optimize.milp): {err}')

    M, K, A = len(targets), n_clusters, len(atom_sets)
    share = atom_counts / atom_counts.sum()
    nx, ny = M * K, M * A
    exclusive = min_exclusive is not None and np.any(np.asarray(min_exclusive) > 0)
    nz = M * A if exclusive else 0

    def xi(m, c):
        return m * K + c

    def yi(m, a):
        return nx + m * A + a

    def di(m):
        return nx + ny + m

    def zi(m, a):
        return nx + ny + M + m * A + a

    rows, cols, vals, lo, hi = [], [], [], [], []

    def add(entries, lb, ub):
        r = len(lo)
        for col, val in entries:
            rows.append(r)
            cols.append(col)
            vals.append(val)
        lo.append(lb)
        hi.append(ub)

    for c in range(K):  # every cluster in at least one model
        add([(xi(m, c), 1.0) for m in range(M)], 1.0, np.inf)
    for m in range(M):  # y[m,a] = OR of x[m,c] over the atom's clusters
        for a, atom in enumerate(atom_sets):
            add([(yi(m, a), 1.0)] + [(xi(m, c), -1.0) for c in atom], -np.inf, 0.0)
            for c in atom:
                add([(yi(m, a), 1.0), (xi(m, c), -1.0)], 0.0, np.inf)
    for m in range(M):  # d[m] >= target - coverage and d[m] >= coverage - target
        cov = [(yi(m, a), share[a]) for a in range(A)]
        add([(di(m), 1.0)] + [(col, -v) for col, v in cov], -targets[m], np.inf)
        add([(di(m), 1.0)] + cov, targets[m], np.inf)
    if exclusive:  # z[m,a] = 1 only where model m alone covers atom a
        for m in range(M):
            for a in range(A):
                add([(zi(m, a), 1.0), (yi(m, a), -1.0)], -np.inf, 0.0)
                for other in range(M):
                    if other != m:
                        add([(zi(m, a), 1.0), (yi(other, a), 1.0)], -np.inf, 1.0)
            add([(zi(m, a), share[a]) for a in range(A)], float(min_exclusive[m]), np.inf)

    n = nx + ny + M + nz
    objective = np.zeros(n)
    objective[:nx] = TIE_BREAK
    objective[nx + ny:nx + ny + M] = 1.0
    integrality = np.zeros(n)
    integrality[:nx] = 1
    upper = np.ones(n)
    upper[nx + ny:nx + ny + M] = np.inf
    constraints = LinearConstraint(coo_matrix((vals, (rows, cols)), shape=(len(lo), n)).tocsr(),
                                   np.array(lo), np.array(hi))
    res = milp(objective, constraints=constraints, integrality=integrality,
               bounds=Bounds(np.zeros(n), upper),
               options={'time_limit': time_limit, 'mip_rel_gap': 1e-9, 'disp': False})
    if res.x is None:
        return None, False, res.message
    return res.x[:nx].reshape(M, K) > 0.5, res.status == 0, res.message


def exclusive_coverage(assign: np.ndarray, atom_sets: list, atom_counts: np.ndarray) -> np.ndarray:
    """Fraction of molecules each model covers and no other model does."""
    total = atom_counts.sum()
    out = np.zeros(assign.shape[0])
    for a, cols in enumerate(atom_sets):
        covering = np.flatnonzero(assign[:, list(cols)].any(axis=1))
        if len(covering) == 1:
            out[covering[0]] += atom_counts[a]
    return out / total


class NoAssignment(RuntimeError):
    """The solver found no assignment; `infeasible` says whether none exists."""

    def __init__(self, message: str, infeasible: bool):
        super().__init__(message)
        self.infeasible = infeasible


def assign_models(cluster_summary: pd.DataFrame,
                  model_specs: pd.DataFrame,
                  pattern_clusters: pd.DataFrame | None = None,
                  time_limit: float = 600.0,
                  min_exclusive: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Assign whole clusters to models. Returns (assignments, coverage by model_name).

    min_exclusive, one fraction per model, is the share of clustered molecules each model must
    cover alone. Raises NoAssignment if the solver finds no assignment.

    assignments has one row per (model, cluster). pattern_clusters (one row per pattern with
    `count`, `cluster_id` and `cluster_memberships`) is needed to count molecules that sit in
    two clusters once; without it the clusters are treated as disjoint, sized by
    cluster_weight.
    """
    cluster_ids = [int(c) for c in cluster_summary['cluster_id']]
    column = {c: i for i, c in enumerate(cluster_ids)}
    if pattern_clusters is None or pattern_clusters.empty:
        atom_ids = [(c,) for c in cluster_ids]
        atom_counts = cluster_summary['cluster_weight'].astype(float).to_numpy()
    else:
        atom_ids, atom_counts = build_atoms(pattern_clusters)
        unknown = sorted({c for atom in atom_ids for c in atom} - set(cluster_ids))
        if unknown:
            raise ValueError(f'pattern memberships name clusters {unknown} that are not in the '
                             f'cluster summary; re-run label_clusters.py on the current '
                             f'pattern_clusters.csv.')
    atom_sets = [tuple(column[c] for c in atom) for atom in atom_ids]

    targets = model_specs['target_coverage'].astype(float).to_numpy()
    assign, optimal, message = solve_assignment(atom_sets, atom_counts, len(cluster_ids),
                                                targets, time_limit, min_exclusive)
    if assign is None:
        raise NoAssignment(f'the solver found no assignment: {message}',
                           infeasible='infeasible' in str(message).lower())
    if not optimal:
        print(f'WARNING: the solver stopped before proving optimality ({message}); using the '
              f'best assignment it found.')

    achieved = coverage(assign, atom_sets, atom_counts)
    alone = exclusive_coverage(assign, atom_sets, atom_counts)
    print('Exclusive share per model (molecules no other model covers): '
          + ', '.join(f'{name} {100 * x:.1f}%' for name, x in zip(model_specs['model_name'], alone)))
    n_models = assign.sum(axis=0)
    rows = []
    for m, spec in model_specs.reset_index(drop=True).iterrows():
        for c in np.flatnonzero(assign[m]):
            cluster = cluster_summary.iloc[c]
            rows.append({
                'model_name': spec['model_name'],
                'cluster_id': cluster_ids[c],
                'cluster_weight': int(cluster['cluster_weight']),
                'centroid_fgs': cluster['centroid_fgs'],
                'centroid_n_fgs': int(cluster['centroid_n_fgs']),
                'n_models_for_cluster': int(n_models[c]),
                'was_reused': bool(n_models[c] > 1),
                'target_coverage': float(spec['target_coverage']),
            })
    assignments = pd.DataFrame(rows)
    return assignments, pd.Series(achieved, index=model_specs['model_name'].to_numpy())


def build_report(assignments: pd.DataFrame, achieved: pd.Series, model_specs: pd.DataFrame,
                 total_weight: int, tolerance: float | None = None) -> pd.DataFrame:
    """One row per model in model_specs, including models that received no cluster.

    effective_coverage is the model's share of unique clustered molecules; coverage_gap is
    target - effective_coverage. With a tolerance, within_tolerance says whether
    |coverage_gap| <= tolerance.
    """
    rows = []
    for _, spec in model_specs.iterrows():
        model = spec['model_name']
        group = assignments[assignments['model_name'] == model] if len(assignments) else assignments
        fgs = sorted({fg for value in group.get('centroid_fgs', [])
                      for fg in str(value).split(',') if fg and fg != 'nan'})
        cov = float(achieved.get(model, 0.0))
        target = float(spec['target_coverage'])
        rows.append({
            'model_name': model,
            'clusters_used': len(group),
            'clusters_reused': int(group['was_reused'].sum()) if len(group) else 0,
            'total_cluster_weight': int(group['cluster_weight'].sum()) if len(group) else 0,
            'n_molecules': int(round(cov * total_weight)),
            'effective_coverage': round(cov, 4),
            'target_coverage': target,
            'coverage_gap': round(target - cov, 4),
            **({'within_tolerance': bool(abs(target - cov) <= tolerance + 1e-12)}
               if tolerance is not None else {}),
            'fgs_used': ','.join(fgs),
            'n_fgs_used': len(fgs),
        })
    return pd.DataFrame(rows)


def too_coarse_message(report: pd.DataFrame, cluster_summary: pd.DataFrame,
                       total_weight: int, tolerance: float) -> str:
    """Explain which models miss their target and that the clustering needs more clusters."""
    lines = [f'The clusters are too coarse to meet the coverage targets within '
             f'+/-{100 * tolerance:.1f} percentage points using whole clusters:', '']
    for _, row in report[~report['within_tolerance']].iterrows():
        lines.append(f"  {row['model_name']}: target {100 * row['target_coverage']:.1f}%, "
                     f"best achievable {100 * row['effective_coverage']:.1f}% "
                     f"({-100 * row['coverage_gap']:+.1f} points)")
    shares = cluster_summary.set_index('cluster_id')['cluster_weight'] / total_weight
    lines += ['', 'Cluster sizes (% of clustered molecules): '
              + ', '.join(f'{cid}: {100 * share:.1f}' for cid, share in shares.items())]
    ceiling = float(report['target_coverage'].max()) + tolerance
    for cid, share in shares.items():
        if share > ceiling:
            lines.append(f'  cluster {cid} holds {100 * share:.1f}% of molecules, more than any '
                         f'model may cover ({100 * ceiling:.1f}%), so it overshoots wherever '
                         f'it goes.')
    lines += ['', f'Re-run build_pattern_clusters.py with a larger --max-components (this run '
                  f'has {len(cluster_summary)} clusters), then label_clusters.py and this script.',
              'Components whose mixing weight ends below the pruning threshold in '
              'bernoulli_mixture_clustering.py are dropped, so a larger --max-components alone '
              'may not add clusters; lower that threshold as well if the cluster count does not '
              'grow.']
    return '\n'.join(lines)


def build_lineage_map(assignments: pd.DataFrame, pattern_clusters: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, cluster, pattern) for EVERY pattern in each assigned cluster.

    A pattern belongs to each cluster in its `cluster_memberships`, primary or secondary,
    and label_clusters.py counts it in every one of them. Joining on `cluster_id` alone
    (the primary cluster) would leave out the molecules a cluster holds as secondary members,
    and drop clusters that have no primary members at all.

    Columns: model_name, cluster_id, pattern_index, is_primary, member_cids.
    """
    pairs = assignments[['model_name', 'cluster_id']].drop_duplicates()
    needed = {'pattern_index', 'cluster_id', 'member_cids'}
    if pattern_clusters.empty or not needed.issubset(pattern_clusters.columns):
        return pairs

    memberships = (pattern_clusters['cluster_memberships']
                   if 'cluster_memberships' in pattern_clusters.columns
                   else pattern_clusters['cluster_id'])
    long = pattern_clusters[['pattern_index', 'member_cids']].copy()
    long['member_of'] = [parse_memberships(v, c)
                         for v, c in zip(memberships, pattern_clusters['cluster_id'])]
    primary = (pattern_clusters['cluster_primary']
               if 'cluster_primary' in pattern_clusters.columns
               else pattern_clusters['cluster_id']).astype(int)
    long['primary'] = primary.to_numpy()
    long = long.explode('member_of').rename(columns={'member_of': 'cluster_id'})
    long['cluster_id'] = long['cluster_id'].astype(int)
    long['is_primary'] = long['cluster_id'] == long['primary']

    lineage = pairs.merge(long[['cluster_id', 'pattern_index', 'is_primary', 'member_cids']],
                          on='cluster_id', how='inner')
    return lineage.sort_values(['model_name', 'cluster_id', 'pattern_index']).reset_index(drop=True)


def model_molecule_counts(lineage: pd.DataFrame, model_specs: pd.DataFrame,
                          total_molecules: int) -> pd.DataFrame | None:
    """Unique molecules per model, counted by cid from the lineage map.

    One row per model in model_specs, then an 'ALL MODELS' row:
      n_unique_molecules     distinct cids in the model's clusters; a molecule in two of the
                             model's clusters counts once
      coverage               n_unique_molecules / all clustered molecules
      n_exclusive_molecules  of those, molecules in no other model
      n_shared_molecules     of those, molecules also in at least one other model
      n_clusters, n_patterns clusters and FG patterns behind the model
    Returns None when the lineage map has no member_cids.
    """
    if 'member_cids' not in lineage.columns:
        return None
    rows = lineage.dropna(subset=['member_cids'])
    long = (rows.assign(cid=rows['member_cids'].astype(str).str.split(','))
            .explode('cid')[['model_name', 'cluster_id', 'pattern_index', 'cid']])
    long['cid'] = long['cid'].str.strip()
    pairs = long[['model_name', 'cid']].drop_duplicates()
    models_per_cid = pairs.groupby('cid')['model_name'].nunique()
    pairs = pairs.assign(n_models=pairs['cid'].map(models_per_cid).to_numpy())

    out = []
    for _, spec in model_specs.iterrows():
        model = spec['model_name']
        mine = pairs[pairs['model_name'] == model]
        used = long[long['model_name'] == model]
        out.append({
            'model_name': model,
            'target_coverage': float(spec['target_coverage']),
            'n_unique_molecules': len(mine),
            'coverage': round(len(mine) / total_molecules, 4) if total_molecules else 0.0,
            'n_exclusive_molecules': int((mine['n_models'] == 1).sum()),
            'n_shared_molecules': int((mine['n_models'] > 1).sum()),
            'n_clusters': int(used['cluster_id'].nunique()),
            'n_patterns': int(used['pattern_index'].nunique()),
        })
    union = int(pairs['cid'].nunique())
    out.append({
        'model_name': 'ALL MODELS',
        'target_coverage': float(model_specs['target_coverage'].sum()),
        'n_unique_molecules': union,
        'coverage': round(union / total_molecules, 4) if total_molecules else 0.0,
        'n_exclusive_molecules': int((models_per_cid == 1).sum()),
        'n_shared_molecules': int((models_per_cid > 1).sum()),
        'n_clusters': int(long['cluster_id'].nunique()),
        'n_patterns': int(long['pattern_index'].nunique()),
    })
    return pd.DataFrame(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tolerance', type=float, default=0.02,
                        help='allowed |coverage - target| per model, as a fraction '
                             '(default 0.02 = 2 percentage points)')
    parser.add_argument('--time-limit', type=float, default=600.0,
                        help='solver time limit in seconds (default 600)')
    parser.add_argument('--min-exclusive', type=float, default=0.01,
                        help='share of clustered molecules each model must cover alone '
                             '(default 0.01 = 1%%; 0 turns the rule off). A min_exclusive column '
                             'in model_specs.csv sets it per model')
    args = parser.parse_args(argv)

    cluster_summary = pd.read_csv(CLUSTER_SUMMARY)
    model_specs = pd.read_csv(MODEL_SPECS)
    # A blank or non-numeric target (e.g. a template whose coverage is not filled in yet) would
    # otherwise reach the solver as NaN and fail with an unclear message.
    if 'target_coverage' not in model_specs.columns:
        print(f'ERROR: {MODEL_SPECS.name} has no target_coverage column.', file=sys.stderr)
        return 2
    targets = pd.to_numeric(model_specs['target_coverage'], errors='coerce')
    bad = model_specs.loc[targets.isna() | (targets < 0) | (targets > 1), 'model_name'].tolist()
    if bad:
        print(f'ERROR: {MODEL_SPECS.name}: target_coverage must be a fraction between 0 and 1 for '
              f'every model; missing or invalid for {bad}.', file=sys.stderr)
        return 2
    model_specs['target_coverage'] = targets
    pattern_file = OUTPUT_DIR / 'pattern_clusters.csv'
    pattern_clusters = pd.read_csv(pattern_file, dtype=TEXT_COLUMNS) if pattern_file.exists() else pd.DataFrame()
    if pattern_clusters.empty:
        print(f'WARNING: {pattern_file.name} not found; treating clusters as disjoint, so '
              f'molecules in two clusters may be counted twice.')
        total_weight = int(cluster_summary['cluster_weight'].sum())
    else:
        total_weight = int(pattern_clusters['count'].sum())
    if {'min_fgs', 'max_fgs'} & set(model_specs.columns):
        print('note: min_fgs / max_fgs in model_specs.csv are ignored; any cluster may go to '
              'any model.')

    min_exclusive = np.full(len(model_specs), args.min_exclusive, dtype=float)
    if 'min_exclusive' in model_specs.columns:
        per_model = pd.to_numeric(model_specs['min_exclusive'], errors='coerce').to_numpy()
        min_exclusive = np.where(np.isnan(per_model), min_exclusive, per_model)
    if min_exclusive.sum() > 1:
        print(f'ERROR: the exclusive shares sum to {100 * min_exclusive.sum():.1f}% of molecules, '
              f'more than all of them.', file=sys.stderr)
        return 2

    print(f'Clustered molecules: {total_weight}')
    print(f'Models to fit: {len(model_specs)}  (targets sum to '
          f'{100 * model_specs["target_coverage"].sum():.0f}% of molecules)')
    print()

    print('Exclusive share required per model: '
          + ', '.join(f'{name} {100 * e:.1f}%' for name, e in zip(model_specs['model_name'], min_exclusive)))
    try:
        assignments, achieved = assign_models(cluster_summary, model_specs, pattern_clusters,
                                              time_limit=args.time_limit,
                                              min_exclusive=min_exclusive)
    except NoAssignment as err:
        if not err.infeasible:
            print(f'ERROR: {err}', file=sys.stderr)
            return 1
        for stale in (MODEL_ASSIGNMENTS, MODEL_REPORT, PATTERN_MODEL_MAP, MODEL_MOLECULE_COUNTS):
            if stale.exists():
                stale.unlink()  # from an earlier run; would not match these clusters
        message = (f'No whole-cluster assignment gives every model its exclusive share '
                   f'({", ".join(f"{100 * e:.1f}%" for e in min_exclusive)}) with these '
                   f'{len(cluster_summary)} clusters.\n\n'
                   f'Re-run build_pattern_clusters.py with a larger --max-components, then '
                   f'label_clusters.py and this script; or lower --min-exclusive.')
        ASSIGNMENT_WARNINGS.write_text(message + '\n')
        print(message, file=sys.stderr)
        print(f'(saved to {ASSIGNMENT_WARNINGS}; exiting with status {EXIT_NEEDS_MORE_CLUSTERS})',
              file=sys.stderr)
        return EXIT_NEEDS_MORE_CLUSTERS
    assignments.to_csv(MODEL_ASSIGNMENTS, index=False)

    lineage = build_lineage_map(assignments, pattern_clusters)
    lineage.to_csv(PATTERN_MODEL_MAP, index=False)

    report = build_report(assignments, achieved, model_specs, total_weight,
                          tolerance=args.tolerance)
    report.to_csv(MODEL_REPORT, index=False)

    shared = assignments.drop_duplicates('cluster_id')['was_reused'].sum()
    print(f'Wrote {len(assignments)} assignments to {MODEL_ASSIGNMENTS}')
    print(f'{int(shared)} of {len(cluster_summary)} clusters are assigned to more than one model')
    print(f'Wrote {len(report)} model reports to {MODEL_REPORT}')
    print(f'Wrote pattern-to-model lineage map to {PATTERN_MODEL_MAP}')
    print()
    print(report.drop(columns=['fgs_used']).to_string(index=False))

    counts = model_molecule_counts(lineage, model_specs, total_weight)
    if counts is not None:
        counts['exclusive_share'] = (counts['n_exclusive_molecules'] / total_weight).round(4)
        counts['min_exclusive'] = list(min_exclusive) + [float(min_exclusive.sum())]
    if counts is None:
        print(f'\nnote: {pattern_file.name} has no member_cids, so {MODEL_MOLECULE_COUNTS.name} '
              f'was not written.')
    else:
        counts.to_csv(MODEL_MOLECULE_COUNTS, index=False)
        print(f'\nUnique molecules per model (by cid), written to {MODEL_MOLECULE_COUNTS}:')
        print(counts.to_string(index=False))
        union = int(counts['n_unique_molecules'].iloc[-1])
        if union != total_weight:
            print(f'note: {union} distinct cids across all models, but {total_weight} clustered '
                  f'molecules. Duplicate cids in the input, or clusters that reached no model, '
                  f'cause this.')

    if report['within_tolerance'].all():
        if ASSIGNMENT_WARNINGS.exists():
            ASSIGNMENT_WARNINGS.unlink()  # stale, from an earlier run
        print(f'\nAll models are within +/-{100 * args.tolerance:.1f} points of their target.')
        return 0

    message = too_coarse_message(report, cluster_summary, total_weight, args.tolerance)
    ASSIGNMENT_WARNINGS.write_text(message + '\n')
    print('\n' + '=' * 78, file=sys.stderr)
    print(message, file=sys.stderr)
    print('=' * 78, file=sys.stderr)
    print(f'(saved to {ASSIGNMENT_WARNINGS}; exiting with status {EXIT_NEEDS_MORE_CLUSTERS})',
          file=sys.stderr)
    return EXIT_NEEDS_MORE_CLUSTERS


if __name__ == '__main__':
    sys.exit(main())
