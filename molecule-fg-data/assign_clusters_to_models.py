from __future__ import annotations

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

REUSE_DECAY = 0.5
OVERLAP_TOLERANCE = 0.5


def assign_models(cluster_summary: pd.DataFrame,
                  model_specs: pd.DataFrame,
                  reuse_decay: float = REUSE_DECAY,
                  overlap_tolerance: float = OVERLAP_TOLERANCE,
                  total_weight: int | None = None) -> pd.DataFrame:
    """
    Assign clusters to models based on target coverage and rules.
    Returns a long-form DataFrame: one row per (model, cluster).

    A final mandatory pass ensures every cluster is used at least once.
    If a cluster falls outside the min/max window for all models, it is
    assigned to the closest model by FG-count distance.

    total_weight is the unique molecule total used as the coverage denominator. This is
    usually the sum of the unique pattern counts, not the sum of overlapping cluster weights.
    """
    if total_weight is None:
        total_weight = int(cluster_summary['cluster_weight'].sum())
    cluster_use_count = {cid: 0 for cid in cluster_summary['cluster_id']}
    assignments = []
    assigned_cluster_ids = set()
    model_coverage_used = {
        spec['model_name']: 0.0 for _, spec in model_specs.iterrows()
    }

    for _, spec in model_specs.iterrows():
        model = spec['model_name']
        target = float(spec['target_coverage'])
        min_fgs = int(spec.get('min_fgs', 1))
        max_fgs = int(spec.get('max_fgs', 999))
        target_weight = target * total_weight

        candidates = cluster_summary[
            (cluster_summary['centroid_n_fgs'] >= min_fgs) &
            (cluster_summary['centroid_n_fgs'] <= max_fgs)
        ].copy()

        reuse_counts = candidates['cluster_id'].map(cluster_use_count).astype(int)
        candidates = candidates.copy()
        candidates['effective_weight'] = (
            candidates['cluster_weight'].astype(float) * (reuse_decay ** reuse_counts.to_numpy())
        )

        candidates = candidates.sort_values(
            ['effective_weight', 'centroid_n_fgs'],
            ascending=[False, False],
        )

        cumulative = model_coverage_used[model]
        reused_weight = 0.0

        for _, row in candidates.iterrows():
            if cumulative >= target_weight:
                break

            cid = int(row['cluster_id'])
            w = float(row['effective_weight'])
            remaining_capacity = max(0.0, target_weight - cumulative)
            if remaining_capacity <= 0:
                break
            w = min(w, remaining_capacity)
            was_reused = cluster_use_count[cid] > 0

            if was_reused and reused_weight + w > overlap_tolerance * target_weight:
                continue

            assignments.append({
                'model_name': model,
                'cluster_id': cid,
                'cluster_weight': int(row['cluster_weight']),
                'effective_weight': round(w, 2),
                'centroid_fgs': row['centroid_fgs'],
                'centroid_n_fgs': int(row['centroid_n_fgs']),
                'was_reused': was_reused,
                'forced_assignment': False,
                'target_coverage': target,
            })

            cumulative += w
            model_coverage_used[model] = cumulative
            if was_reused:
                reused_weight += w
            cluster_use_count[cid] += 1
            assigned_cluster_ids.add(cid)

    for _, row in cluster_summary.iterrows():
        cid = int(row['cluster_id'])
        if cid in assigned_cluster_ids:
            continue

        model_candidates = []
        for _, spec in model_specs.iterrows():
            model_name = spec['model_name']
            target = float(spec['target_coverage'])
            target_weight = target * total_weight
            remaining_capacity = max(0.0, target_weight - model_coverage_used[model_name])
            distance = 0 if int(spec['min_fgs']) <= int(row['centroid_n_fgs']) <= int(spec['max_fgs']) else min(
                abs(int(row['centroid_n_fgs']) - int(spec['min_fgs'])),
                abs(int(row['centroid_n_fgs']) - int(spec['max_fgs'])),
            )
            model_candidates.append((distance, -remaining_capacity, model_name, remaining_capacity))

        valid_candidates = [m for m in model_candidates if m[3] > 0]
        if valid_candidates:
            best = min(valid_candidates, key=lambda x: (x[0], x[1]))
            model_name = best[2]
            remaining_capacity = best[3]
            target = float(model_specs.loc[model_specs['model_name'] == model_name, 'target_coverage'].iloc[0])
            effective_weight = min(float(row['cluster_weight']), remaining_capacity)
        else:
            best = min(model_candidates, key=lambda x: (x[0], x[1]))
            model_name = best[2]
            effective_weight = 0.0
            target = float(model_specs.loc[model_specs['model_name'] == model_name, 'target_coverage'].iloc[0])

        assignments.append({
            'model_name': model_name,
            'cluster_id': cid,
            'cluster_weight': int(row['cluster_weight']),
            'effective_weight': round(effective_weight, 2),
            'centroid_fgs': row['centroid_fgs'],
            'centroid_n_fgs': int(row['centroid_n_fgs']),
            'was_reused': False,
            'forced_assignment': True,
            'target_coverage': target,
        })
        model_coverage_used[model_name] += effective_weight
        cluster_use_count[cid] += 1
        assigned_cluster_ids.add(cid)

    return pd.DataFrame(assignments)


def build_report(assignments: pd.DataFrame, total_weight: int) -> pd.DataFrame:
    """Aggregate per-model coverage, FG set, and overlap stats.

    total_weight should be the unique-molecule denominator rather than the sum of
    overlapping cluster weights, otherwise repeated memberships artificially dilute coverage.
    """
    rows = []
    for model, group in assignments.groupby('model_name'):
        total_effective = group['effective_weight'].sum()
        reused = group[group['was_reused']]
        fgs = sorted({
            fg
            for fgs in group['centroid_fgs']
            for fg in str(fgs).split(',')
            if fg
        })

        rows.append({
            'model_name': model,
            'clusters_used': len(group),
            'clusters_reused': len(reused),
            'total_cluster_weight': int(group['cluster_weight'].sum()),
            'effective_coverage': round(total_effective / total_weight, 4),
            'target_coverage': group['target_coverage'].iloc[0],
            'coverage_gap': round(
                group['target_coverage'].iloc[0] - total_effective / total_weight,
                4,
            ),
            'fgs_used': ','.join(fgs),
            'n_fgs_used': len(fgs),
        })

    return pd.DataFrame(rows)


def main():
    cluster_summary = pd.read_csv(CLUSTER_SUMMARY)
    model_specs = pd.read_csv(MODEL_SPECS)
    pattern_clusters = pd.read_csv(OUTPUT_DIR / 'pattern_clusters.csv') if (OUTPUT_DIR / 'pattern_clusters.csv').exists() else pd.DataFrame()
    total_weight = int(pattern_clusters['count'].sum()) if not pattern_clusters.empty and 'count' in pattern_clusters.columns else int(cluster_summary['cluster_weight'].sum())

    print(f'Total molecule weight across all clusters: {total_weight}')
    print(f'Models to fit: {len(model_specs)}')
    print()

    assignments = assign_models(cluster_summary, model_specs, total_weight=total_weight)
    assignments.to_csv(MODEL_ASSIGNMENTS, index=False)

    pattern_clusters = pd.read_csv(OUTPUT_DIR / 'pattern_clusters.csv') if (OUTPUT_DIR / 'pattern_clusters.csv').exists() else pd.DataFrame()
    if not pattern_clusters.empty and {'pattern_index', 'cluster_id', 'member_cids'}.issubset(pattern_clusters.columns):
        model_map = assignments[['model_name', 'cluster_id']].merge(
            pattern_clusters[['pattern_index', 'cluster_id', 'member_cids']],
            on='cluster_id',
            how='left',
        )
        model_map = model_map.dropna(subset=['pattern_index']).copy()
        model_map.to_csv(PATTERN_MODEL_MAP, index=False)
    else:
        assignments[['model_name', 'cluster_id']].copy().to_csv(PATTERN_MODEL_MAP, index=False)

    report = build_report(assignments, total_weight)
    report.to_csv(MODEL_REPORT, index=False)

    print(f'Wrote {len(assignments)} assignments to {MODEL_ASSIGNMENTS}')
    print(f'Wrote {len(report)} model reports to {MODEL_REPORT}')
    print(f'Wrote pattern-to-model lineage map to {PATTERN_MODEL_MAP}')
    print()
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
