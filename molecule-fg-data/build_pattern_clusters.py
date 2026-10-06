from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accfg import AccFG
from patterns import pattern_count_dataframe
from bernoulli_mixture_clustering import cluster_pattern_counts_overlapping, describe_clusters

ROOT = PROJECT_ROOT
DATA_DIR = ROOT / 'molecule-fg-data'
OUTPUT_DIR = DATA_DIR / 'csv_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

FG_PRESENCE = OUTPUT_DIR / 'fg_presence.csv'
SMILES_JSON = DATA_DIR / 'smiles.json'
SAMPLE_DATASET = DATA_DIR / 'pubchem_like_sample_120.csv'
PATTERN_OUTPUT = OUTPUT_DIR / 'pubchem_like_pattern_counts.csv'
CLUSTER_OUTPUT = OUTPUT_DIR / 'pattern_clusters.csv'


def load_smiles_list(dataset_path: Path):
    if dataset_path.suffix.lower() == '.json':
        payload = json.loads(dataset_path.read_text())
        if isinstance(payload, list):
            return [str(s).strip() for s in payload if str(s).strip()]
        if isinstance(payload, dict) and 'smiles' in payload and isinstance(payload['smiles'], list):
            return [str(s).strip() for s in payload['smiles'] if str(s).strip()]
        raise ValueError(f'Unsupported JSON structure in {dataset_path}')

    dataset = pd.read_csv(dataset_path)
    return dataset['smiles'].astype(str).tolist()


def main(max_components: int = 12, tau: float = 0.3, top_n: int | None = 2, seed: int = 0):
    source_path = SMILES_JSON if SMILES_JSON.exists() else SAMPLE_DATASET
    afg = AccFG(print_load_info=False, lite=False)
    smiles_list = load_smiles_list(source_path)

    pattern_df = pattern_count_dataframe(afg, smiles_list, canonical=True)
    pattern_df.to_csv(PATTERN_OUTPUT, index=False)

    clustered, means, weights = cluster_pattern_counts_overlapping(
        pattern_df,
        max_components=max_components,
        tau=tau,
        top_n=top_n,
        seed=seed,
        verbose=True,
    )

    canonical_df = clustered.copy().sort_values(['cluster_id', 'pattern_index']).reset_index(drop=True)
    if FG_PRESENCE.exists():
        fg_presence = pd.read_csv(FG_PRESENCE)
        if {'cid', 'pattern_index'}.issubset(fg_presence.columns):
            pattern_cids = (
                fg_presence.groupby('pattern_index', sort=False)['cid']
                .agg(lambda s: ','.join(str(int(x)) for x in sorted(pd.unique(s))))
                .rename('member_cids')
            )
            canonical_df['member_cids'] = canonical_df['pattern_index'].map(pattern_cids)

    canonical_df.to_csv(CLUSTER_OUTPUT, index=False)

    fg_names = list(afg.dict_fgs.keys())
    primary_clusters = sorted(clustered['cluster_id'].unique())
    for cid in primary_clusters:
        rep = clustered[clustered['cluster_id'] == cid]['cluster_representative'].iloc[0]
        fgs = [fg_names[i] for i, bit in enumerate(rep) if bit == '1']
        weight = clustered[clustered['cluster_id'] == cid]['cluster_weight'].sum()
        print(f'Cluster {cid} (weight={int(weight)}): {fgs[:20]}')

    describe_clusters(means, weights, fg_names, top_k=8)

    print(f'Wrote {len(pattern_df)} unique patterns to {PATTERN_OUTPUT}')
    print(f'Wrote {len(canonical_df)} clustered pattern rows to {CLUSTER_OUTPUT}')
    print(canonical_df.head(10).to_string(index=False))


if __name__ == '__main__':
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description='Automatic-K overlapping clustering of binary FG patterns using a Bernoulli mixture.')
    parser.add_argument('--max-components', type=int, default=12, help='Upper bound on K; weak components are pruned.')
    parser.add_argument('--tau', type=float, default=0.3, help='Posterior threshold for extra membership assignments.')
    parser.add_argument('--top-n', type=int, default=2, help='Maximum number of clusters a pattern can join.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for deterministic initialization.')
    parser.add_argument('--k', type=int, default=None, help='Deprecated alias for max-components.')
    args = parser.parse_args()

    if args.k is not None:
        max_components = args.k
    else:
        max_components = args.max_components

    main(max_components=max_components, tau=args.tau, top_n=args.top_n, seed=args.seed)
