from __future__ import annotations

from typing import Iterable, List

import pandas as pd

try:
    from accfg.main import AccFG
    from accfg.spreadsheet import fg_presence_vector, canonical_smiles
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from accfg.main import AccFG
    from accfg.spreadsheet import fg_presence_vector, canonical_smiles


def fg_pattern_string(afg: AccFG, smiles: str, canonical: bool = True) -> str:
    """Return the full FG pattern as a binary string of 0/1 values."""
    if canonical:
        smiles = canonical_smiles(smiles)

    vector = fg_presence_vector(afg, smiles, canonical=False)
    return ''.join('1' if vector.get(fg_name, 0) else '0' for fg_name in afg.dict_fgs.keys())


def fg_pattern_bitmask(afg: AccFG, smiles: str, canonical: bool = True) -> int:
    """Compatibility helper: convert the binary string pattern to an integer bitmask."""
    return int(fg_pattern_string(afg, smiles, canonical=canonical), 2)


def fg_presence_rows(afg: AccFG, smiles_list: Iterable[str], canonical: bool = True, cid_values: Iterable[int] | None = None) -> List[dict]:
    """Return one row per molecule with the FG 0/1 vector, pattern index, and exact cid."""
    pattern_counts = pattern_count_dictionary(afg, smiles_list, canonical=canonical)
    pattern_index = {pattern: idx for idx, pattern in enumerate(pattern_counts.keys())}

    rows = []
    cid_source = list(cid_values) if cid_values is not None else list(range(len(list(smiles_list))))
    for idx, smiles in enumerate(smiles_list):
        canon_smi = canonical_smiles(smiles) if canonical else smiles
        vector = fg_presence_vector(afg, canon_smi, canonical=False)
        pattern = fg_pattern_string(afg, canon_smi, canonical=False)
        cid = cid_source[idx] if idx < len(cid_source) else idx
        row = {'cid': cid, 'Molecule': canon_smi}
        row.update(vector)
        row['pattern_index'] = pattern_index[pattern]
        rows.append(row)
    return rows


def pattern_count_dictionary(afg: AccFG, smiles_list: Iterable[str], canonical: bool = True) -> dict:
    """Count how many molecules share each FG pattern string."""
    counts = {}
    for smiles in smiles_list:
        pattern = fg_pattern_string(afg, smiles, canonical=canonical)
        counts[pattern] = counts.get(pattern, 0) + 1
    return dict(sorted(counts.items()))


def pattern_count_dataframe(afg: AccFG, smiles_list: Iterable[str], canonical: bool = True) -> pd.DataFrame:
    counts = pattern_count_dictionary(afg, smiles_list, canonical=canonical)
    rows = []
    for idx, (pattern, count) in enumerate(counts.items()):
        rows.append({'pattern_index': idx, 'pattern': pattern, 'count': int(count)})
    df = pd.DataFrame(rows)
    return df.sort_values('pattern_index').reset_index(drop=True)


def save_pattern_count_csv(afg: AccFG, smiles_list: Iterable[str], output_csv: str, canonical: bool = True):
    df = pattern_count_dataframe(afg, smiles_list, canonical=canonical)
    df.to_csv(output_csv, index=False)
    return df
