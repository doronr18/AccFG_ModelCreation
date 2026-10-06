import json
import sys
from pathlib import Path

STAGE1_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data' / 'stage1'
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

import extract_smiles  # noqa: E402


def test_extract_writes_cid_and_smiles(tmp_path):
    src = tmp_path / 'part.jsonl'
    src.write_text('\n'.join(json.dumps(r) for r in [
        {'cid': '176000001', 'smiles': 'CCO', 'c-smiles': 'CCO'},
        {'cid': 176000002, 'smiles': 'C/C=C\\C'},   # bare-number cid; escaped backslash
        {'smiles': 'CCN'},                          # no cid: skipped and counted
    ]) + '\n')
    out = tmp_path / 'part.smi'

    stats = extract_smiles.extract_one((str(src), str(out), 'smiles', 'cid', False, 1.0, 0))

    assert out.read_text().splitlines() == ['176000001\tCCO', '176000002\tC/C=C\\C']
    assert stats['written'] == 2
    assert stats['no_id'] == 1
