#!/usr/bin/env python3
"""Extract every SMILES string from .jsonl record(s) into a JSON array.

    python smiles_to_json.py processed_Compound_176000001_176500000.jsonl -o smiles.json

Defaults to the `smiles` field, which preserves stereochemistry. `c-smiles` is
PubChem's Canonical SMILES and has stereo stripped -- see README_pipeline.md.
Order is preserved and duplicates are kept unless --dedupe is passed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", type=Path, nargs="+", help=".jsonl file(s)")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output .json path")
    ap.add_argument("--field", default="smiles",
                    help="JSON key to pull (default: smiles -- keeps stereochemistry)")
    ap.add_argument("--dedupe", action="store_true",
                    help="drop repeats, keeping first-seen order")
    ap.add_argument("--indent", type=int, default=None,
                    help="pretty-print with this indent (default: compact, one line)")
    args = ap.parse_args()

    smiles: list[str] = []
    seen: set[str] = set()
    n_lines = n_bad = n_missing = n_dup = 0

    for path in args.inputs:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                val = rec.get(args.field)
                if not val:
                    n_missing += 1
                    continue
                if args.dedupe:
                    if val in seen:
                        n_dup += 1
                        continue
                    seen.add(val)
                smiles.append(val)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as w:
        json.dump(smiles, w, indent=args.indent, ensure_ascii=False)

    print(f"records read      {n_lines:,}")
    print(f"smiles extracted  {len(smiles):,}   (field {args.field!r})")
    if n_missing:
        print(f"field missing     {n_missing:,}")
    if n_bad:
        print(f"unparseable JSON  {n_bad:,}")
    if args.dedupe:
        print(f"duplicates removed {n_dup:,}")
    print(f"wrote             {args.out}  ({args.out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
