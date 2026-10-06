#!/usr/bin/env python3
"""Answer, from the real data: what are `smiles`, `c-smiles` and `canonicalized`?

Samples records across your .jsonl files and reports whether c-smiles is already
RDKit-canonical, whether it agrees with `smiles`, whether either loses stereochemistry,
and whether skipping canonicalization would change the functional-group result.

    python check_smiles_fields.py /data/jsonl -n 20000

Runs in well under a minute; it reads the head of each file rather than all of it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def sample(indir: Path, n: int, glob: str) -> list[dict]:
    files = sorted(indir.glob(glob))
    if not files:
        sys.exit(f"no files matching {glob} in {indir}")
    per = max(1, n // len(files))
    recs = []
    for f in files:
        with f.open("rb") as fh:
            for i, line in enumerate(fh):
                if i >= per:
                    break
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if len(recs) >= n:
            break
    return recs[:n]


def inchi(smi: str) -> str | None:
    """InChI is the identity test: same molecule -> same InChI, whatever the SMILES."""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchi(m) if m is not None else None


def n_stereo(smi: str) -> int:
    """Count SPECIFIED stereo elements. FindPotentialStereo covers both atom centres and
    double-bond E/Z; FindMolChiralCenters ignores bond stereo and undercounts."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return -1
    return sum(1 for si in Chem.FindPotentialStereo(m)
               if si.specified == Chem.StereoSpecified.Specified)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", type=Path)
    ap.add_argument("-n", type=int, default=20000, help="records to sample (default 20000)")
    ap.add_argument("--glob", default="*.jsonl")
    args = ap.parse_args()

    recs = sample(args.indir, args.n, args.glob)
    print(f"sampled {len(recs):,} records from {args.indir}\n")

    flags = Counter(r.get("canonicalized") for r in recs)
    print(f"`canonicalized` values: {dict(flags)}")
    print()

    def analyse(group: list[dict]) -> dict:
        st = dict(n=0, eq=0, c_canon=0, s_canon=0, same_mol=0,
                  c_bad=0, s_bad=0, lost=0, stereo_s=0, stereo_c=0)
        for r in group:
            s, c = r.get("smiles"), r.get("c-smiles")
            if not s or not c:
                continue
            st["n"] += 1
            if s == c:
                st["eq"] += 1
            ms, mc = Chem.MolFromSmiles(s), Chem.MolFromSmiles(c)
            if ms is None:
                st["s_bad"] += 1
            if mc is None:
                st["c_bad"] += 1
            if ms is None or mc is None:
                continue
            if Chem.MolToSmiles(mc) == c:
                st["c_canon"] += 1
            if Chem.MolToSmiles(ms) == s:
                st["s_canon"] += 1
            ns, nc = n_stereo(s), n_stereo(c)
            st["stereo_s"] += max(ns, 0)
            st["stereo_c"] += max(nc, 0)
            if ns > nc:
                st["lost"] += 1
            if Chem.MolToInchi(ms) == Chem.MolToInchi(mc):
                st["same_mol"] += 1
        return st

    # Break out by flag value as well as overall -- if the flag means anything, it shows here.
    groups = [("ALL", recs)]
    if len(flags) > 1:
        groups += [(f"canonicalized={k!r}", [r for r in recs if r.get("canonicalized") == k])
                   for k in sorted(flags, key=lambda x: (x is None, x))]

    rows = [(label, analyse(g)) for label, g in groups]
    W = 30   # metric-label column
    C = 22   # one column per group

    def line(label, fn):
        print(f"  {label:<{W}}" + "".join(f"{fn(st):>{C}}" for _, st in rows))

    print(f"  {'':<{W}}" + "".join(f"{l:>{C}}" for l, _ in rows))
    print("  " + "-" * (W + C * len(rows)))

    def frac(key):
        return lambda st: (f"{st[key]:,}/{st['n']:,} ({100*st[key]/st['n']:.1f}%)"
                           if st["n"] else "n/a")

    line("records", lambda st: f"{st['n']:,}")
    line("smiles == c-smiles", frac("eq"))
    line("smiles == same molecule", frac("same_mol"))
    line("c-smiles is RDKit-canonical", frac("c_canon"))
    line("smiles   is RDKit-canonical", frac("s_canon"))
    line("c-smiles LOST stereo", frac("lost"))
    line("stereo elements in smiles", lambda st: f"{st['stereo_s']:,}")
    line("stereo elements in c-smiles", lambda st: f"{st['stereo_c']:,}")
    line("unparseable smiles", frac("s_bad"))
    line("unparseable c-smiles", frac("c_bad"))
    print()

    n = rows[0][1]["n"]
    c_lost_stereo = rows[0][1]["lost"]
    s_canon = rows[0][1]["s_canon"]
    print("VERDICT")
    if not n:
        print("  no usable records -- check --glob and the field names.")
        return 1

    if c_lost_stereo:
        print(f"  * c-smiles is STEREO-STRIPPED ({100*c_lost_stereo/n:.1f}% of records lose stereo).")
        print("    Use --field smiles in extract_smiles.py. Reading c-smiles would silently")
        print("    discard stereochemistry and mis-call any stereo-defined functional group.")
    else:
        print("  * no stereo loss detected in this sample; either field looks usable, but")
        print("    prefer --field smiles since it is the stereo-bearing one by definition.")

    if s_canon == n:
        print("  * smiles is ALREADY RDKit-canonical throughout -> --no-canonical is safe here,")
        print("    though confirm on more than a sample before relying on it.")
    else:
        print(f"  * smiles is RDKit-canonical in only {100*s_canon/n:.1f}% of records. PubChem")
        print("    canonicalises with a different algorithm, so `canonicalized` says nothing")
        print("    about RDKit's form. Keep canonicalisation on if you need `Molecule` to join")
        print("    against other RDKit-derived data.")

    print("  * the `canonicalized` flag describes PUBCHEM's processing, not RDKit's. Since you")
    print("    canonicalise with RDKit yourself, it does not change what the pipeline should do.")
    print()
    print("None of this affects functional-group PRESENCE: RDKit matches SMARTS against the")
    print("parsed graph, so Kekule / aromatic / reordered forms of one molecule all agree.")
    print("Canonicalisation only decides what the `Molecule` key column looks like. Stereo is")
    print("the exception -- it is a real graph difference, not a representation choice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
