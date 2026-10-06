#!/usr/bin/env python3
"""Stage 1: extract (ID, SMILES) pairs from a directory of .jsonl files into flat text shards.

Each output line is `<cid>\t<smiles>`. The ID is what lets every later stage say which
molecule ended up in which cluster and model; records without one are skipped and counted.

One worker process per input file. Workers write their own output file and return
only counts, so no molecule data ever crosses a process boundary.

Samples a random --fraction of the lines in EVERY file (default 3%), spread across the
whole file rather than taken from the head -- files are in CID order, so the head of a
file is not representative of it. Sampling is seeded per file, so the same command
always selects the same lines regardless of worker count or completion order.

    python extract_smiles.py /data/jsonl -o /data/smiles -j 16              # 3% sample
    python extract_smiles.py /data/jsonl -o /data/smiles -j 16 --fraction 1 # everything
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Matches "<field>": "<value>" capturing the raw (still JSON-escaped) value.
# (?:[^"\\]|\\.)* correctly walks over \" and \\ so SMILES like C/C=C\C don't truncate.
_CACHE: dict[str, re.Pattern[bytes]] = {}


def _pattern(field: str) -> re.Pattern[bytes]:
    if field not in _CACHE:
        _CACHE[field] = re.compile(
            rb'"' + re.escape(field.encode()) + rb'"\s*:\s*"((?:[^"\\]|\\.)*)"'
        )
    return _CACHE[field]


def _id_pattern(field: str) -> re.Pattern[bytes]:
    """Like _pattern, but also accepts a bare JSON number ("cid": 123 as well as "cid": "123")."""
    key = "#id:" + field
    if key not in _CACHE:
        _CACHE[key] = re.compile(
            rb'"' + re.escape(field.encode()) + rb'"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(-?\d+))'
        )
    return _CACHE[key]


def _unescape(raw: bytes) -> str:
    """Decode a captured JSON string body. Fast path for the common no-escape case."""
    if b"\\" not in raw:
        return raw.decode("utf-8")
    return json.loads(b'"' + raw + b'"')


def default_jobs() -> int:
    """Cores this job may use -- NOT os.cpu_count(), which on a cluster node reports the
    whole machine. Under LSF, the slots `bsub -n` granted on this host (LSB_MCPU_HOSTS);
    otherwise the CPU affinity mask, which respects cgroups and taskset.
    (Duplicated in fg_matrix.py so each script stays standalone.)"""
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        avail = os.cpu_count() or 1
    tokens = os.environ.get("LSB_MCPU_HOSTS", "").split()  # "hostA 8 hostB 8"
    if tokens:
        slots = {h.split(".")[0]: int(n) for h, n in zip(tokens[0::2], tokens[1::2])}
        here = socket.gethostname().split(".")[0]
        if len(slots) > 1:
            print(f"WARNING: LSF spread this job over {len(slots)} hosts {slots}. Worker "
                  f"processes can only use this host ({here}); request "
                  f'-R "span[hosts=1]" to get all slots on one node.',
                  file=sys.stderr, flush=True)
        if here in slots:
            return max(1, min(slots[here], avail))
    return max(1, avail)


def _file_rng(seed: int, name: str) -> random.Random:
    """Per-file RNG. crc32, not hash(): str hashing is salted per process, which would
    make the sample change between runs."""
    return random.Random(seed * 1_000_003 + zlib.crc32(name.encode()))


def extract_one(args: tuple[str, str, str, str, bool, float, int]) -> dict:
    """Extract one .jsonl -> one .smi. Runs in a worker process."""
    src_s, out_s, field, id_field, dedupe, fraction, seed = args
    src, out = Path(src_s), Path(out_s)
    pat = _pattern(field)
    id_pat = _id_pattern(id_field)
    draw = _file_rng(seed, src.name).random
    take_all = fraction >= 1.0

    tmp = out.with_suffix(out.suffix + ".partial")
    n_lines = n_sampled = n_found = n_dup = n_noid = 0
    seen: set[str] | None = set() if dedupe else None
    t0 = time.perf_counter()

    with src.open("rb") as fh, tmp.open("w", encoding="utf-8", newline="\n") as w:
        write = w.write
        for line in fh:
            n_lines += 1
            # Draw BEFORE parsing, so the 97% of skipped lines cost almost nothing.
            if not take_all and draw() >= fraction:
                continue
            n_sampled += 1
            m = pat.search(line)
            if m is None:
                continue
            smi = _unescape(m.group(1))
            if not smi:
                continue
            m_id = id_pat.search(line)
            mol_id = ""
            if m_id is not None:
                mol_id = (_unescape(m_id.group(1)) if m_id.group(1) is not None
                          else m_id.group(2).decode()).strip()
            if not mol_id or "\t" in mol_id:
                n_noid += 1
                continue
            if seen is not None:
                if smi in seen:
                    n_dup += 1
                    continue
                seen.add(smi)
            n_found += 1
            write(mol_id)
            write("\t")
            write(smi)
            write("\n")

    os.replace(tmp, out)  # atomic: a complete .smi on disk means that file is done
    return {
        "src": src.name,
        "out": out.name,
        "lines": n_lines,
        "sampled": n_sampled,
        "written": n_found,
        "missing": n_sampled - n_found - n_dup - n_noid,
        "no_id": n_noid,
        "dupes": n_dup,
        "secs": round(time.perf_counter() - t0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", type=Path, help="directory containing .jsonl files")
    ap.add_argument("-o", "--outdir", type=Path, required=True,
                    help="directory for .smi shards")
    ap.add_argument("--field", default="smiles",
                    help="JSON key to pull. Default 'smiles' = PubChem Isomeric SMILES, which "
                         "KEEPS stereochemistry. 'c-smiles' is PubChem Canonical SMILES and has "
                         "stereo STRIPPED -- measured 0 specified stereo elements across a real "
                         "sample vs 32 in `smiles`. Only use c-smiles if you want stereo gone.")
    ap.add_argument("--id-field", default="cid",
                    help="JSON key holding the molecule ID (default 'cid'). Written as the "
                         "first column of every shard line and carried through every later stage")
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="worker processes (default: cores allocated to this job -- under "
                         "LSF, the slots from bsub -n on this host)")
    ap.add_argument("--glob", default="*.jsonl")
    ap.add_argument("--dedupe", action="store_true",
                    help="drop duplicate SMILES WITHIN each shard, keeping the first ID seen. "
                         "The IDs of dropped duplicates do not reach any later stage")
    ap.add_argument("--resume", action="store_true",
                    help="skip inputs whose .smi shard already exists")
    ap.add_argument("--fraction", type=float, default=0.03,
                    help="fraction of lines to sample from EACH file (default 0.03 = 3%%; "
                         "1 = take every line)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed (default 0). Same seed -> same lines selected")
    args = ap.parse_args()
    if args.jobs is None:
        args.jobs = default_jobs()

    if not 0 < args.fraction <= 1:
        print("--fraction must be in (0, 1]", file=sys.stderr)
        return 1

    files = sorted(args.indir.glob(args.glob))
    if not files:
        print(f"no files matching {args.glob} in {args.indir}", file=sys.stderr)
        return 1
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Record the sampling parameters. Resuming into a directory made with different ones
    # would silently mix, e.g., 3% shards with 100% shards.
    params = {"field": args.field, "id_field": args.id_field, "fraction": args.fraction,
              "seed": args.seed, "dedupe": args.dedupe}
    params_file = args.outdir / "_params.json"
    if params_file.exists():
        prev = json.loads(params_file.read_text())
        if prev != params:
            print(f"ERROR: {params_file} was written with {prev},\n"
                  f"       but this run uses {params}. Use a fresh --outdir.",
                  file=sys.stderr)
            return 2
    else:
        params_file.write_text(json.dumps(params, indent=1))

    tasks = []
    skipped = 0
    for f in files:
        out = args.outdir / (f.stem + ".smi")
        if args.resume and out.exists():
            skipped += 1
            continue
        tasks.append((str(f), str(out), args.field, args.id_field, args.dedupe,
                      args.fraction, args.seed))

    if args.field == "c-smiles":
        print("WARNING: 'c-smiles' is PubChem Canonical SMILES -- stereochemistry is stripped.\n"
              "         Use --field smiles unless you specifically want stereo discarded.",
              file=sys.stderr, flush=True)

    print(f"{len(files)} input files, {skipped} already done, {len(tasks)} to process, "
          f"{args.jobs} workers, field={args.field!r}, fraction={args.fraction:g}, "
          f"seed={args.seed}", flush=True)
    if not tasks:
        return 0

    tot_lines = tot_sampled = tot_written = tot_missing = tot_dup = tot_noid = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(extract_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:  # keep going; one bad file shouldn't sink the run
                print(f"  !! FAILED {Path(futs[fut]).name}: {e!r}", file=sys.stderr, flush=True)
                continue
            tot_lines += r["lines"]
            tot_sampled += r["sampled"]
            tot_written += r["written"]
            tot_missing += r["missing"]
            tot_dup += r["dupes"]
            tot_noid += r["no_id"]
            el = time.perf_counter() - t0
            rate = tot_written / el if el else 0
            print(f"[{i}/{len(tasks)}] {r['src']:<32} {r['written']:>8,} of {r['lines']:>8,} "
                  f"lines ({r['missing']:,} missing, {r['no_id']:,} no id, {r['dupes']:,} dup) "
                  f"{r['secs']:>6.1f}s "
                  f"| total so far {tot_written:,}", flush=True)

    el = time.perf_counter() - t0
    print(f"\ndone in {el/60:.1f} min")
    print(f"  lines read     {tot_lines:,}")
    print(f"  lines sampled  {tot_sampled:,}  "
          f"({100*tot_sampled/max(tot_lines,1):.2f}% of lines read)")
    print(f"  smiles written {tot_written:,}")
    print(f"  field missing  {tot_missing:,}")
    print(f"  no ID          {tot_noid:,}  (no '{args.id_field}' field; skipped, cannot be tracked)")
    if args.dedupe:
        print(f"  in-shard dupes {tot_dup:,}")
    print(f"  read speed     {tot_lines/el/1000:,.0f}k lines/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
