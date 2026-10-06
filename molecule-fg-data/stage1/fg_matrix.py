#!/usr/bin/env python3
"""Stage 2: functional-group presence matrix over SMILES shards, in parallel.

Reads the `<cid>\t<smiles>` shards from extract_smiles.py, runs AccFG (the logic of
spreadsheet.py) across processes, and writes one Parquet part per task into an output
directory that IS the final dataset -- no merge step. Every part has columns
`cid, Molecule, <one uint8 column per FG>`.

Salts and mixtures (any SMILES containing '.') are rejected before AccFG runs. Every
rejected molecule is listed with its cid in _rejects/, so each input cid is accounted for.

Requires: pip install accfg rdkit pyarrow. On a headless machine `import accfg` needs
the X11 library libXrender (AccFG imports RDKit's drawing code); see RUNBOOK.md.

The reconciliation problem is solved by construction: the parent process builds
the `afg` object once, freezes `list(afg.dict_fgs)` as the column order, and ships
that order to every worker. Every part therefore has a byte-identical schema, so
the parts concatenate for free.

    python fg_matrix.py /data/smiles -o /data/fg --batch 20000 -j 16
    # then:  pyarrow.dataset.dataset('/data/fg')  /  duckdb "SELECT * FROM '/data/fg/*.parquet'"

Resumable: re-run the same command and completed parts are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")  # RDKit's C++ parse warnings are per-molecule and very loud

# Use the accfg package vendored in this repository (two levels up) if it is not installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if (_REPO_ROOT / "accfg").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

class _InlineExecutor:
    """Stand-in for concurrent.futures executors that runs each call at once, in this
    process, and hands back an already-completed Future.

    AccFG 0.0.7 (the newest release installable on Python 3.9) opens a 4-process pool
    inside run_mol FOR EVERY MOLECULE and sends all 534 pattern checks through it. That
    measured 2 mol/s, makes 16 cluster workers spawn ~64 more processes, and cannot pickle
    the CachedAccFG subclass. AccFG 0.1.0 dropped the pool and checks patterns in a plain
    loop. Swapping this in gives 0.0.7 that same sequential behaviour.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as e:  # surfaces at fut.result(), exactly as with a real pool
            fut.set_exception(e)
        return fut

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return map(fn, *iterables)

    def shutdown(self, wait=True, **kwargs):
        pass


def _run_accfg_in_process(AccFG) -> None:
    """Point any executor AccFG's module uses at _InlineExecutor. No-op on versions that
    don't use one (0.1.0 imports ThreadPoolExecutor but never calls it)."""
    mod = sys.modules[AccFG.__module__]
    for name in ("ProcessPoolExecutor", "ThreadPoolExecutor"):
        if hasattr(mod, name):
            setattr(mod, name, _InlineExecutor)


def import_accfg(verbose: bool = False):
    """Return the AccFG class, even on machines without X11 libraries.

    accfg/__init__.py also imports accfg.draw, which imports RDKit's drawing code, which
    needs libXrender / libXext. Headless cluster nodes often lack them, and installing
    them needs root. The pipeline never draws. So if the normal import fails, load only
    accfg.main -- the module that defines AccFG, which needs just rdkit, csv and networkx --
    without running the package __init__. Results are identical: it is the same class.
    """
    try:
        from accfg import AccFG
        return AccFG
    except ImportError as err:
        first_error = err

    import importlib
    import importlib.util
    import types

    for name in [m for m in sys.modules if m == "accfg" or m.startswith("accfg.")]:
        del sys.modules[name]  # drop anything half-imported by the failed attempt
    spec = importlib.util.find_spec("accfg")  # locates the package without running it
    if spec is None or not spec.submodule_search_locations:
        raise first_error
    pkg = types.ModuleType("accfg")
    pkg.__path__ = list(spec.submodule_search_locations)
    pkg.__spec__ = spec
    sys.modules["accfg"] = pkg
    AccFG = importlib.import_module("accfg.main").AccFG
    if verbose:
        print(f"note: `import accfg` failed ({first_error}); loaded accfg.main directly "
              f"instead. Only AccFG's drawing functions are unavailable, and fg_matrix.py "
              f"does not use them.", file=sys.stderr, flush=True)
    return AccFG


def build_afg(verbose: bool = False):
    """Construct the AccFG analyser (pip install accfg). Called once per worker process.

    AccFG's _is_fg_in_mol re-compiles every SMARTS pattern from text for every molecule,
    which is ~56% of its runtime. This subclass compiles each pattern once and reuses it.
    Measured 2.75x faster on 3,000 real molecules with identical FG sets, atom mappings
    and FG-graph edges. If a future AccFG renames _is_fg_in_mol, the override silently
    stops applying -- correct results, just slower.
    """
    AccFG = import_accfg(verbose)
    _run_accfg_in_process(AccFG)

    class CachedAccFG(AccFG):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._smarts_cache = {}

        def _is_fg_in_mol(self, mol, fg):
            query = self._smarts_cache.get(fg)
            if query is None:
                query = self._smarts_cache[fg] = Chem.MolFromSmarts(fg)
            matches = mol.GetSubstructMatches(query, uniquify=True)
            return len(matches) > 0, matches

    return CachedAccFG()


def default_jobs() -> int:
    """Cores this job may use -- NOT os.cpu_count(), which on a cluster node reports the
    whole machine. Under LSF, the slots `bsub -n` granted on this host (LSB_MCPU_HOSTS);
    otherwise the CPU affinity mask, which respects cgroups and taskset."""
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


_AFG = None
_COLS: list[str] = []


def _init_worker(cols: list[str]) -> None:
    """ProcessPoolExecutor initializer: pay the analyser setup cost once per process."""
    global _AFG, _COLS
    _AFG = build_afg()
    _COLS = cols


def fg_vector(afg, mol) -> set[str]:
    """Names of every functional group present. Mirrors spreadsheet.fg_presence_vector.
    run_mol(mol) is exactly what AccFG.run(smiles, canonical=False) does after parsing."""
    _, fg_graph = afg.run_mol(mol, show_atoms=True, show_graph=True)
    return set(fg_graph.nodes)


def process_batch(task: tuple[str, int, int, str, bool]) -> dict:
    """Handle one (shard, start_line, count) slice. Runs in a worker process."""
    shard_s, start, count, out_s, canonical = task
    out = Path(out_s)
    # Leading dot: pyarrow.dataset() skips names starting with '.' or '_'. Without it, a
    # job killed mid-write (e.g. by a cluster wall-time limit) leaves a truncated file
    # that makes the whole output directory unreadable.
    tmp = out.parent / f".{out.name}.partial"

    # Read only this slice, so nothing large crosses the process boundary.
    recs: list[tuple[str, str]] = []
    with open(shard_s, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < start:
                continue
            if i >= start + count:
                break
            s = line.rstrip("\n")
            if s:
                mol_id, _, smi = s.partition("\t")
                recs.append((mol_id, smi))

    ids: list[str] = []
    mols: list[str] = []
    bits: list[set[str]] = []
    rejects: list[tuple[str, str, str, str]] = []  # (cid, smiles, code, detail)

    for mol_id, smi in recs:
        if "." in smi:  # '.' only ever means a disconnected component: salt or mixture
            rejects.append((mol_id, smi, "salt_or_mixture", ""))
            continue
        try:
            # Parse once here and hand the mol to AccFG, rather than letting AccFG parse
            # the string. Same single parse, but a bad SMILES gets a clear reject reason
            # instead of RDKit's opaque ArgumentError from inside AccFG.
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rejects.append((mol_id, smi, "unparseable", ""))
                continue
            key = Chem.MolToSmiles(mol) if canonical else smi
            bits.append(fg_vector(_AFG, mol))
            mols.append(key)
            ids.append(mol_id)
        except Exception as e:  # never let one molecule kill a 20k batch
            rejects.append((mol_id, smi, "accfg_error", f"{type(e).__name__}: {e}"))

    # Build the table column-wise as uint8 -- far cheaper in RAM than a dict-of-rows
    # DataFrame, and Parquet's RLE makes the mostly-zero columns tiny on disk.
    arrays = [pa.array(ids, type=pa.string()), pa.array(mols, type=pa.string())]
    for name in _COLS:
        arrays.append(pa.array([1 if name in b else 0 for b in bits], type=pa.uint8()))
    table = pa.table(arrays, names=["cid", "Molecule"] + _COLS)

    pq.write_table(table, tmp, compression="zstd", compression_level=3)
    os.replace(tmp, out)  # atomic -> presence of the .parquet means this task is complete

    if rejects:
        # Underscore-prefixed dir so pyarrow.dataset() skips it -- a stray .txt beside the
        # parts makes dataset discovery try to parse it as Parquet and blow up.
        rej_dir = out.parent / "_rejects"
        rej_dir.mkdir(exist_ok=True)
        # One line per molecule: cid, smiles, code, detail. Codes: salt_or_mixture,
        # unparseable, accfg_error. Stage 2 reads these into molecule_status.parquet.
        with (rej_dir / f"{out.stem}.txt").open("w", encoding="utf-8") as w:
            for mol_id, smi, code, detail in rejects:
                detail = " ".join(detail.split())  # no tabs/newlines inside a field
                w.write(f"{mol_id}\t{smi}\t{code}\t{detail}\n")

    return {"out": out.name, "rows": len(mols), "rejects": len(rejects)}


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for _ in fh:
            n += 1
    return n


def shard_counts(shards: list[Path], cache_file: Path, jobs: int) -> dict[str, int]:
    """Line counts per shard, cached -- a resumed run shouldn't re-read every shard.
    Keyed on (size, mtime) so an edited shard is recounted."""
    cache: dict[str, list] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cache = {}

    stale = [s for s in shards
             if cache.get(s.name, [None, None, None])[:2]
             != [s.stat().st_size, int(s.stat().st_mtime)]]

    if stale:
        print(f"  counting {len(stale)} uncached shard(s) ...", flush=True)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for sh, n in zip(stale, pool.map(count_lines, stale)):
                cache[sh.name] = [sh.stat().st_size, int(sh.stat().st_mtime), n]
        cache_file.write_text(json.dumps(cache, indent=1))
    else:
        print("  all shard counts cached", flush=True)

    return {s.name: cache[s.name][2] for s in shards}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", type=Path, help="directory of .smi shards from stage 1")
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="worker processes (default: cores allocated to this job -- under "
                         "LSF, the slots from bsub -n on this host)")
    ap.add_argument("--batch", type=int, default=20000,
                    help="molecules per Parquet part (default 20000)")
    ap.add_argument("--canonical", action="store_true",
                    help="OPT-IN: round-trip each SMILES through RDKit so the Molecule column "
                         "is RDKit-canonical. Does NOT change any FG column, and with real AccFG "
                         "its cost is lost in the noise (one MolToSmiles vs ~5 ms of FG matching "
                         "per molecule). Only worth it if you need Molecule to join against "
                         "other RDKit-derived data. Default is off: keys are the SMILES as "
                         "extracted.")
    ap.add_argument("--limit-shards", type=int, default=None,
                    help="only process the first N shards -- use this to time a pilot run")
    args = ap.parse_args()
    if args.jobs is None:
        args.jobs = default_jobs()

    shards = sorted(args.indir.glob("*.smi"))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if not shards:
        print(f"no .smi shards in {args.indir}", file=sys.stderr)
        return 1
    for sh in shards:
        with sh.open("r", encoding="utf-8") as fh:
            first = next((ln for ln in fh if ln.strip()), "")
        if first and "\t" not in first:
            print(f"ERROR: {sh.name} has no ID column (expected '<cid>\\t<smiles>' lines).\n"
                  f"       It was written by an older extract_smiles.py. Re-run stage 1 into\n"
                  f"       a fresh directory so every molecule can be tracked by its ID.",
                  file=sys.stderr)
            return 2
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Parts written before IDs were added have no `cid` column and kept salts/mixtures.
    # Refuse to mix them with new parts in one dataset.
    meta = {"id_column": "cid", "drop_salts_and_mixtures": True}
    meta_file = args.outdir / "_meta.json"
    if meta_file.exists():
        if json.loads(meta_file.read_text()) != meta:
            print(f"ERROR: {meta_file} disagrees with this version of fg_matrix.py. Use a "
                  f"fresh --outdir.", file=sys.stderr)
            return 2
    elif any(args.outdir.glob("*.parquet")):
        print(f"ERROR: {args.outdir} holds parts from an older fg_matrix.py (no cid column).\n"
              f"       Use a fresh --outdir.", file=sys.stderr)
        return 2
    else:
        meta_file.write_text(json.dumps(meta, indent=1))

    # Freeze the column order ONCE, here. This is what makes the parts mergeable.
    print("building analyser in parent to fix column order ...", flush=True)
    cols = list(build_afg(verbose=True).dict_fgs.keys())
    print(f"{len(cols)} functional-group columns", flush=True)

    schema_file = args.outdir / "_columns.json"
    if schema_file.exists():
        prev = json.loads(schema_file.read_text())
        if prev != cols:
            print("ERROR: _columns.json in outdir disagrees with the current analyser.\n"
                  "       Mixing them would produce an unmergeable dataset. Use a fresh\n"
                  "       --outdir, or delete the old one.", file=sys.stderr)
            return 2
    else:
        schema_file.write_text(json.dumps(cols, indent=1))

    print(f"counting molecules in {len(shards)} shards ...", flush=True)
    counts = shard_counts(shards, args.outdir / "_counts.json", args.jobs)
    tasks, total = [], 0
    for sh in shards:
        n = counts[sh.name]
        total += n
        for start in range(0, n, args.batch):
            out = args.outdir / f"{sh.stem}-{start:09d}.parquet"
            if out.exists():
                continue
            tasks.append((str(sh), start, args.batch, str(out), args.canonical))

    print(f"{total:,} molecules; {len(tasks):,} batches to run on {args.jobs} workers")
    if not tasks:
        print("nothing to do -- all parts already present")
        return 0

    done_rows = done_rej = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs,
                             initializer=_init_worker, initargs=(cols,)) as pool:
        futs = {pool.submit(process_batch, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:
                sh, start, *_ = futs[fut]
                print(f"  !! FAILED {Path(sh).name}@{start}: {e!r}", file=sys.stderr, flush=True)
                continue
            done_rows += r["rows"]
            done_rej += r["rejects"]
            el = time.perf_counter() - t0
            rate = (done_rows + done_rej) / el if el else 0
            remaining = (total - done_rows - done_rej) / rate if rate else 0
            print(f"[{i}/{len(tasks)}] {r['out']:<40} {r['rows']:>6,} rows "
                  f"| {rate:,.0f} mol/s | ETA {remaining/3600:5.2f} h", flush=True)

    el = time.perf_counter() - t0
    print(f"\ndone in {el/3600:.2f} h  |  {done_rows:,} rows, {done_rej:,} rejected"
          f"  |  {done_rows/el:,.0f} mol/s")
    print(f"dataset: {args.outdir}  (read all parts as one table, no merge needed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
