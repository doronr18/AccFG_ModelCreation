# SMILES → functional-group matrix, 355 files × 500k lines

## The answer: extract first, in two stages

Measured on one core, with your record shape (~610 bytes/line):

| Stage | Throughput | 177M molecules, 1 core |
|---|---|---|
| Extract SMILES, `json.loads` | 143k/s | 21 min |
| Extract SMILES, `orjson` | 368k/s | 8 min |
| Extract SMILES, regex | 442k/s | 7 min |
| RDKit canonical round-trip **alone** | 5.7k/s | **8.6 h** |
| + `afg.run` FG matching | ~2–3k/s | **~16–24 h** |

> **Superseded (2026-10-01).** The FG-matching row came from a 10-pattern stand-in. Real
> AccFG (534 patterns) measures **81 mol/s** as shipped and **~170–220 mol/s** with the
> SMARTS caching in `fg_matrix.py`, per core, on real molecules. That makes the
> extraction-vs-chemistry gap far larger than 100×, so the conclusion below holds even
> more strongly. Scope is also now a ~5M sample, not 177M; see RUNBOOK.md for current
> figures.

Extraction is **~100× cheaper** than the chemistry. So the question isn't really
"which is faster" — the two options differ by minutes out of a ~20-hour job.
Extract first anyway, because it buys things that matter at that duration:

- **You can restart stage 2 without re-parsing 108 GB of JSON.** On a 16-hour job
  you *will* restart it — to fix a pattern, after an OOM, after a machine reboot.
- **You get an exact molecule count up front**, so the ETA is real rather than a guess.
- **Global dedupe becomes possible.** Can't dedupe a stream you're consuming once.
- Stage 1 is I/O-bound, stage 2 is CPU-bound. Fusing them makes every core wait on disk.

## Threads vs processes

Use **processes, not threads.** `Chem.MolFromSmiles`, `MolToSmiles` and the SMARTS
matching in `afg.run` hold the GIL, so a `ThreadPoolExecutor` gives you ~1 core of
throughput no matter how many threads you spawn. Both scripts use
`ProcessPoolExecutor`. Threads would help only stage 1, and stage 1 is already 7 min.

## Three things I found that will bite you

**1. `c-smiles` is stereo-stripped — use `--field smiles`.** Measured on the 32-record
real sample: `smiles` carries **32** specified stereo elements, `c-smiles` carries **0**.
`c-smiles` is PubChem's *Canonical* SMILES (stereo discarded); `smiles` is its *Isomeric*
SMILES (stereo kept). Among `canonicalized=1` records, "has stereo" predicts
`smiles != c-smiles` **perfectly** — 29/29, no counterexamples. The relationship is
simply `c-smiles == smiles minus stereo`:

```
smiles   : CC(=O)N[C@@H]1[C@H]([C@H]([C@H](O[C@H]1O[C@@H]2...
c-smiles : CC(=O)NC1C(C(C(OC1OC2...                          <- all 10 centres gone
```

This is a **correctness** issue, not a representation one: stereo is a real graph
difference. Reading `c-smiles` would mis-call any stereo-defined group in `dict_fgs`
(cis/trans alkenes, sugar configurations, D/L amino acids) on ~40% of your molecules.
`extract_smiles.py` now defaults to `--field smiles` and warns if you ask for `c-smiles`.

**2. `canonicalized` does not mean "RDKit-canonical", and doesn't affect your pipeline.**
"Canonical SMILES" is only canonical relative to an *algorithm*; there is no universal
form. On the real sample **0/29** `canonicalized=1` records are RDKit-canonical in either
field. The flag describes PubChem's own processing — since you canonicalise with RDKit
yourself, it changes nothing about what to do.

The 3 `canonicalized=0` records are all allene-like (cumulated `C=C=C` / `C=C=N` / `C(=N)=S`),
have no specified stereo, and all parse and round-trip cleanly in RDKit. But they also come
from a completely different CID block (176,00x,xxx vs 10,000,0xx), so with n=3 the structural
explanation and the deposition-era explanation are confounded — I can't tell you which it is.
It doesn't matter operationally: they need no special handling.

(An earlier draft of this file cited "only 20% match" as a corpus measurement. It wasn't —
it came from a synthetic fixture whose SMILES I'd hand-written in Kekulé style. The real
figure is 3.1% over the 32-record sample. Re-run `check_smiles_fields.py` on more files.)

**What it does and doesn't affect.** Functional-group *presence* does not depend on
representation at all — verified across 10 drug-like molecules, including under random
atom reorderings, because RDKit matches SMARTS against the parsed graph and perceives
aromaticity at parse time. Canonicalisation only determines what the `Molecule` **key**
column looks like, so it matters iff you join or dedupe against other RDKit-derived data.

And the cost is not where it appears. `MolFromSmiles` (~89 µs) is mandatory and is the
expensive half; `MolToSmiles` (~58 µs) is all you can actually skip. Most of the apparent
canonicalisation cost is a **redundant second parse**, because `afg.run` takes a string
and re-parses the molecule you just built:

| Approach | Parses | Canonical key? | Measured on 62,628 real molecules |
|---|---|---|---|
| **default** — straight to `afg.run` | 1 | no | **4,946 mol/s** |
| `--canonical` (RDKit round-trip first) | 2 | yes | 1,964 mol/s |
| parse once, pass the **mol** to `afg.run` | 1 | yes | not implemented — see below |

**Decided: no canonicalisation.** Verified on all 62,628 real molecules — **0 of 10 FG
columns differ** between the two runs. Only the `Molecule` key column changes. The default
is therefore off, and `--canonical` is opt-in; that way a 20-hour run can't silently pay
2.5× for keys you don't want.

The third row is the option you'd want *if* you ever need canonical keys back: it keeps
them while dropping the redundant parse. It needs `afg.run` to accept a mol object rather
than a string. Not implemented, since the decision made it moot.

**3. `spreadsheet.py`'s `update_fg_presence_csv` will never finish at this scale.**
Every call does `pd.read_csv(output_csv)` on the whole accumulated file, concats,
and rewrites it (lines 54–67). That's O(n²) in total bytes — by row 100M you're
re-reading and re-writing tens of GB per batch. It also builds `records` for every
molecule in memory before constructing the DataFrame. Use it for a few thousand
molecules; do not point it at 177M. `fg_matrix.py` writes independent Parquet parts
instead, which is O(n) and parallel-safe.

## "Reconciling the outputs" — designed away, not solved

You flagged this as the cost of running concurrently. It isn't one, if the column
set is decided in one place. `fg_matrix.py` builds the analyser **once in the parent**,
freezes `list(afg.dict_fgs)` as the column order, and ships that list to every worker
via the pool `initializer`. Every part therefore gets a byte-identical schema, so the
output directory *is* the dataset — concatenation is free and there is no merge step:

```python
import pyarrow.dataset as ds
ds.dataset('/data/fg').to_table()          # or scanner() to stream it
```
```sql
duckdb -c "SELECT * FROM '/data/fg/*.parquet' WHERE amide = 1"
```

The frozen order is saved to `_columns.json` and re-checked on every run. If you
change your FG definitions mid-job it refuses to write into the same directory
rather than silently producing an unmergeable mix.

Note what *would* have needed reconciling and doesn't: a worker that saw no molecule
containing some FG still emits that column, as zeros, because the column list comes
from `dict_fgs` rather than from what it happened to observe.

## Running it

```bash
pip install rdkit pyarrow          # orjson optional; the regex path is used by default

# Stage 0 — confirm smiles / c-smiles / canonicalized on real data before a long run.
python check_smiles_fields.py /data/jsonl -n 20000

# Stage 1 — ~7 min for the lot. --dedupe drops repeats within each shard.
python extract_smiles.py /data/jsonl -o /data/smiles -j $(nproc) --dedupe --resume

# Cross-shard dedupe, if stage 1 reports meaningful in-shard duplication. Lines are
# `cid<TAB>smiles`; key on the SMILES column so each kept line keeps its own cid:
LC_ALL=C sort -t "$(printf '\t')" -k2,2 -s -u --parallel=$(nproc) -S 50% -T /var/tmp \
  /data/smiles/*.smi \
  | split -l 500000 -d -a 4 --additional-suffix=.smi - /data/smiles_u/u

# Pilot: time 2 shards before committing to the full run.
python fg_matrix.py /data/smiles -o /data/fg_pilot -j $(nproc) --limit-shards 2

# Full run. Re-run the identical command to resume; finished parts are skipped.
python fg_matrix.py /data/smiles -o /data/fg -j $(nproc) --batch 20000
```

`build_afg()` in `fg_matrix.py` is already wired to AccFG (`pip install accfg`); no edit
is needed. See RUNBOOK.md step 2 for what it does and step 1 for the libXrender import
problem on headless machines.

### Tuning

- `-j`: use physical cores, not SMT threads. RDKit is FP/branch heavy and
  hyperthreads oversubscribe it. Start at `nproc`, try `nproc/2` in the pilot.
- `--batch 20000`: ~10s of work per task. Small enough that a crash loses little
  and the progress bar moves; large enough that process overhead is invisible.
- Stage 2 RAM is ~`batch × n_fg` bytes per worker — a few MB. It won't OOM.

### Sizing

Measured on 62,628 real molecules, decomposed so it extrapolates to your actual FG count:

| Component | Bytes/molecule |
|---|---|
| `Molecule` SMILES string column | 16.47 |
| each uint8 FG column | 0.108 |

- Input: ~108 GB. Stage 1 output: ~8 GB of plain text.
- Stage 2 output for 177M molecules: **3.9 GB** at 50 FG columns, **4.8 GB** at 100,
  **6.8 GB** at 200. The string column dominates; the mostly-zero FG columns RLE away
  to about a tenth of a byte each.
- The same data as CSV would be **~43 GB** at 100 columns. Use Parquet.

### Operational notes

- Both scripts write to a `.partial` file and `os.replace` it, so an interrupted
  run never leaves a half-written shard that `--resume` would mistake for complete.
- A molecule that fails to parse is written to `_rejects/<part>.txt` and skipped. The
  underscore matters: `pyarrow.dataset()` tries to read any stray `.txt` beside the parts
  as Parquet and raises, so rejects must stay out of the dataset namespace.
  `spreadsheet.py` raises `ValueError` here, which would kill a whole batch.
- Run stage 2 under `nohup`/`tmux`. It prints a rolling `mol/s` and ETA.
