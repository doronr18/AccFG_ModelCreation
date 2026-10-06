# Runbook: 355 `.jsonl` files → functional-group matrix

The linear list of steps. Rationale and measurements live in `README_pipeline.md`. All
scripts named here are in this folder (`molecule-fg-data/stage1/`).

## How the pieces relate

```
/data/jsonl/*.jsonl          355 files
        │
        │  extract_smiles.py          random 3% of lines from EVERY file,
        │                             `smiles` field             [minutes; reads every file]
        ▼
/data/smiles/*.smi           one `cid<TAB>smiles` per line    (target ~5M SMILES)
        │
        │  fg_matrix.py               drops salts/mixtures,      [~170 mol/s per core:
        ▼                             runs AccFG                  ~8 core-hours per 5M]
/data/fg/*.parquet           cid + Molecule + 534 0/1 FG columns   ~1 GB per 5M
        │                    (+ _rejects/: every dropped cid with its reason)
        │
        │  export_csv.py              only if you need CSV
        ▼
/data/fg_matrix.csv          cid + Molecule + 534 0/1 FG columns
```

The molecule ID (`cid`) is carried from the JSONL records into every Parquet row and every
reject line, so each input molecule can be traced. Stages 2–5 in `molecule-fg-data/` do not
read this output yet; until they do, they run AccFG again from `smiles.json`.

Sizes and times are given per 5M molecules because the total depends on how many lines
the files really hold; see step 4. All figures measured on your real molecules with real
AccFG.

**`fg_matrix.py` replaces `spreadsheet.py` for the bulk run — it is not a step feeding
into it.** It produces the same table: `Molecule` first, then one 0/1 column per
functional group in `afg.dict_fgs` order. Verified against
`spreadsheet.fg_presence_dataframe(AccFG(), smis, canonical=False)` running the
unmodified AccFG on 500 real molecules: **0 of 267,000 cells differ**. Keep `spreadsheet.py`
for small ad-hoc runs; just never point its `update_fg_presence_csv` at a large set (it
re-reads and rewrites the whole CSV per call — O(n²)).

---

## Step 1 — install

```bash
pip install accfg rdkit networkx pyarrow pandas
# check, from the directory holding fg_matrix.py -- expect "534":
python -c "import fg_matrix; print(len(fg_matrix.build_afg(verbose=True).dict_fgs))"
```

`pandas` is only needed if you still use `spreadsheet.py` directly; the pipeline
scripts don't import it. On Python 3.9, pip installs AccFG 0.0.7, which does not pull in
`rdkit` or `networkx` itself, so list them explicitly.

**`ImportError: libXext.so.6` / `libXrender.so.1` — handled, no root needed.** A plain
`from accfg import AccFG` fails on headless machines: AccFG's `__init__.py` imports its
drawing module, which loads RDKit's drawing code, which needs these X11 libraries.
`fg_matrix.py` catches that and loads `accfg.main` directly, the module that defines
`AccFG`. It prints one `note:` line saying so. The result is the same class, and the
pipeline never draws. Verified on Python 3.9 and 3.12 with no X11 libraries at all.

Only plain `import accfg` is affected — for example in `spreadsheet.py` or a notebook. If
you need that to work too, these install the libraries without root into a conda env:

```bash
conda install -c conda-forge xorg-libxext xorg-libxrender
```

**Python version.** Python 3.9 can only install AccFG 0.0.7; 3.10+ gets 0.1.0. Both give
**identical results** (0 differing cells, 1,875 real molecules × 534 FGs), but 0.1.0 is
~1.8× faster. 0.0.7 also opens a 4-process pool per molecule, at 2 mol/s; `fg_matrix.py`
neutralises that (see step 2). A Python 3.10+ conda env is optional, not required:
`conda create -n accfg python=3.11 && conda activate accfg && pip install accfg pyarrow`.

## Step 2 — the analyser  (no edit needed)

`build_afg()` in `fg_matrix.py` is already wired to AccFG. It returns a small subclass,
`CachedAccFG`, that differs from AccFG in one way: each SMARTS pattern is compiled once
and reused. Stock AccFG recompiles all 534 patterns for every molecule, which is ~56% of
its runtime. Measured on 3,000 real molecules: **2.75× faster, with 0 differences** in FG
sets, atom mappings, or FG-graph edges.

On AccFG 0.0.7 (what Python 3.9 installs), `build_afg()` makes one more adjustment.
That version's `run_mol` opens a fresh 4-process pool **for every molecule** and sends
all 534 checks through it. That measured **2 mol/s**, would make 16 cluster workers spawn
about 64 more processes, and cannot work with the subclass above (it fails with
`Can't pickle local object`). `fg_matrix.py` replaces that pool, inside AccFG's module
only, with a stand-in that runs each check immediately in-process. That matches what 0.1.0
does natively. Measured against stock 0.0.7 on 150 real molecules: **0 differences, 56×
faster** (112 vs 2 mol/s). On 0.1.0 the adjustment is a no-op.

Things you might choose to change there:

- `AccFG(lite=True)` — 504 broader groups instead of 534 (for example, one `hydroxyl`
  instead of primary/secondary/tertiary). Changes the column set, so use a fresh `--outdir`.
- `AccFG(user_defined_fgs={...})` — add your own groups.

`fg_matrix.py` parses each SMILES itself and passes the molecule to `AccFG.run_mol`.
That is exactly what `AccFG.run(smiles, canonical=False)` does internally, so it costs no
extra parse. A SMILES that fails to parse is logged to `_rejects/` as `unparseable SMILES`.

## Step 3 — sanity-check the data (optional, seconds)

```bash
python check_smiles_fields.py /data/jsonl -n 20000
```

Confirms on your real files what we established on samples: `c-smiles` is stereo-stripped
(use `smiles`), and neither field is RDKit-canonical. Worth running once before a
multi-hour job.

## Step 4 — extract a 3% sample of SMILES  (minutes, parallel)

```bash
python extract_smiles.py /data/jsonl -o /data/smiles -j $(nproc) --resume
```

**Samples a random 3% of the lines in every file by default** (`--fraction 0.03`). The
sampled lines are spread across each whole file rather than taken from its head, since the
files are in CID order. It is seeded (`--seed 0`), so re-running gives exactly the same
lines regardless of `-j`. `--fraction 1` takes everything.

Sampling does not make this step much faster: every line still has to be read to find the
3%. Skipped lines are never parsed, though, so it is I/O-bound.

**Check the total at the end of the run.** It prints `smiles written`. The 5M target
assumes ~500k lines per file, but the one real file seen so far
(`processed_Compound_176000001_176500000.jsonl`) has **62,628** lines. Its name spans
500k CIDs, but it holds far fewer records. If the other files are similar, 3% gives
~0.7M, not 5M. To change the fraction afterwards, use a fresh `--outdir`: the script
records its parameters in `_params.json` and refuses to resume with different ones.

Defaults to `--field smiles`, which preserves stereochemistry. **Do not use
`--field c-smiles`** — it has stereo stripped and would mis-call stereo-defined groups.

Each line is `cid<TAB>smiles` (`--id-field cid`). Records with no `cid` are skipped and
counted as `no ID` in the log. Shards written by the older, ID-less version are refused
by `fg_matrix.py`.

Add `--dedupe` to drop repeated SMILES within each shard, keeping the first `cid`. The
dropped duplicates' IDs do not reach any later stage. If duplication is material, dedupe
across shards too. This keys on the SMILES column only, so each kept line keeps its own `cid`:

```bash
mkdir -p /data/smiles_u
LC_ALL=C sort -t "$(printf '\t')" -k2,2 -s -u --parallel=$(nproc) -S 50% -T /var/tmp \
  /data/smiles/*.smi \
  | split -l 500000 -d -a 4 --additional-suffix=.smi - /data/smiles_u/u
```

Then use `/data/smiles_u` as the input to step 5.

## Step 5 — pilot the FG run  ← do not skip this

```bash
python fg_matrix.py /data/smiles -o /data/fg_pilot -j $(nproc) --limit-shards 2
```

This prints the real aggregate `mol/s` on **your** machine. The ~170 mol/s per core
quoted above was measured on one core of a small VM; yours will differ. Multiply out
before committing:

    total hours ≈ (smiles written in step 4) / (aggregate mol_per_sec) / 3600

For example, 5M molecules on 16 cores at ~170 mol/s each is about 30 minutes.

Also try `-j $(nproc)` vs `-j $(( $(nproc) / 2 ))`. RDKit is FP- and branch-heavy, so
hyperthreads often do not help.

## Step 6 — the full run  (hours; use tmux or nohup)

```bash
nohup python fg_matrix.py /data/smiles -o /data/fg -j $(nproc) --batch 20000 \
      > /data/fg.log 2>&1 &
tail -f /data/fg.log
```

- Prints rolling `mol/s` and an ETA.
- **Resumable**: re-run the identical command. Completed Parquet parts are skipped, so a
  crash or reboot costs you at most one batch.
- Canonicalisation is **off by default**, per the decision: the `Molecule` column holds
  the SMILES exactly as extracted. Pass `--canonical` only if you later need
  RDKit-canonical keys. It changes no FG column and, with real AccFG, costs nothing
  measurable.
- Failed molecules land in `/data/fg/_rejects/*.txt` with reasons. Check it when done:
  `wc -l /data/fg/_rejects/*.txt`

## Step 7 — use the result

The output directory **is** the dataset. No merge step — every part shares one schema by
construction.

```python
import pyarrow.dataset as ds
t = ds.dataset('/data/fg').to_table()        # or .scanner() to stream it
```

```bash
duckdb -c "SELECT count(*) FROM '/data/fg/*.parquet' WHERE amide=1 AND alcohol=1"
```

Only if something downstream truly needs CSV:

```bash
python export_csv.py /data/fg -o /data/fg_matrix.csv          # ~1.2 GB per 5M rows at 100 FGs
python export_csv.py /data/fg -o /data/fg_matrix.csv --gzip   # ~10x smaller
```

That CSV is byte-identical in shape to what `spreadsheet.py` would have produced.

---

## Running on an LSF cluster (`bsub`)

Steps 4–6 become two batch jobs. `extract.lsf` and `fg.lsf` are ready-made; edit the
paths block at the top of each, then:

```bash
bsub < extract.lsf                                  # stage 1
bsub -w 'done(extract_smiles)' < fg.lsf             # stage 2, starts when stage 1 succeeds
bjobs                                               # status
tail -f fg_matrix.<jobid>.log                       # live progress and ETA
```

**The `<` is required.** `bsub < fg.lsf` makes LSF read the `#BSUB` lines.
`bsub fg.lsf` runs the file as a plain command and silently ignores all of them: no
core count, memory, or time limit.

What is different from running by hand:

- **Leave out `-j`.** Both scripts now detect the cores LSF granted to the job (from
  `LSB_MCPU_HOSTS`). The old default, `os.cpu_count()`, reported every core on the node.
  Asking for 16 slots on a 64-core node would have started 64 workers on 16 cores.
- **Keep `-R "span[hosts=1]"`.** Worker processes cannot cross nodes. Without it, LSF may
  scatter the slots, and the job runs on only the first node's share (it prints a warning
  if this happens).
- **Memory: about 210 MB per worker plus 150 MB for the main process**, measured at
  `--batch 20000`. The script reserves 500 MB per slot. Whether `rusage[mem=…]` means per
  slot or per job, and MB or GB, depends on how your site configured LSF. Check before the
  first big run.
- **Wall time.** For 5M molecules on 16 workers: about 55 minutes on Python 3.9 / AccFG
  0.0.7 (~100 mol/s per worker), or about 25 minutes on 3.10+ / 0.1.0 (~200 mol/s), if your
  nodes match these measurements. `-W 2:00` leaves headroom. If a job is killed at its limit,
  resubmit `fg.lsf` unchanged: finished parts are kept and only the remaining batches run.
  Verified by killing a run with `kill -9` and resuming. The result was identical to an
  uninterrupted run.
- **Do not run two `fg.lsf` jobs on the same output directory at once**, for example by
  resubmitting while the first is still pending. They would repeat each other's work and
  race on `_counts.json`. Check `bjobs` first.
- **Test the compute-node environment, not just the login node.** Environments and
  libraries can differ between them. `fg.lsf` checks that AccFG loads before starting and
  fails within seconds if it can't. To check before queueing a real job:
  `bsub -Is -n 1 python -c "import fg_matrix; print(len(fg_matrix.build_afg(verbose=True).dict_fgs))"`
  (run from the directory holding `fg_matrix.py`; expect `534`)
- **Use scratch space for outputs** if your site has it. Home directories often have small
  quotas and slow shared filesystems.
- **Pilot first** (step 5): run `fg.lsf` with `--limit-shards 2` and `-W 0:30` to get your
  nodes' real speed before sizing `-n` and `-W` for the full job.

---

## Quick reference

| Script | Does |
|---|---|
| `check_smiles_fields.py` | diagnoses `smiles` / `c-smiles` / `canonicalized` |
| `extract_smiles.py` | `.jsonl` → `.smi` shards |
| `smiles_to_json.py` | `.jsonl` → one JSON array (ad-hoc, not part of the pipeline) |
| `fg_matrix.py` | `.smi` → Parquet FG matrix — **the long step** |
| `export_csv.py` | Parquet → CSV |
| `accfg/spreadsheet.py` | original; fine for small runs, not for the corpus |
| `extract.lsf`, `fg.lsf` | LSF job scripts for extraction and the FG matrix (`bsub < file`) |

## If something goes wrong

- **`ArrowInvalid: Parquet magic bytes not found`** — a non-Parquet file is sitting among
  the parts. Rejects belong in `_rejects/`; the leading underscore is what makes
  `ds.dataset()` skip them.
- **`_columns.json disagrees with the current analyser`** — you changed `dict_fgs`
  mid-run. Mixing schemas would make the parts unmergeable, so it refuses. Use a fresh
  `--outdir`.
- **Run is slower than the pilot predicted** — check you aren't swapping or oversubscribed.
  Each worker peaks at about 210 MB; if memory keeps climbing past that, it's the analyser
  holding state, not the writer. On a cluster, check the log's first line: the worker
  count should match your `bsub -n`.
- **Job was killed (time limit, node failure)** — resubmit the same job. A temp file from a
  part that was being written is named `.<part>.partial`. The leading dot keeps it out of
  `ds.dataset()`, and the resumed run overwrites it.
- **`ImportError: libXext.so.6` / `libXrender.so.1`** — `fg_matrix.py` handles this itself
  and prints a `note:` line (step 1). If you still see it as a crash, something is
  calling plain `import accfg` (e.g. `spreadsheet.py`). Use the conda command in step 1.
- **`Can't pickle local object 'build_afg.<locals>.CachedAccFG'`** in the rejects, or a
  crawl at ~2 mol/s — AccFG 0.0.7's per-molecule process pool is active, meaning you are
  running an older copy of `fg_matrix.py`. Copy the current one to the cluster.
- **Every FG column is zero** — check the analyser by hand:
  `python -c "import fg_matrix as m; from rdkit import Chem; print(m.fg_vector(m.build_afg(), Chem.MolFromSmiles('CC(=O)O')))"`
  should list carboxylic-acid groups.
