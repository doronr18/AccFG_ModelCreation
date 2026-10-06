#!/usr/bin/env python3
"""Final step: export the Parquet dataset from fg_matrix.py to one CSV.

Produces the shape spreadsheet.py's fg_presence_dataframe returns -- `Molecule`, then one
0/1 column per functional group in afg.dict_fgs order -- with a leading `cid` column.

Streams batch by batch, so RAM stays flat regardless of row count.

    python export_csv.py /data/fg -o /data/fg_matrix.csv

At 177M rows a CSV is ~43 GB and nothing will open it as a spreadsheet. Prefer querying
the Parquet directly unless something downstream truly requires CSV:

    duckdb -c "SELECT * FROM '/data/fg/*.parquet' WHERE amide=1 AND alcohol=1"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow.csv as pcsv
import pyarrow.dataset as ds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", type=Path, help="fg_matrix.py output directory")
    ap.add_argument("-o", "--out", type=Path, required=True, help="output .csv path")
    ap.add_argument("--batch-rows", type=int, default=65536,
                    help="rows per streamed batch (default 65536)")
    ap.add_argument("--gzip", action="store_true",
                    help="write .csv.gz instead -- roughly 10x smaller, still streamed")
    args = ap.parse_args()

    dataset = ds.dataset(args.indir, format="parquet")
    schema = dataset.schema
    n_key = sum(name in ("cid", "Molecule") for name in schema.names)
    print(f"{len(schema.names)} columns: {', '.join(schema.names[:n_key])} + "
          f"{len(schema.names)-n_key} functional groups")

    out = args.out
    if args.gzip and out.suffix != ".gz":
        out = out.with_suffix(out.suffix + ".gz")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    t0 = time.perf_counter()
    # CSVWriter streams; never materialises the whole table.
    with pcsv.CSVWriter(out, schema) as w:
        for batch in dataset.scanner(batch_size=args.batch_rows).to_batches():
            w.write(batch)
            rows += batch.num_rows
            if rows % (args.batch_rows * 50) == 0:
                el = time.perf_counter() - t0
                print(f"  {rows:,} rows  ({rows/el/1000:,.0f}k rows/s)", flush=True)

    el = time.perf_counter() - t0
    sz = out.stat().st_size
    print(f"\nwrote {out}")
    print(f"  {rows:,} rows in {el:.1f}s  ({sz:,} bytes, {sz/max(rows,1):.1f} B/row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
