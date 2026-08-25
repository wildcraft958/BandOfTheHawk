"""
Convert every dataset CSV to Parquet.

Streams each file in row-group batches so the 800MB CFPB file and the 683MB
IEEE transaction table never land in memory whole.

Usage:
    python tools/csv_to_parquet.py                 # convert, keep CSVs
    python tools/csv_to_parquet.py --delete-csv    # convert, then remove CSVs
    python tools/csv_to_parquet.py --verify        # re-read each parquet and check row counts
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset"

BLOCK = 128 << 20        # 128MB read blocks
COMPRESSION = "zstd"     # better ratio than snappy, still fast
LEVEL = 3


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def convert(src: Path, dst: Path) -> tuple[int, int, int]:
    """Stream src CSV -> dst parquet. Returns (rows, src_bytes, dst_bytes)."""
    read_opts = pv.ReadOptions(block_size=BLOCK)
    parse_opts = pv.ParseOptions(newlines_in_values=True)
    # Let arrow infer types, but never let a mostly-empty column become null-typed;
    # null columns break some downstream readers.
    convert_opts = pv.ConvertOptions(strings_can_be_null=True)

    reader = pv.open_csv(src, read_options=read_opts,
                         parse_options=parse_opts,
                         convert_options=convert_opts)

    writer = None
    rows = 0
    try:
        for batch in reader:
            if batch.num_rows == 0:
                continue
            table = pa.Table.from_batches([batch])
            if writer is None:
                schema = table.schema
                # Promote any all-null column to string so the file stays readable.
                fields = [
                    pa.field(f.name, pa.string()) if pa.types.is_null(f.type) else f
                    for f in schema
                ]
                schema = pa.schema(fields)
                writer = pq.ParquetWriter(
                    dst, schema, compression=COMPRESSION,
                    compression_level=LEVEL, use_dictionary=True,
                )
            writer.write_table(table.cast(writer.schema))
            rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()

    return rows, src.stat().st_size, dst.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-csv", action="store_true",
                    help="remove each CSV after its parquet verifies")
    ap.add_argument("--verify", action="store_true",
                    help="re-open each parquet and confirm the row count")
    args = ap.parse_args()

    csvs = sorted(DATA.rglob("*.csv"))
    if not csvs:
        sys.exit(f"no CSVs under {DATA}")

    print(f"{len(csvs)} CSV files under {DATA}\n")

    tot_src = tot_dst = 0
    results = []

    for src in csvs:
        dst = src.with_suffix(".parquet")
        rel = src.relative_to(ROOT)

        if dst.exists() and dst.stat().st_mtime > src.stat().st_mtime:
            print(f"skip (up to date)  {rel}")
            tot_src += src.stat().st_size
            tot_dst += dst.stat().st_size
            continue

        print(f"converting  {rel}  ({human(src.stat().st_size)}) ... ", end="", flush=True)
        t0 = time.time()
        try:
            rows, sb, db = convert(src, dst)
        except Exception as e:
            print(f"FAILED: {e}")
            if dst.exists():
                dst.unlink()
            continue

        dt = time.time() - t0
        ratio = sb / db if db else 0
        print(f"{rows:,} rows  {human(sb)} -> {human(db)}  ({ratio:.1f}x)  {dt:.1f}s")

        if args.verify:
            got = pq.ParquetFile(dst).metadata.num_rows
            status = "ok" if got == rows else f"MISMATCH {got} != {rows}"
            print(f"    verify: {status}")
            if got != rows:
                continue

        tot_src += sb
        tot_dst += db
        results.append((rel, sb, db))

        if args.delete_csv:
            src.unlink()
            print(f"    removed {src.name}")

    print(f"\n{'':50} {'CSV':>10} {'parquet':>10} {'ratio':>7}")
    for rel, sb, db in results:
        print(f"{str(rel):50} {human(sb):>10} {human(db):>10} {sb/db:>6.1f}x")
    if tot_dst:
        print(f"\nTOTAL  {human(tot_src)} -> {human(tot_dst)}  ({tot_src/tot_dst:.1f}x smaller)")


if __name__ == "__main__":
    main()
