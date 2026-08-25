"""
Fetch every dataset this project uses into Dataset/.

Public sets come from Kaggle (needs an API token, see README). The derived CFPB
files are rebuilt locally from the full CFPB database rather than downloaded.

Usage:
    python tools/fetch_data.py              # everything
    python tools/fetch_data.py --only ieee  # one dataset
    python tools/fetch_data.py --parquet    # convert to parquet when done
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Dataset"

# name -> (kaggle ref, is_competition, target subdir, a file that proves it landed)
KAGGLE = {
    "ieee":    ("ieee-fraud-detection",          True,  "ieee-fraud-detection", "train_transaction.csv"),
    "sparkov": ("kartik2112/fraud-detection",    False, "sparkov",              "fraudTrain.csv"),
    "paysim":  ("ealaxi/paysim1",                False, "paysim",               None),
}

CFPB_BULK = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"


def have_kaggle():
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def done(sub, marker):
    d = DATA / sub
    if not d.is_dir():
        return False
    if marker:
        return (d / marker).exists() or (d / marker).with_suffix(".parquet").exists()
    return any(d.iterdir())


def fetch_kaggle(name):
    ref, is_comp, sub, marker = KAGGLE[name]
    target = DATA / sub

    if done(sub, marker):
        print(f"  {name}: already present, skipping")
        return

    target.mkdir(parents=True, exist_ok=True)
    cmd = (["kaggle", "competitions", "download", "-c", ref]
           if is_comp else
           ["kaggle", "datasets", "download", "-d", ref])
    cmd += ["-p", str(target)]

    print(f"  {name}: downloading {ref} ...")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  {name}: FAILED. For competitions you must accept the rules "
              f"at https://www.kaggle.com/c/{ref} first.", file=sys.stderr)
        return

    for z in target.glob("*.zip"):
        print(f"  {name}: extracting {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(target)
        z.unlink()
    print(f"  {name}: done")


def cfpb_note():
    print("  cfpb: derived files are built locally, not downloaded.")
    print(f"        1. download {CFPB_BULK}  (~2-3GB zipped)")
    print("        2. extract complaints.csv into Dataset/complaints_raw/")
    print("        3. python tools/filter_cfpb.py")
    print("        Or grab the prebuilt parquets from the team Kaggle dataset")
    print("        (see README) and drop them in Dataset/complaints/.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(KAGGLE) + ["cfpb"])
    ap.add_argument("--parquet", action="store_true",
                    help="run tools/csv_to_parquet.py afterwards")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    names = [args.only] if args.only else list(KAGGLE) + ["cfpb"]

    if any(n in KAGGLE for n in names) and not have_kaggle():
        sys.exit("kaggle CLI not found.  pip install kaggle  and place your "
                 "kaggle.json in %USERPROFILE%\\.kaggle\\  (see README)")

    for n in names:
        if n == "cfpb":
            cfpb_note()
        else:
            fetch_kaggle(n)

    if args.parquet:
        print("\nconverting to parquet ...")
        subprocess.run([sys.executable, str(ROOT / "tools" / "csv_to_parquet.py")])


if __name__ == "__main__":
    main()
