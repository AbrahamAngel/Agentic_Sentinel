"""
filter_mc3_files.py

After extracting the full EHUNAM zip, this script copies only the files
listed in mc3_industrial_files.csv (your filtered MC3 industrial-environment
list) into a clean, smaller folder -- so you don't have to work with all
2,401 files, just the ~678 relevant to your project.

Usage:
    python filter_mc3_files.py \
        --source "D:/EHUNAM_extracted" \
        --dest "D:/angel/Web/Agentic_sentinel/csi_pillar/data/mc3" \
        --csv "mc3_industrial_files.csv"

Adjust --source to wherever you extracted the big zip.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def find_file_in_source(filename, source_root):
    """
    The extracted EHUNAM zip may have files nested inside subfolders
    per campaign/set rather than all in one flat folder. This does a
    recursive search so it doesn't matter how it's organized.
    """
    matches = list(source_root.rglob(filename))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder where the full EHUNAM zip was extracted")
    parser.add_argument("--dest", required=True, help="Folder to copy the filtered MC3 files into")
    parser.add_argument("--csv", default="mc3_industrial_files.csv", help="Path to the filtered file list CSV")
    args = parser.parse_args()

    source_root = Path(args.source)
    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    filenames = df["File"].tolist()

    print(f"Looking for {len(filenames)} files under {source_root} ...")

    found_count = 0
    missing = []

    for filename in filenames:
        src_path = find_file_in_source(filename, source_root)
        if src_path is None:
            missing.append(filename)
            continue

        dest_path = dest_root / filename
        shutil.copy2(src_path, dest_path)
        found_count += 1

        if found_count % 50 == 0:
            print(f"  copied {found_count} files so far...")

    print(f"\nDone. Copied {found_count} / {len(filenames)} files to {dest_root}")

    if missing:
        print(f"\n{len(missing)} files were NOT found. First few missing:")
        for m in missing[:10]:
            print(f"  - {m}")
        print("\nIf many are missing, double-check the --source path points")
        print("to the actual extracted folder (not the .zip itself).")


if __name__ == "__main__":
    main()