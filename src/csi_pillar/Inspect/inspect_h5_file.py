"""
inspect_h5_file.py

Loads an HDF5 file (like training_dataset.h5) and prints its structure --
what groups/datasets exist, their shapes, and a few sample values --
so you can actually "see" what's inside, similar to inspect_mat_file.py
for the raw .mat files.

Usage:
    python inspect_h5_file.py "data/training_dataset.h5"
"""

import sys
import h5py
import numpy as np


def inspect_h5_file(filepath):
    print(f"Inspecting: {filepath}\n")

    with h5py.File(filepath, "r") as hf:
        print("Top-level groups:", list(hf.keys()))
        print()

        for group_name in hf.keys():
            group = hf[group_name]
            print(f"=== Group: {group_name} ===")

            for dataset_name in group.keys():
                ds = group[dataset_name]
                print(f"  {dataset_name}: shape={ds.shape}, dtype={ds.dtype}")

                # Show a small sample of actual values
                if ds.ndim == 1:
                    sample = ds[:5]
                    print(f"    first 5 values: {sample}")
                else:
                    sample = ds[0]
                    print(f"    first sample shape: {sample.shape}")
                    print(f"    first sample min/max: {sample.min():.3f} / {sample.max():.3f}")

            print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python inspect_h5_file.py "path/to/file.h5"')
        sys.exit(1)

    inspect_h5_file(sys.argv[1])