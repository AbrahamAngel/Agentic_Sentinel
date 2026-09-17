"""
check_machine_distribution.py

Prints the machine-status distribution across splits, in TWO forms:
  1. The original 3-class labels (off / on-idle / running) -- shows
     the severe imbalance that caused all the earlier training
     instability (the "on-idle" collapse).
  2. The binary merged version (off / on) currently used by your
     warm-started machine_onoff model -- shows whether merging
     actually fixed the imbalance, as intended.

Usage:
    python check_machine_distribution.py --data "data/training_dataset_original.h5"
"""

import argparse
import numpy as np
import h5py


def print_distribution(counts, labels, total):
    print(f"{'Class':<15}{'Number of samples':<20}{'Percentage':<12}")
    for i, label in enumerate(labels):
        count = counts[i]
        percentage = (count / total) * 100
        print(f"{label:<15}{count:<20}{percentage:.2f}%")

    max_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = max_count / max(min_count, 1)

    print(f"\nLargest class: {labels[counts.argmax()]} ({max_count} samples)")
    print(f"Smallest class: {labels[counts.argmin()]} ({min_count} samples)")
    print(f"Imbalance ratio (largest / smallest): {imbalance_ratio:.2f}x")

    if imbalance_ratio > 3:
        print("  -> Meaningful imbalance: consider this when interpreting accuracy per class.")
    elif imbalance_ratio > 1.5:
        print("  -> Mild imbalance, worth noting but not severe.")
    else:
        print("  -> Roughly balanced.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    with h5py.File(args.data, "r") as hf:
        for split_name in ["train", "val", "test"]:
            y_machine = hf[split_name]["y_machine"][:]
            total = len(y_machine)

            print(f"\n{'=' * 55}")
            print(f"SPLIT: {split_name}  (total samples: {total})")
            print(f"{'=' * 55}")

            print("\n--- Original 3-class breakdown ---")
            counts_3class = np.bincount(y_machine, minlength=3)
            print_distribution(counts_3class, ["off", "on/idle", "running"], total)

            print("\n--- Merged binary breakdown (off vs on) ---")
            y_binary = np.isin(y_machine, [1, 2]).astype(np.int64)
            counts_binary = np.bincount(y_binary, minlength=2)
            print_distribution(counts_binary, ["off", "on"], total)


if __name__ == "__main__":
    main()