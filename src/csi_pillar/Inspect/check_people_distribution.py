"""
check_presence_distribution.py

Prints the presence distribution across splits, in TWO forms:
  1. The original people-count labels (0 / 1 / 2 / 3) -- shows the
     underlying data before deriving a binary label from it.
  2. The binary derived version (no person / person present) currently
     used by your warm-started presence model -- shows whether
     merging 1, 2, and 3 people into a single "person present" class
     is balanced against "no person" (0).

Usage:
    python check_presence_distribution.py --data "data/training_dataset_original.h5"
"""

import argparse
import numpy as np
import h5py


def print_distribution(counts, labels, total):
    print(f"{'Class':<20}{'Number of samples':<20}{'Percentage':<12}")
    for i, label in enumerate(labels):
        count = counts[i]
        percentage = (count / total) * 100
        print(f"{label:<20}{count:<20}{percentage:.2f}%")

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
            y_people = hf[split_name]["y_people"][:]
            total = len(y_people)

            print(f"\n{'=' * 55}")
            print(f"SPLIT: {split_name}  (total samples: {total})")
            print(f"{'=' * 55}")

            print("\n--- Original people-count breakdown (0/1/2/3) ---")
            counts_4class = np.bincount(y_people, minlength=4)
            print_distribution(counts_4class, ["0 people", "1 person", "2 people", "3 people"], total)

            print("\n--- Derived binary breakdown (no person vs present) ---")
            y_binary = (y_people > 0).astype(np.int64)
            counts_binary = np.bincount(y_binary, minlength=2)
            print_distribution(counts_binary, ["no person", "present"], total)


if __name__ == "__main__":
    main()