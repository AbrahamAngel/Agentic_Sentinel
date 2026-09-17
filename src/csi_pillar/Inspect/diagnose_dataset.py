"""
diagnose_dataset.py

Investigates the two most likely causes of lower-than-expected accuracy:

1. Amplitude/phase SCALE MISMATCH -- amplitude values are unnormalized
   (can be in the thousands), while phase is naturally small (-pi to pi).
   Feeding both into a CNN without normalizing lets the amplitude channel
   dominate the learning signal.

2. CLASS IMBALANCE -- checks how skewed each label's distribution is,
   per split, which can bias the model toward over-predicting majority
   classes.

Usage:
    python diagnose_dataset.py "data/training_dataset.h5"
"""

import sys
import h5py
import numpy as np


def diagnose(filepath):
    print(f"Diagnosing: {filepath}\n")

    with h5py.File(filepath, "r") as hf:
        for split_name in ["train", "val", "test"]:
            print(f"{'=' * 60}")
            print(f"SPLIT: {split_name}")
            print(f"{'=' * 60}")

            X = hf[split_name]["X"]
            n = X.shape[0]

            # Sample a subset for speed, rather than loading everything
            sample_size = min(5000, n)
            sample_idx = np.sort(np.random.choice(n, sample_size, replace=False))
            X_sample = X[sample_idx]

            amplitude_channel = X_sample[..., 0]
            phase_channel = X_sample[..., 1]

            print(f"\n--- Scale check (based on {sample_size} random samples) ---")
            print(f"Amplitude channel: min={amplitude_channel.min():.3f}, "
                  f"max={amplitude_channel.max():.3f}, "
                  f"mean={amplitude_channel.mean():.3f}, "
                  f"std={amplitude_channel.std():.3f}")
            print(f"Phase channel:     min={phase_channel.min():.3f}, "
                  f"max={phase_channel.max():.3f}, "
                  f"mean={phase_channel.mean():.3f}, "
                  f"std={phase_channel.std():.3f}")

            scale_ratio = amplitude_channel.std() / max(phase_channel.std(), 1e-8)
            print(f"\nAmplitude std / Phase std ratio: {scale_ratio:.1f}x")
            if scale_ratio > 10:
                print("  -> WARNING: amplitude dominates phase by a large margin.")
                print("     The model may be learning almost entirely from amplitude,")
                print("     effectively ignoring phase information.")

            # Class balance check
            print(f"\n--- Class balance ---")
            for label_name in ["y_people", "y_movement", "y_machine"]:
                y = hf[split_name][label_name][:]
                counts = np.bincount(y)
                proportions = counts / counts.sum()
                print(f"{label_name}: counts={counts}, proportions={np.round(proportions, 3)}")

                max_prop = proportions.max()
                if max_prop > 0.6:
                    dominant_class = proportions.argmax()
                    print(f"  -> WARNING: class {dominant_class} makes up {max_prop*100:.1f}% "
                          f"of this split. A model predicting only this class would already "
                          f"score {max_prop*100:.1f}% accuracy without learning anything real.")

            print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python diagnose_dataset.py "path/to/training_dataset.h5"')
        sys.exit(1)

    diagnose(sys.argv[1])