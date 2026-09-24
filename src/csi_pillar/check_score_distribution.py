"""
check_score_distribution.py

Checks whether a model's confidence scores are genuinely well-spread
(good calibration) or artificially saturated near 0 and 1 (a risk with
duplication-based oversampling, where the model may partly memorize
repeated samples rather than learn a genuine, graded decision
boundary).

Usage:
    python check_score_distribution.py --model "best_model_presence_warmstart.keras" --data "data/training_dataset_original.h5"
"""

import argparse
import numpy as np
import h5py
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = keras.models.load_model(args.model)

    with h5py.File(args.data, "r") as hf:
        X_test = hf["test"]["X"][:]

    print("Running predictions ...")
    probs = model.predict(X_test, batch_size=64).flatten()

    print(f"\nTotal test predictions: {len(probs)}")
    print(f"Min: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}")

    # Check how many scores are "saturated" near the extremes
    near_zero = np.sum(probs < 0.01)
    near_one = np.sum(probs > 0.99)
    middle = np.sum((probs >= 0.01) & (probs <= 0.99))

    print(f"\nScores < 0.01 (essentially 0):  {near_zero} ({100*near_zero/len(probs):.1f}%)")
    print(f"Scores > 0.99 (essentially 1):  {near_one} ({100*near_one/len(probs):.1f}%)")
    print(f"Scores in between (0.01-0.99): {middle} ({100*middle/len(probs):.1f}%)")

    if (near_zero + near_one) / len(probs) > 0.9:
        print("\n-> WARNING: over 90% of scores are saturated near 0 or 1.")
        print("   This suggests possible overconfidence, possibly from")
        print("   duplication-based oversampling causing the model to")
        print("   memorize repeated samples rather than learn a genuinely")
        print("   graded decision boundary. Scores may not be well-calibrated")
        print("   even if raw accuracy looks good.")
    else:
        print("\n-> Scores show genuine spread across the range -- good sign")
        print("   for calibration, not just accuracy.")


if __name__ == "__main__":
    main()