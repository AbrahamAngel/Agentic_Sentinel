"""
tune_threshold.py

Tests several decision thresholds on an already-trained model, without
retraining anything. Useful when a model's precision is high but recall
is low (like machine_onoff: 100% precision, 27% recall on "on") --
this often means the model IS picking up on the right signal, just not
confidently enough to cross the default 0.5 cutoff. Lowering the
threshold can trade some precision for meaningfully better recall.

Usage:
    python tune_threshold.py --model "final_model_machine_onoff_warmstart.keras" --data "data/training_dataset_original.h5" --labels off,on
"""

import argparse
import numpy as np
import h5py
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report


def derive_machine_onoff(y_machine):
    return np.isin(y_machine, [1, 2]).astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels", default="off,on")
    args = parser.parse_args()

    label_names = args.labels.split(",")

    print(f"Loading model from {args.model} ...")
    model = keras.models.load_model(args.model)

    with h5py.File(args.data, "r") as hf:
        X_test = hf["test"]["X"][:]
        y_machine = hf["test"]["y_machine"][:]

    y_true = derive_machine_onoff(y_machine)

    print("Running predictions once (reused across all thresholds) ...")
    probs = model.predict(X_test, batch_size=64).flatten()

    for threshold in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
        y_pred = (probs > threshold).astype(np.int64)
        print(f"\n{'=' * 50}")
        print(f"THRESHOLD = {threshold}")
        print(f"{'=' * 50}")
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))


if __name__ == "__main__":
    main()