"""
train_movement_kfold.py

Runs plain Group 5-Fold cross-validation for the movement task.
"Group" = file (all blocks from one recording stay together in the
same fold, same leakage protection as before). No stratification is
used here, because movement labels are computed per-file using a
median variance threshold -- meaning every single file already
contributes a roughly 50/50 mix of "moving"/"stationary" blocks by
construction, so stratifying on top of that adds negligible value.

Trains the model 5 separate times, warm-starting fresh each time from
the same base multi-task model, and reports mean +/- std accuracy
across folds -- directly relevant given movement previously showed
real instability (82% vs 62% on separate single-split runs).

Usage:
    python train_movement_kfold.py --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 15
"""

import argparse
import random

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, classification_report


BLOCK_SIZE = 25
BATCH_SIZE = 64
SEED = 42


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_movement_labels(amplitude, block_size=BLOCK_SIZE):
    """Same per-file median-variance heuristic used throughout this project."""
    num_readings, num_subcarriers = amplitude.shape
    num_blocks = num_readings // block_size

    variances = np.zeros(num_blocks, dtype=np.float32)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        variances[b] = np.var(amplitude[start:end, :], axis=0).mean()

    threshold = np.median(variances)
    return (variances > threshold).astype(np.float32)


def load_file_blocks(filepath):
    data = np.load(filepath, allow_pickle=True)
    amplitude = data["amplitude"]
    phase = data["phase"]

    if amplitude.shape[1] != 64:
        return None

    num_readings, num_subcarriers = amplitude.shape
    num_blocks = num_readings // BLOCK_SIZE

    blocks = np.zeros((num_blocks, num_subcarriers, BLOCK_SIZE, 2), dtype=np.float32)
    for b in range(num_blocks):
        start = b * BLOCK_SIZE
        end = start + BLOCK_SIZE
        blocks[b, :, :, 0] = amplitude[start:end, :].T
        blocks[b, :, :, 1] = phase[start:end, :].T

    labels = compute_movement_labels(amplitude)
    return blocks, labels


def build_warmstart_model(base_model_path):
    base_model = keras.models.load_model(base_model_path)
    shared_output = base_model.get_layer("dense_1").output
    new_output = layers.Dense(1, activation="sigmoid", name="movement_output")(shared_output)
    return keras.Model(inputs=base_model.input, outputs=new_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder of processed .npz files (mc3_processed)")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    set_all_seeds()

    source_root = Path(args.source)
    all_files = sorted(source_root.glob("*_processed.npz"))

    valid_files = []
    print("Scanning files for valid (20 MHz) recordings ...")
    for f in all_files:
        data = np.load(f, allow_pickle=True)
        if data["amplitude"].shape[1] == 64:
            valid_files.append(f)

    valid_files = np.array(valid_files)
    groups = np.arange(len(valid_files))  # each file is its own group

    print(f"Found {len(valid_files)} valid (20 MHz) files")

    gkf = GroupKFold(n_splits=args.n_splits)

    fold_results = []

    # GroupKFold's split() just needs X and groups; we pass a dummy y
    dummy_y = np.zeros(len(valid_files))

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(valid_files, dummy_y, groups), start=1):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx}/{args.n_splits}")
        print(f"{'=' * 60}")

        train_files = valid_files[train_idx]
        test_files = valid_files[test_idx]
        print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

        X_train_list, y_train_list = [], []
        for f in train_files:
            result = load_file_blocks(f)
            if result is not None:
                blocks, labels = result
                X_train_list.append(blocks)
                y_train_list.append(labels)
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        del X_train_list, y_train_list

        X_test_list, y_test_list = [], []
        for f in test_files:
            result = load_file_blocks(f)
            if result is not None:
                blocks, labels = result
                X_test_list.append(blocks)
                y_test_list.append(labels)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        del X_test_list, y_test_list

        print(f"Train blocks: {X_train.shape[0]}, Test blocks: {X_test.shape[0]}")

        model = build_warmstart_model(args.base_model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ]

        model.fit(
            X_train, y_train,
            validation_split=0.15,
            batch_size=BATCH_SIZE,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        probs = model.predict(X_test, batch_size=BATCH_SIZE).flatten()
        y_pred = (probs > 0.5).astype(np.int64)  # default threshold -- movement didn't need tuning

        acc = np.mean(y_pred == y_test)
        print(f"\nFold {fold_idx} test accuracy (threshold=0.5): {acc:.4f}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=["stationary", "moving"], zero_division=0))

        fold_results.append(acc)

        del X_train, y_train, X_test, y_test, model

    print(f"\n{'=' * 60}")
    print("FINAL CROSS-VALIDATION SUMMARY (movement)")
    print(f"{'=' * 60}")
    print(f"Per-fold accuracies: {[f'{a:.4f}' for a in fold_results]}")
    print(f"Mean accuracy: {np.mean(fold_results):.4f}")
    print(f"Std deviation:  {np.std(fold_results):.4f}")
    print("\n(Given the earlier 82% vs 62% single-split reproduction gap, this")
    print(" standard deviation is the key number -- a large spread would confirm")
    print(" real instability in this task, not just bad luck on one run.)")


if __name__ == "__main__":
    main()