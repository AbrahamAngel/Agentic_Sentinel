"""
train_machine_onoff_kfold.py

Runs Stratified Group 5-Fold cross-validation for the machine on/off
task. "Group" = file (all blocks from one recording stay together in
the same fold, preventing the data leakage we fixed earlier in this
project). "Stratified" = each fold gets a representative mix of
off/on files, not a random skew like we found in our single-split
validation set.

This trains the model 5 separate times (once per fold), each time
warm-starting fresh from the same base multi-task model, and reports
the average +/- spread of accuracy across all 5 -- telling you how
reliable your machine on/off result really is, not just one lucky
(or unlucky) split.

NOTE: this takes roughly 5x as long as a single training run. Expect
several hours to a day depending on your hardware.

Usage:
    python train_machine_onoff_kfold.py --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 15
"""

import argparse
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report
import random


BLOCK_SIZE = 25
BATCH_SIZE = 64
SEED = 42


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_file_label(filepath):
    """Reads just the metadata needed to determine this file's machine on/off label."""
    data = np.load(filepath, allow_pickle=True)
    n_machine = int(data["n_machine"]) if data["n_machine"].size > 0 else 0
    status = str(data["status"])
    if n_machine == 0 or status == "":
        return 0  # off
    return 1  # on (covers both on-idle and running)


def load_file_blocks(filepath):
    """Loads one file's amplitude/phase and splits into blocks (same as before)."""
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

    label = get_file_label(filepath)
    labels = np.full(num_blocks, label, dtype=np.float32)
    return blocks, labels


def build_warmstart_model(base_model_path):
    base_model = keras.models.load_model(base_model_path)
    shared_output = base_model.get_layer("dense_1").output
    new_output = layers.Dense(1, activation="sigmoid", name="machine_onoff_output")(shared_output)
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

    # Only keep 64-subcarrier (20 MHz) files, and get each file's label for stratification
    valid_files = []
    file_labels = []
    print("Scanning files to determine labels for stratification ...")
    for f in all_files:
        data = np.load(f, allow_pickle=True)
        if data["amplitude"].shape[1] != 64:
            continue
        valid_files.append(f)
        file_labels.append(get_file_label(f))

    valid_files = np.array(valid_files)
    file_labels = np.array(file_labels)
    groups = np.arange(len(valid_files))  # each file is its own group

    print(f"Found {len(valid_files)} valid (20 MHz) files")
    print(f"Label distribution across files: off={np.sum(file_labels==0)}, on={np.sum(file_labels==1)}")

    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(valid_files, file_labels, groups), start=1):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold_idx}/{args.n_splits}")
        print(f"{'=' * 60}")

        train_files = valid_files[train_idx]
        test_files = valid_files[test_idx]
        print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")

        # Build train blocks (in memory -- fine at this scale per fold)
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
        y_pred = (probs > 0.25).astype(np.int64)  # using your confirmed threshold

        acc = np.mean(y_pred == y_test)
        print(f"\nFold {fold_idx} test accuracy (threshold=0.25): {acc:.4f}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=["off", "on"], zero_division=0))

        fold_results.append(acc)

        del X_train, y_train, X_test, y_test, model

    print(f"\n{'=' * 60}")
    print("FINAL CROSS-VALIDATION SUMMARY (machine on/off)")
    print(f"{'=' * 60}")
    print(f"Per-fold accuracies: {[f'{a:.4f}' for a in fold_results]}")
    print(f"Mean accuracy: {np.mean(fold_results):.4f}")
    print(f"Std deviation:  {np.std(fold_results):.4f}")
    print("\n(A small std deviation means your result is stable across different")
    print(" data splits. A large one means the single-split result you reported")
    print(" earlier may not be fully representative -- similar to what we saw")
    print(" with movement's 82% vs 62% reproduction gap.)")


if __name__ == "__main__":
    main()