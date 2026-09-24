"""
train_presence_kfold.py

Runs Stratified Group 5-Fold cross-validation for the presence task,
combined with real-sample oversampling of the minority "no person"
class within each fold's training data.

Why both techniques together:
  - Stratified Group K-Fold ensures each fold's TEST set has a fair,
    representative mix of "no person"/"present" files (same reasoning
    as machine on/off -- presence is a per-file label, vulnerable to
    the same kind of accidental split-imbalance we measured earlier).
  - Oversampling addresses the actual root cause of presence's weak,
    uncertain confidence scores: a real 74%/26% imbalance in the
    underlying data itself. Stratification alone only makes
    *measurement* fair -- it doesn't fix what the model learns during
    TRAINING, which is why oversampling is added on top.

Oversampling method: REAL sample duplication (not SMOTE). Minority-
class blocks in the training set are randomly duplicated (with
replacement) until roughly matching the majority class count. This
avoids SMOTE's synthetic interpolation, which risks creating
physically implausible CSI signal patterns (structured signal data
doesn't interpolate the way simple tabular data does).

IMPORTANT: oversampling is applied ONLY to the training portion of
each fold, never to the test portion -- the test set must reflect
real, natural class proportions to give an honest accuracy measure.

Usage:
    python train_presence_kfold.py --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 15
"""

import argparse
import random

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, classification_report


BLOCK_SIZE = 25
BATCH_SIZE = 64
SEED = 42


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_file_label(filepath):
    """Presence label for this whole file: 1 if any person was present, else 0."""
    data = np.load(filepath, allow_pickle=True)
    n_people = int(data["n_people"]) if data["n_people"].size > 0 else 0
    return 1 if n_people > 0 else 0


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

    n_people = int(data["n_people"]) if data["n_people"].size > 0 else 0
    label_value = 1 if n_people > 0 else 0
    labels = np.full(num_blocks, label_value, dtype=np.float32)
    return blocks, labels


def oversample_minority(X_train, y_train, seed=SEED):
    """
    Duplicates real minority-class blocks (with replacement) until the
    training set is roughly balanced. No synthetic data is created --
    every duplicated sample is a genuine, real CSI block.

    Pre-allocates the final array ONCE and fills it in chunks, instead
    of using np.concatenate -- which would otherwise briefly need
    memory for the original array, a copy of the duplicated blocks,
    AND the combined result all at once (roughly 3x the dataset size),
    causing severe slowdowns from disk swapping on memory-constrained
    machines. This version peaks at roughly 2x instead.
    """
    rng = np.random.RandomState(seed)

    class_0_idx = np.where(y_train == 0)[0]
    class_1_idx = np.where(y_train == 1)[0]

    count_0 = len(class_0_idx)
    count_1 = len(class_1_idx)

    if count_0 == count_1:
        return X_train, y_train  # already balanced, nothing to do

    minority_idx, majority_count = (class_0_idx, count_1) if count_0 < count_1 else (class_1_idx, count_0)
    minority_count = len(minority_idx)
    num_needed = majority_count - minority_count

    original_count = X_train.shape[0]
    final_count = original_count + num_needed

    print(f"  Oversampling: allocating space for {final_count} total blocks "
          f"(adding {num_needed} duplicated minority blocks) ...")

    X_final = np.empty((final_count,) + X_train.shape[1:], dtype=X_train.dtype)
    y_final = np.empty(final_count, dtype=y_train.dtype)

    X_final[:original_count] = X_train
    y_final[:original_count] = y_train

    duplicated_idx = rng.choice(minority_idx, size=num_needed, replace=True)

    # Copy duplicated blocks in small chunks, avoiding one large
    # temporary copy of all num_needed blocks at once
    chunk_size = 5000
    for start in range(0, num_needed, chunk_size):
        end = min(start + chunk_size, num_needed)
        idx_chunk = duplicated_idx[start:end]
        X_final[original_count + start: original_count + end] = X_train[idx_chunk]
        y_final[original_count + start: original_count + end] = y_train[idx_chunk]

    print("  Oversampling: shuffling combined training set ...")
    shuffle_idx = rng.permutation(final_count)
    return X_final[shuffle_idx], y_final[shuffle_idx]


def build_warmstart_model(base_model_path):
    base_model = keras.models.load_model(base_model_path)
    shared_output = base_model.get_layer("dense_1").output
    new_output = layers.Dense(1, activation="sigmoid", name="presence_output")(shared_output)
    return keras.Model(inputs=base_model.input, outputs=new_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder of processed .npz files (mc3_processed)")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--start_fold", type=int, default=1,
                         help="Skip folds before this number (e.g. 2 to resume after Fold 1 already completed)")
    args = parser.parse_args()

    set_all_seeds()

    source_root = Path(args.source)
    all_files = sorted(source_root.glob("*_processed.npz"))

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
    groups = np.arange(len(valid_files))

    print(f"Found {len(valid_files)} valid (20 MHz) files")
    print(f"Label distribution across files: no_person={np.sum(file_labels==0)}, present={np.sum(file_labels==1)}")

    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(valid_files, file_labels, groups), start=1):
        if fold_idx < args.start_fold:
            print(f"Skipping Fold {fold_idx}/{args.n_splits} (already completed in a previous run) ...")
            continue

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

        print(f"Train blocks BEFORE oversampling: {X_train.shape[0]} "
              f"(no_person={np.sum(y_train==0)}, present={np.sum(y_train==1)})")

        X_train, y_train = oversample_minority(X_train, y_train)

        print(f"Train blocks AFTER oversampling:  {X_train.shape[0]} "
              f"(no_person={np.sum(y_train==0)}, present={np.sum(y_train==1)})")

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

        print(f"Test blocks (NOT oversampled, natural distribution): {X_test.shape[0]}")

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
        y_pred = (probs > 0.5).astype(np.int64)

        acc = np.mean(y_pred == y_test)
        print(f"\nFold {fold_idx} test accuracy (threshold=0.5): {acc:.4f}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=["no person", "present"], zero_division=0))
        print(f"Sample confidence scores (first 10 test predictions): {np.round(probs[:10], 3)}")

        # Full distribution check -- settles whether oversampling is causing
        # genuine saturation/overconfidence, rather than guessing from 10 samples
        near_zero = np.sum(probs < 0.01)
        near_one = np.sum(probs > 0.99)
        middle = np.sum((probs >= 0.01) & (probs <= 0.99))
        total = len(probs)
        print(f"\nFull score distribution for Fold {fold_idx}:")
        print(f"  Min: {probs.min():.4f}, Max: {probs.max():.4f}, Mean: {probs.mean():.4f}")
        print(f"  Scores < 0.01: {near_zero} ({100*near_zero/total:.1f}%)")
        print(f"  Scores > 0.99: {near_one} ({100*near_one/total:.1f}%)")
        print(f"  Scores 0.01-0.99: {middle} ({100*middle/total:.1f}%)")
        if (near_zero + near_one) / total > 0.9:
            print("  -> WARNING: over 90% saturated near 0/1 -- possible oversampling overconfidence.")
        else:
            print("  -> Genuine spread -- calibration looks healthy for this fold.")

        fold_results.append(acc)

        del X_train, y_train, X_test, y_test, model

    print(f"\n{'=' * 60}")
    print("FINAL CROSS-VALIDATION SUMMARY (presence, with oversampling)")
    print(f"{'=' * 60}")
    print(f"Per-fold accuracies: {[f'{a:.4f}' for a in fold_results]}")
    print(f"Mean accuracy: {np.mean(fold_results):.4f}")
    print(f"Std deviation:  {np.std(fold_results):.4f}")
    print("\n(Compare this mean accuracy AND the sample confidence scores above")
    print(" against your original single-split result -- oversampling should")
    print(" show more decisive scores, not just clustered near 0.5.)")


if __name__ == "__main__":
    main()