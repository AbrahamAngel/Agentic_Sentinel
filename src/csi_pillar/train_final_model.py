"""
train_final_model.py

Trains the FINAL, production-ready model for one CSI task (presence,
movement, or machine_onoff), using your FULL dataset -- not a held-out
k-fold split. K-fold already gave you the honest evidence that each
approach generalizes well; this step produces the actual model file
you'll deploy and use for real confidence-score predictions.

Split: files are split 85% train / 15% val (file-level, no leakage).
There is no separate held-out test set here -- k-fold already served
that purpose. The val split is used only so EarlyStopping knows when
to stop training.

For presence specifically, this FIXES a bug found during k-fold: the
validation split is now taken BEFORE oversampling (at the file level),
so duplicated samples can never leak between train and validation.
Oversampling is applied ONLY to the training portion, exactly as it
should be.

Usage:
    python train_final_model.py --task presence --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 20
    python train_final_model.py --task movement --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 20
    python train_final_model.py --task machine_onoff --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 20
"""

import argparse
import random

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report


BLOCK_SIZE = 25
BATCH_SIZE = 64
SEED = 42

LABEL_NAMES = {
    "presence": ["no person", "present"],
    "movement": ["stationary", "moving"],
    "machine_onoff": ["off", "on"],
}

THRESHOLDS = {
    "presence": 0.5,
    "movement": 0.5,
    "machine_onoff": 0.25,  # your confirmed tuned threshold
}


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_file_label(filepath, task):
    """File-level label, used only for stratifying the train/val split."""
    data = np.load(filepath, allow_pickle=True)
    if task == "presence":
        n_people = int(data["n_people"]) if data["n_people"].size > 0 else 0
        return 1 if n_people > 0 else 0
    elif task == "machine_onoff":
        n_machine = int(data["n_machine"]) if data["n_machine"].size > 0 else 0
        status = str(data["status"])
        if n_machine == 0 or status == "":
            return 0
        return 1
    else:  # movement -- no meaningful single file-level label, not used for stratification
        return 0


def compute_movement_labels(amplitude, block_size=BLOCK_SIZE):
    num_readings, num_subcarriers = amplitude.shape
    num_blocks = num_readings // block_size
    variances = np.zeros(num_blocks, dtype=np.float32)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        variances[b] = np.var(amplitude[start:end, :], axis=0).mean()
    threshold = np.median(variances)
    return (variances > threshold).astype(np.float32)


def load_file_blocks(filepath, task):
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

    if task == "presence":
        n_people = int(data["n_people"]) if data["n_people"].size > 0 else 0
        label_value = 1 if n_people > 0 else 0
        labels = np.full(num_blocks, label_value, dtype=np.float32)
    elif task == "machine_onoff":
        n_machine = int(data["n_machine"]) if data["n_machine"].size > 0 else 0
        status = str(data["status"])
        label_value = 0 if (n_machine == 0 or status == "") else 1
        labels = np.full(num_blocks, label_value, dtype=np.float32)
    else:  # movement
        labels = compute_movement_labels(amplitude)

    return blocks, labels


def oversample_minority(X_train, y_train, seed=SEED):
    """Real-sample duplication oversampling (memory-efficient version)."""
    rng = np.random.RandomState(seed)

    class_0_idx = np.where(y_train == 0)[0]
    class_1_idx = np.where(y_train == 1)[0]
    count_0, count_1 = len(class_0_idx), len(class_1_idx)

    if count_0 == count_1:
        return X_train, y_train

    minority_idx, majority_count = (class_0_idx, count_1) if count_0 < count_1 else (class_1_idx, count_0)
    num_needed = majority_count - len(minority_idx)

    original_count = X_train.shape[0]
    final_count = original_count + num_needed

    print(f"  Oversampling training set: {original_count} -> {final_count} blocks "
          f"(+{num_needed} duplicated minority blocks)")

    X_final = np.empty((final_count,) + X_train.shape[1:], dtype=X_train.dtype)
    y_final = np.empty(final_count, dtype=y_train.dtype)
    X_final[:original_count] = X_train
    y_final[:original_count] = y_train

    duplicated_idx = rng.choice(minority_idx, size=num_needed, replace=True)
    chunk_size = 5000
    for start in range(0, num_needed, chunk_size):
        end = min(start + chunk_size, num_needed)
        idx_chunk = duplicated_idx[start:end]
        X_final[original_count + start: original_count + end] = X_train[idx_chunk]
        y_final[original_count + start: original_count + end] = y_train[idx_chunk]

    shuffle_idx = rng.permutation(final_count)
    return X_final[shuffle_idx], y_final[shuffle_idx]


def build_warmstart_model(base_model_path, task):
    base_model = keras.models.load_model(base_model_path)
    shared_output = base_model.get_layer("dense_1").output
    new_output = layers.Dense(1, activation="sigmoid", name=f"{task}_output")(shared_output)
    return keras.Model(inputs=base_model.input, outputs=new_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(LABEL_NAMES.keys()))
    parser.add_argument("--source", required=True, help="Folder of processed .npz files (mc3_processed)")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    set_all_seeds()

    source_root = Path(args.source)
    all_files = sorted(source_root.glob("*_processed.npz"))

    valid_files = []
    file_labels = []
    print(f"Scanning files for task '{args.task}' ...")
    for f in all_files:
        data = np.load(f, allow_pickle=True)
        if data["amplitude"].shape[1] != 64:
            continue
        valid_files.append(f)
        file_labels.append(get_file_label(f, args.task))

    valid_files = np.array(valid_files)
    file_labels = np.array(file_labels)
    print(f"Found {len(valid_files)} valid (20 MHz) files")

    # FILE-LEVEL train/val split, done BEFORE any oversampling -- this is
    # the fix for the leakage bug found during k-fold: validation files
    # are set aside first, so duplicated training samples can never end
    # up "leaking" into validation.
    if args.task == "movement":
        train_files, val_files = train_test_split(
            valid_files, test_size=0.15, random_state=SEED
        )
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        train_idx, val_idx = next(sss.split(valid_files, file_labels))
        train_files, val_files = valid_files[train_idx], valid_files[val_idx]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    print("\nLoading training blocks ...")
    X_train_list, y_train_list = [], []
    for f in train_files:
        result = load_file_blocks(f, args.task)
        if result is not None:
            blocks, labels = result
            X_train_list.append(blocks)
            y_train_list.append(labels)
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    del X_train_list, y_train_list

    print(f"Train blocks BEFORE any oversampling: {X_train.shape[0]} "
          f"(class 0={np.sum(y_train==0)}, class 1={np.sum(y_train==1)})")

    if args.task == "presence":
        X_train, y_train = oversample_minority(X_train, y_train)
        print(f"Train blocks AFTER oversampling: {X_train.shape[0]} "
              f"(class 0={np.sum(y_train==0)}, class 1={np.sum(y_train==1)})")

    print("\nLoading validation blocks (never oversampled, kept clean) ...")
    X_val_list, y_val_list = [], []
    for f in val_files:
        result = load_file_blocks(f, args.task)
        if result is not None:
            blocks, labels = result
            X_val_list.append(blocks)
            y_val_list.append(labels)
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)
    del X_val_list, y_val_list

    print(f"Val blocks: {X_val.shape[0]}")

    print(f"\nLoading and warm-starting from {args.base_model} ...")
    model = build_warmstart_model(args.base_model, args.task)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(f"best_model_{args.task}_FINAL.keras", monitor="val_loss", save_best_only=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating final model on validation set (for reference; k-fold already gave the honest generalization estimate) ...")
    threshold = THRESHOLDS[args.task]
    probs = model.predict(X_val, batch_size=BATCH_SIZE).flatten()
    y_pred = (probs > threshold).astype(np.int64)

    print(f"Using threshold = {threshold}")
    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred, target_names=LABEL_NAMES[args.task], zero_division=0))
    print(f"Sample confidence scores (first 10): {np.round(probs[:10], 3)}")

    model.save(f"final_model_{args.task}_FINAL.keras")
    print(f"\nSaved final production model to final_model_{args.task}_FINAL.keras")
    print(f"(Best checkpoint also saved to best_model_{args.task}_FINAL.keras)")


if __name__ == "__main__":
    main()