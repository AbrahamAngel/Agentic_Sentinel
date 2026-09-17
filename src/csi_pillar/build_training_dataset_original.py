"""
build_training_dataset.py (ORIGINAL, pre-normalization version -- reconstructed)

This is the dataset-building script exactly as it was BEFORE amplitude
normalization was added. Use this to regenerate the original,
unnormalized training_dataset.h5 -- the dataset that produced your
best documented result (people 56%, movement 82%, machine 70%).

Splits files by file-level (65/17.5/17.5) to prevent data leakage,
chops each file into 25-reading blocks, and computes the three labels
(people_count, movement, machine_status) exactly as before -- with NO
amplitude normalization applied.

Usage:
    python build_training_dataset.py --source "data/mc3_processed" --out "data/training_dataset_original.h5"

NOTE: output path is deliberately named "training_dataset_original.h5"
(not "training_dataset.h5") so it never risks overwriting or being
overwritten by the normalized version again. Use this exact filename
with train_cnn.py to regenerate your best final model.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import h5py


BLOCK_SIZE = 25
RANDOM_SEED = 42


def status_to_machine_label(status, n_machine):
    if n_machine == 0 or status == "":
        return 0
    if status == "O":
        return 1
    if status == "R":
        return 2
    return 0


def split_into_blocks(amplitude, phase, block_size=BLOCK_SIZE):
    num_readings, num_subcarriers = amplitude.shape
    num_blocks = num_readings // block_size

    blocks = np.zeros((num_blocks, num_subcarriers, block_size, 2), dtype=np.float32)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        blocks[b, :, :, 0] = amplitude[start:end, :].T  # RAW amplitude, no normalization
        blocks[b, :, :, 1] = phase[start:end, :].T
    return blocks


def compute_movement_labels(amplitude, block_size=BLOCK_SIZE):
    num_readings, num_subcarriers = amplitude.shape
    num_blocks = num_readings // block_size

    variances = np.zeros(num_blocks, dtype=np.float32)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        variances[b] = np.var(amplitude[start:end, :], axis=0).mean()

    threshold = np.median(variances)
    return (variances > threshold).astype(np.int64)


def process_one_file(filepath):
    data = np.load(filepath, allow_pickle=True)
    amplitude = data["amplitude"]
    phase = data["phase"]

    if amplitude.shape[1] != 64:
        return None  # not a 20 MHz file

    n_people = min(int(data["n_people"]) if data["n_people"].size > 0 else 0, 3)
    n_machine = int(data["n_machine"]) if data["n_machine"].size > 0 else 0
    status = str(data["status"])

    blocks = split_into_blocks(amplitude, phase)
    movement_labels = compute_movement_labels(amplitude)
    machine_label = status_to_machine_label(status, n_machine)

    num_blocks = blocks.shape[0]
    people_labels = np.full(num_blocks, n_people, dtype=np.int64)
    machine_labels = np.full(num_blocks, machine_label, dtype=np.int64)

    return blocks, people_labels, movement_labels, machine_labels


def create_split_datasets(hf, split_name, num_subcarriers, block_size):
    group = hf.create_group(split_name)
    X_ds = group.create_dataset(
        "X", shape=(0, num_subcarriers, block_size, 2),
        maxshape=(None, num_subcarriers, block_size, 2),
        dtype=np.float32, chunks=(64, num_subcarriers, block_size, 2), compression="gzip",
    )
    y_people_ds = group.create_dataset("y_people", shape=(0,), maxshape=(None,), dtype=np.int64)
    y_movement_ds = group.create_dataset("y_movement", shape=(0,), maxshape=(None,), dtype=np.int64)
    y_machine_ds = group.create_dataset("y_machine", shape=(0,), maxshape=(None,), dtype=np.int64)
    return X_ds, y_people_ds, y_movement_ds, y_machine_ds


def append_to_datasets(datasets, current_len, blocks, people, movement, machine):
    X_ds, y_people_ds, y_movement_ds, y_machine_ds = datasets
    num_new = blocks.shape[0]
    new_len = current_len + num_new

    X_ds.resize(new_len, axis=0)
    X_ds[current_len:new_len] = blocks
    y_people_ds.resize(new_len, axis=0)
    y_people_ds[current_len:new_len] = people
    y_movement_ds.resize(new_len, axis=0)
    y_movement_ds[current_len:new_len] = movement
    y_machine_ds.resize(new_len, axis=0)
    y_machine_ds[current_len:new_len] = machine

    return new_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_root = Path(args.source)
    all_files = sorted(source_root.glob("*_processed.npz"))
    print(f"Found {len(all_files)} processed files in {source_root}")

    random.seed(RANDOM_SEED)
    shuffled = all_files[:]
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * 0.65)
    n_val = int(n_total * 0.175)

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]

    print(f"File-level split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    num_subcarriers = 64
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skipped_wrong_bw = []
    skipped_errors = []

    with h5py.File(out_path, "w") as hf:
        splits = {
            "train": (train_files, create_split_datasets(hf, "train", num_subcarriers, BLOCK_SIZE)),
            "val": (val_files, create_split_datasets(hf, "val", num_subcarriers, BLOCK_SIZE)),
            "test": (test_files, create_split_datasets(hf, "test", num_subcarriers, BLOCK_SIZE)),
        }

        for split_name, (files, datasets) in splits.items():
            current_len = 0
            print(f"\nProcessing {split_name} split ({len(files)} files) ...")

            for i, filepath in enumerate(files, start=1):
                try:
                    result = process_one_file(filepath)
                    if result is None:
                        skipped_wrong_bw.append(filepath.name)
                        continue
                    blocks, people, movement, machine = result
                    current_len = append_to_datasets(datasets, current_len, blocks, people, movement, machine)
                    del blocks
                except Exception as e:
                    skipped_errors.append((filepath.name, str(e)))

                if i % 25 == 0 or i == len(files):
                    print(f"  {split_name}: {i}/{len(files)} files -- {current_len} blocks so far")

            print(f"{split_name} split final size: {current_len} blocks")

        for split_name in ["train", "val", "test"]:
            y_people = hf[split_name]["y_people"][:]
            y_movement = hf[split_name]["y_movement"][:]
            y_machine = hf[split_name]["y_machine"][:]
            print(f"\n{split_name} label distributions:")
            print(f"  people:  {np.bincount(y_people, minlength=4)}")
            print(f"  movement: {np.bincount(y_movement, minlength=2)}")
            print(f"  machine:  {np.bincount(y_machine, minlength=3)}")

    print(f"\nSaved to {out_path}")
    print(f"\nSkipped {len(skipped_wrong_bw)} files (wrong bandwidth/subcarrier count)")
    if skipped_errors:
        print(f"Skipped {len(skipped_errors)} files due to errors:")
        for name, err in skipped_errors[:10]:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()