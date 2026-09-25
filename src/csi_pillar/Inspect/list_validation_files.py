"""
list_validation_files.py

Reproduces the EXACT same file-level train/validation split that
train_final_model.py used for each task (same fixed seed = 42), and
lists which files ended up in the VALIDATION set for each task --
i.e., files the model's weights were never updated on, safe to use
for a genuine, honest qualitative test.

Also reports which files were held out for ALL THREE tasks at once,
since those are the cleanest choice for a combined demo (guaranteed
untouched by presence, movement, AND machine_onoff training alike).

Usage:
    python list_validation_files.py --source "data/mc3_processed"
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


SEED = 42


def get_file_label(filepath, task):
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
    else:
        return 0


def get_validation_files(valid_files, file_labels, task):
    if task == "movement":
        _, val_files = train_test_split(valid_files, test_size=0.15, random_state=SEED)
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        _, val_idx = next(sss.split(valid_files, file_labels))
        val_files = valid_files[val_idx]
    return set(val_files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    source_root = Path(args.source)
    all_files = sorted(source_root.glob("*_processed.npz"))

    valid_files = []
    print("Scanning files ...")
    for f in all_files:
        data = np.load(f, allow_pickle=True)
        if data["amplitude"].shape[1] == 64:
            valid_files.append(f)
    valid_files = np.array(valid_files)
    print(f"Found {len(valid_files)} valid (20 MHz) files\n")

    val_sets = {}
    for task in ["presence", "movement", "machine_onoff"]:
        file_labels = np.array([get_file_label(f, task) for f in valid_files])
        val_files = get_validation_files(valid_files, file_labels, task)
        val_sets[task] = val_files
        print(f"{task}: {len(val_files)} files held out for validation (never trained on)")

    common = val_sets["presence"] & val_sets["movement"] & val_sets["machine_onoff"]

    print(f"\n{'=' * 60}")
    print(f"Files held out for ALL THREE tasks (safest for a combined demo): {len(common)}")
    print(f"{'=' * 60}")
    for f in sorted(common):
        print(f"  {f.name}")

    if len(common) == 0:
        print("\n(No file was held out for all three simultaneously -- this can")
        print(" happen since each task's split is stratified differently.")
        print(" Use a per-task validation file instead, printed below.)")

        for task in ["presence", "movement", "machine_onoff"]:
            print(f"\n--- {task} validation files (first 5) ---")
            for f in sorted(val_sets[task])[:5]:
                print(f"  {f.name}")


if __name__ == "__main__":
    main()