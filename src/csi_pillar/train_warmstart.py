"""
train_warmstart.py

Trains a single-task binary classifier (presence, movement, or
machine on/off) by warm-starting from your existing multi-task
model's shared backbone weights, rather than random initialization.

This implements the new architecture direction: three independent
models, each producing a confidence score, instead of one shared
multi-task model. Warm-starting reuses everything the multi-task
model already learned about reading CSI patterns in general, then
fine-tunes a fresh, single output head for the specific binary task.

Label derivation:
  - presence:       1 if y_people > 0, else 0   (derived from y_people)
  - movement:        y_movement as-is           (already binary)
  - machine_onoff:   1 if y_machine in {1, 2} (on-idle OR running),
                      0 if y_machine == 0 (off) -- this collapses the
                      old 3-class problem (which had a severe minority
                      class) into a much more balanced binary problem

Output: a single confidence score (0.0-1.0) via sigmoid activation,
not a hard class label -- exactly what the risk-classifier stage
downstream needs to combine with the YOLO confidence scores.

Usage:
    python train_warmstart.py --task presence --data "data/training_dataset_original.h5" --base_model "best_model_20mhz.keras" --epochs 15
    python train_warmstart.py --task movement --data "data/training_dataset_original.h5" --base_model "best_model_20mhz.keras" --epochs 15
    python train_warmstart.py --task machine_onoff --data "data/training_dataset_original.h5" --base_model "best_model_20mhz.keras" --epochs 15
"""

import argparse
import random
import time

import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report


BATCH_SIZE = 64
CHUNK_SIZE = 4096
SEED = 42

TASK_CONFIG = {
    "presence": {"source_label": "y_people", "derive": lambda y: (y > 0).astype(np.int64)},
    "movement": {"source_label": "y_movement", "derive": lambda y: y},
    "machine_onoff": {"source_label": "y_machine", "derive": lambda y: np.isin(y, [1, 2]).astype(np.int64)},
}


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_generator(h5_path, split_name, task):
    source_label = TASK_CONFIG[task]["source_label"]
    derive_fn = TASK_CONFIG[task]["derive"]

    def gen():
        with h5py.File(h5_path, "r") as hf:
            X = hf[split_name]["X"]
            y_source = hf[split_name][source_label]
            num_samples = X.shape[0]

            while True:
                chunk_starts = list(range(0, num_samples, CHUNK_SIZE))
                np.random.shuffle(chunk_starts)

                for chunk_start in chunk_starts:
                    chunk_end = min(chunk_start + CHUNK_SIZE, num_samples)

                    X_chunk = X[chunk_start:chunk_end]
                    y_chunk_raw = y_source[chunk_start:chunk_end]
                    y_chunk = derive_fn(y_chunk_raw)

                    order = np.arange(len(X_chunk))
                    np.random.shuffle(order)

                    for start in range(0, len(order), BATCH_SIZE):
                        batch_order = order[start:start + BATCH_SIZE]
                        if len(batch_order) == 0:
                            continue
                        yield (X_chunk[batch_order], y_chunk[batch_order].astype(np.float32))
    return gen


def get_split_size(h5_path, split_name):
    with h5py.File(h5_path, "r") as hf:
        return hf[split_name]["X"].shape[0], hf[split_name]["X"].shape[1:]


def build_dataset(h5_path, split_name, num_subcarriers, block_size, task):
    output_signature = (
        tf.TensorSpec(shape=(None, num_subcarriers, block_size, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        make_generator(h5_path, split_name, task), output_signature=output_signature
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def build_warmstart_model(base_model_path, task):
    """
    Loads the existing multi-task model, reuses its shared backbone
    (every layer up through the last shared dense layer, "dense_1"),
    and attaches a fresh single-output sigmoid head for this task.
    All backbone weights are carried over from the multi-task model,
    not randomly initialized -- this is the actual "warm start".
    """
    base_model = keras.models.load_model(base_model_path)
    shared_output = base_model.get_layer("dense_1").output

    new_output = layers.Dense(1, activation="sigmoid", name=f"{task}_output")(shared_output)
    model = keras.Model(inputs=base_model.input, outputs=new_output)
    return model


def evaluate_on_test(model, h5_path, task):
    source_label = TASK_CONFIG[task]["source_label"]
    derive_fn = TASK_CONFIG[task]["derive"]

    with h5py.File(h5_path, "r") as hf:
        X_test = hf["test"]["X"][:]
        y_source = hf["test"][source_label][:]

    y_true = derive_fn(y_source)
    probs = model.predict(X_test, batch_size=BATCH_SIZE).flatten()
    y_pred = (probs > 0.5).astype(np.int64)

    label_names = {
        "presence": ["no person", "person present"],
        "movement": ["stationary", "moving"],
        "machine_onoff": ["off", "on"],
    }[task]

    print(f"\n=== {task} (warm-started, confidence-score model) ===")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=label_names))
    print(f"\nSample confidence scores (first 10 test predictions): {np.round(probs[:10], 3)}")


def main():
    set_all_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--data", required=True)
    parser.add_argument("--base_model", required=True, help="Path to the existing multi-task model to warm-start from")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    train_size, sample_shape = get_split_size(args.data, "train")
    val_size, _ = get_split_size(args.data, "val")
    num_subcarriers, block_size, _ = sample_shape

    steps_per_epoch = train_size // BATCH_SIZE
    validation_steps = val_size // BATCH_SIZE

    print(f"Task: {args.task}")
    print(f"Train samples: {train_size} ({steps_per_epoch} steps/epoch)")
    print(f"Val samples: {val_size} ({validation_steps} steps/epoch)")

    train_ds = build_dataset(args.data, "train", num_subcarriers, block_size, args.task)
    val_ds = build_dataset(args.data, "val", num_subcarriers, block_size, args.task)

    print(f"\nLoading and warm-starting from {args.base_model} ...")
    model = build_warmstart_model(args.base_model, args.task)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # lower LR since backbone is already trained
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    print("\nTiming a few batches to estimate epoch duration ...")
    t0 = time.time()
    for i, _ in enumerate(train_ds.take(10)):
        pass
    elapsed = time.time() - t0
    per_step = elapsed / 10
    print(f"~{per_step:.2f}s/step -> estimated ~{(per_step * steps_per_epoch) / 60:.1f} minutes per epoch\n")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(f"best_model_{args.task}_warmstart.keras", monitor="val_loss", save_best_only=True),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("\nEvaluating on held-out test set ...")
    evaluate_on_test(model, args.data, args.task)

    model.save(f"final_model_{args.task}_warmstart.keras")
    print(f"\nSaved final model to final_model_{args.task}_warmstart.keras")


if __name__ == "__main__":
    main()