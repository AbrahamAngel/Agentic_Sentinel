"""
train_cnn_80mhz.py

Trains the multi-task CNN on the 80 MHz (256-subcarrier) dataset,
following the exact same architecture and methodology as the confirmed
20 MHz result (best_model_20mhz.keras), so the two are genuinely
comparable.

Given what we learned from the 20 MHz reproduction attempt (meaningful
run-to-run variance from random weight initialization), treat this
run's result as one honest data point -- not necessarily a perfectly
reproducible number on a second attempt.

Usage:
    python train_cnn_80mhz.py --data "data/training_dataset_80mhz.h5" --epochs 20
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
import matplotlib.pyplot as plt


BATCH_SIZE = 64
CHUNK_SIZE = 4096
SEED = 42  # fixed seed for reproducible model weight initialization and shuffling


def set_all_seeds(seed=SEED):
    """
    Fixes randomness across Python, NumPy, and TensorFlow so that model
    weight initialization and data shuffling are reproducible run to run.

    This was NOT done in earlier training scripts, which is the most
    likely explanation for why the 20 MHz model could not be reproduced
    on a rerun (movement accuracy varied from 82% to 62% using identical
    code and data, differing only in random initialization).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_generator(h5_path, split_name):
    def gen():
        with h5py.File(h5_path, "r") as hf:
            X = hf[split_name]["X"]
            y_people = hf[split_name]["y_people"]
            y_movement = hf[split_name]["y_movement"]
            y_machine = hf[split_name]["y_machine"]
            num_samples = X.shape[0]

            while True:
                chunk_starts = list(range(0, num_samples, CHUNK_SIZE))
                np.random.shuffle(chunk_starts)

                for chunk_start in chunk_starts:
                    chunk_end = min(chunk_start + CHUNK_SIZE, num_samples)

                    X_chunk = X[chunk_start:chunk_end]
                    people_chunk = y_people[chunk_start:chunk_end]
                    movement_chunk = y_movement[chunk_start:chunk_end]
                    machine_chunk = y_machine[chunk_start:chunk_end]

                    order = np.arange(len(X_chunk))
                    np.random.shuffle(order)

                    for start in range(0, len(order), BATCH_SIZE):
                        batch_order = order[start:start + BATCH_SIZE]
                        if len(batch_order) == 0:
                            continue
                        yield (
                            X_chunk[batch_order],
                            {
                                "people_output": people_chunk[batch_order],
                                "movement_output": movement_chunk[batch_order],
                                "machine_output": machine_chunk[batch_order],
                            },
                        )
    return gen


def get_split_size(h5_path, split_name):
    with h5py.File(h5_path, "r") as hf:
        return hf[split_name]["X"].shape[0], hf[split_name]["X"].shape[1:]


def build_dataset(h5_path, split_name, num_subcarriers, block_size):
    output_signature = (
        tf.TensorSpec(shape=(None, num_subcarriers, block_size, 2), dtype=tf.float32),
        {
            "people_output": tf.TensorSpec(shape=(None,), dtype=tf.int64),
            "movement_output": tf.TensorSpec(shape=(None,), dtype=tf.int64),
            "machine_output": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        },
    )
    ds = tf.data.Dataset.from_generator(
        make_generator(h5_path, split_name), output_signature=output_signature
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(input_shape):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (5, 5), padding="same", activation="mish")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="mish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="mish")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="mish")(x)
    x = layers.Dropout(0.3)(x)
    shared = layers.Dense(128, activation="mish")(x)

    people_output = layers.Dense(4, activation="softmax", name="people_output")(shared)
    movement_output = layers.Dense(2, activation="softmax", name="movement_output")(shared)
    machine_output = layers.Dense(3, activation="softmax", name="machine_output")(shared)

    return keras.Model(inputs=inputs, outputs=[people_output, movement_output, machine_output])


def plot_history(history, out_path="training_history_80mhz.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    heads = ["people_output", "movement_output", "machine_output"]
    titles = ["People Count", "Movement", "Machine Status"]

    for ax, head, title in zip(axes, heads, titles):
        acc_key = f"{head}_accuracy"
        val_acc_key = f"val_{head}_accuracy"
        if acc_key in history.history:
            ax.plot(history.history[acc_key], label="train")
            ax.plot(history.history[val_acc_key], label="val")
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved training history plot to {out_path}")


def evaluate_on_test(model, h5_path, num_subcarriers, block_size):
    with h5py.File(h5_path, "r") as hf:
        X_test = hf["test"]["X"][:]
        y_people_true = hf["test"]["y_people"][:]
        y_movement_true = hf["test"]["y_movement"][:]
        y_machine_true = hf["test"]["y_machine"][:]

    preds = model.predict(X_test, batch_size=BATCH_SIZE)
    people_pred = np.argmax(preds[0], axis=1)
    movement_pred = np.argmax(preds[1], axis=1)
    machine_pred = np.argmax(preds[2], axis=1)

    for name, y_true, y_pred, labels in [
        ("People Count", y_people_true, people_pred, ["0", "1", "2", "3"]),
        ("Movement", y_movement_true, movement_pred, ["stationary", "moving"]),
        ("Machine Status", y_machine_true, machine_pred, ["off", "on/idle", "running"]),
    ]:
        print(f"\n=== {name} ===")
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=labels))


def main():
    set_all_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train_size, sample_shape = get_split_size(args.data, "train")
    val_size, _ = get_split_size(args.data, "val")
    num_subcarriers, block_size, _ = sample_shape

    steps_per_epoch = train_size // BATCH_SIZE
    validation_steps = val_size // BATCH_SIZE

    print(f"Train samples: {train_size} ({steps_per_epoch} steps/epoch)")
    print(f"Val samples: {val_size} ({validation_steps} steps/epoch)")
    print(f"Input shape per sample: {sample_shape}")

    train_ds = build_dataset(args.data, "train", num_subcarriers, block_size)
    val_ds = build_dataset(args.data, "val", num_subcarriers, block_size)

    model = build_model(input_shape=sample_shape)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "people_output": "sparse_categorical_crossentropy",
            "movement_output": "sparse_categorical_crossentropy",
            "machine_output": "sparse_categorical_crossentropy",
        },
        metrics={
            "people_output": "accuracy",
            "movement_output": "accuracy",
            "machine_output": "accuracy",
        },
    )

    model.summary()

    print("\nTiming a few batches to estimate epoch duration ...")
    t0 = time.time()
    test_batches = 10
    for i, _ in enumerate(train_ds.take(test_batches)):
        pass
    elapsed = time.time() - t0
    per_step = elapsed / test_batches
    est_epoch_minutes = (per_step * steps_per_epoch) / 60
    print(f"~{per_step:.2f}s/step -> estimated ~{est_epoch_minutes:.1f} minutes per epoch")
    print(f"Estimated total for {args.epochs} epochs: ~{est_epoch_minutes * args.epochs / 60:.1f} hours\n")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best_model_80mhz.keras", monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history)

    print("\nEvaluating on held-out test set ...")
    evaluate_on_test(model, args.data, num_subcarriers, block_size)

    model.save("final_model_80mhz.keras")
    print("\nSaved final model to final_model_80mhz.keras")


if __name__ == "__main__":
    main()