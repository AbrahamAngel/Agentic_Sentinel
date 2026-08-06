"""
train_cnn.py

Builds and trains a CNN on the file-level-split HDF5 dataset, following
EHUNAM's own described architecture (3 conv layers, maxpooling, dropout,
2 dense layers, Mish activation), but extended with THREE output heads
instead of one, since we're predicting three separate labels at once:

  - people_count  (4 classes: 0, 1, 2, 3)
  - movement      (2 classes: stationary, moving)
  - machine_status (3 classes: off, on/idle, running)

Reads data in batches directly from the HDF5 file (via a generator),
rather than loading the whole dataset into RAM, since we already learned
the hard way that this dataset is too large to hold in memory at once.

Usage:
    python train_cnn.py --data "data/training_dataset.h5" --epochs 20
"""

import argparse

import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


BATCH_SIZE = 64


def make_generator(h5_path, split_name):
    """
    A generator function that yields one batch at a time, reading
    directly from the HDF5 file. This keeps memory usage bounded to
    one batch, regardless of how large the full dataset is.
    """
    def gen():
        with h5py.File(h5_path, "r") as hf:
            X = hf[split_name]["X"]
            y_people = hf[split_name]["y_people"]
            y_movement = hf[split_name]["y_movement"]
            y_machine = hf[split_name]["y_machine"]

            num_samples = X.shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start in range(0, num_samples, BATCH_SIZE):
                batch_idx = sorted(indices[start:start + BATCH_SIZE].tolist())
                X_batch = X[batch_idx]
                yield (
                    X_batch,
                    {
                        "people_output": y_people[batch_idx],
                        "movement_output": y_movement[batch_idx],
                        "machine_output": y_machine[batch_idx],
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
    """
    Shared convolutional backbone (following EHUNAM's described
    architecture), branching into three separate output heads.
    """
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

    model = keras.Model(inputs=inputs, outputs=[people_output, movement_output, machine_output])
    return model


def plot_history(history, out_path="training_history.png"):
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
    """Runs the trained model on the full test set and prints a
    confusion matrix + classification report for each output head."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training_dataset.h5")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    train_size, sample_shape = get_split_size(args.data, "train")
    val_size, _ = get_split_size(args.data, "val")
    num_subcarriers, block_size, _ = sample_shape

    print(f"Train samples: {train_size}, Val samples: {val_size}")
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

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history)

    print("\nEvaluating on held-out test set ...")
    evaluate_on_test(model, args.data, num_subcarriers, block_size)

    model.save("final_model.keras")
    print("\nSaved final model to final_model.keras")


if __name__ == "__main__":
    main()