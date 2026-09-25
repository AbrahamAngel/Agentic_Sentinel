"""
predict_csi_unified.py

The actual usable output of the CSI sensing pillar: loads all three
final production models (presence, movement, machine on/off) once,
and provides a single function that takes a raw CSI recording and
returns all three confidence scores together -- exactly the structured
input your risk classifier and eventually the agentic layer will need.

Usage (command line):
    python predict_csi_unified.py "data/mc3/some_file.mat"

Usage (as a module):
    from predict_csi_unified import CSISensor
    sensor = CSISensor()
    result = sensor.predict_from_file("some_file.mat")
"""

import sys
import numpy as np
import scipy.io
from scipy.signal import savgol_filter
from tensorflow import keras


BLOCK_SIZE = 25

MODEL_PATHS = {
    "presence": "final_model_presence_FINAL.keras",
    "movement": "final_model_movement_FINAL.keras",
    "machine_onoff": "final_model_machine_onoff_FINAL.keras",
}

# Confirmed decision thresholds per task -- machine on/off uses the
# tuned 0.25 threshold, the others use the standard 0.5
THRESHOLDS = {
    "presence": 0.5,
    "movement": 0.5,
    "machine_onoff": 0.25,
}

LABELS = {
    "presence": ["no_person", "present"],
    "movement": ["stationary", "moving"],
    "machine_onoff": ["off", "on"],
}


def correct_phase_per_reading(phase):
    num_readings, num_subcarriers = phase.shape
    sc_idx = np.arange(num_subcarriers)
    corrected = np.zeros_like(phase)
    for i in range(num_readings):
        row_unwrapped = np.unwrap(phase[i, :])
        coeffs = np.polyfit(sc_idx, row_unwrapped, deg=1)
        trend = np.polyval(coeffs, sc_idx)
        corrected[i, :] = row_unwrapped - trend
    return corrected


def smooth_savgol_over_time(phase_corrected, window_length=11, polyorder=3):
    if window_length >= phase_corrected.shape[0]:
        window_length = phase_corrected.shape[0] - 1
        if window_length % 2 == 0:
            window_length -= 1
    return savgol_filter(phase_corrected, window_length=window_length, polyorder=polyorder, axis=0)


def correct_discontinuities(phase_smoothed, threshold=None):
    corrected = phase_smoothed.copy()
    diffs = np.diff(corrected, axis=0)
    if threshold is None:
        threshold = 3 * np.std(diffs)
    for sc in range(corrected.shape[1]):
        jump_indices = np.where(np.abs(diffs[:, sc]) > threshold)[0]
        for idx in jump_indices:
            corrected[idx + 1, sc] = corrected[idx, sc]
    return corrected


def preprocess(csi):
    """Same cleaning pipeline used throughout training: unwrap -> detrend -> smooth -> correct."""
    amplitude = np.abs(csi)
    phase_raw = np.angle(csi)
    phase_corrected = correct_phase_per_reading(phase_raw)
    phase_smoothed = smooth_savgol_over_time(phase_corrected)
    phase_final = correct_discontinuities(phase_smoothed)
    return amplitude, phase_final


def split_into_blocks(amplitude, phase, block_size=BLOCK_SIZE):
    num_readings, num_subcarriers = amplitude.shape
    num_blocks = num_readings // block_size
    blocks = np.zeros((num_blocks, num_subcarriers, block_size, 2), dtype=np.float32)
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        blocks[b, :, :, 0] = amplitude[start:end, :].T
        blocks[b, :, :, 1] = phase[start:end, :].T
    return blocks


class CSISensor:
    """Loads all three final CSI models once; call predict_from_file() or
    predict_from_array() as often as needed without reloading."""

    def __init__(self, model_paths=None):
        model_paths = model_paths or MODEL_PATHS
        print("Loading CSI models ...")
        self.models = {}
        for task, path in model_paths.items():
            print(f"  {task}: {path}")
            self.models[task] = keras.models.load_model(path)
        print("All CSI models loaded.\n")

    def predict_from_array(self, csi):
        """
        csi: raw complex-valued CSI matrix, shape (num_readings, 64)

        Returns a dict with a confidence score and thresholded label
        for each of the three tasks, using majority vote across all
        blocks in this recording.
        """
        if csi.shape[1] != 64:
            raise ValueError(
                f"These models expect 64-subcarrier (20 MHz) data, got {csi.shape[1]} subcarriers."
            )

        amplitude, phase = preprocess(csi)
        blocks = split_into_blocks(amplitude, phase)

        if blocks.shape[0] == 0:
            raise ValueError("Recording too short to produce even one block.")

        result = {}
        for task, model in self.models.items():
            probs = model.predict(blocks, verbose=0).flatten()
            threshold = THRESHOLDS[task]
            preds = (probs > threshold).astype(int)

            # Majority vote across blocks for the final label, mean
            # confidence for the reported score
            final_label_idx = int(np.round(preds.mean()))
            confidence = float(probs.mean())

            result[task] = {
                "label": LABELS[task][final_label_idx],
                "confidence": round(confidence, 4),
                "threshold_used": threshold,
                "num_blocks_analyzed": int(blocks.shape[0]),
            }

        return result

    def predict_from_file(self, filepath):
        data = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        csi = data["CSI"]
        return self.predict_from_array(csi)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python predict_csi_unified.py "path/to/file.mat"')
        sys.exit(1)

    sensor = CSISensor()
    result = sensor.predict_from_file(sys.argv[1])

    print("=== CSI Sensor Output ===")
    for task, info in result.items():
        print(f"{task:15s}: {info['label']:12s} (confidence={info['confidence']}, "
              f"threshold={info['threshold_used']}, blocks={info['num_blocks_analyzed']})")