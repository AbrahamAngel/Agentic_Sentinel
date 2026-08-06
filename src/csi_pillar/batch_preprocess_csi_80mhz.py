"""
batch_preprocess_csi_80mhz.py

Same as batch_preprocess_csi.py, but specifically for the 256-subcarrier
(80 MHz bandwidth) EHUNAM files that were skipped earlier. Following
EHUNAM's own methodology, these are processed and later trained as a
SEPARATE model from the 64-subcarrier (20 MHz) data -- not merged into
one model, since the input shapes are genuinely different.

Usage:
    python batch_preprocess_csi_80mhz.py --source "data/mc3" --dest "data/mc3_processed_80mhz"
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import savgol_filter


def load_csi(filepath):
    data = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    csi = data["CSI"]
    labels = {
        "Application": str(data.get("Application", "")),
        "People": str(data.get("People", "")),
        "Machine": str(data.get("Machine", "")),
        "Status": str(data.get("Status", "")),
        "N_People": data.get("N_People", 0),
        "N_Machine": data.get("N_Machine", 0),
    }
    return csi, labels


def extract_amplitude_phase(csi):
    amplitude = np.abs(csi)
    phase = np.angle(csi)
    return amplitude, phase


def correct_phase_per_reading(phase):
    """Unwrap + detrend across subcarriers, per individual reading."""
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


def process_one_file(filepath):
    csi, labels = load_csi(filepath)
    amplitude, phase_raw = extract_amplitude_phase(csi)
    phase_corrected = correct_phase_per_reading(phase_raw)
    phase_smoothed = smooth_savgol_over_time(phase_corrected)
    phase_final = correct_discontinuities(phase_smoothed)
    return amplitude, phase_final, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder containing raw .mat files")
    parser.add_argument("--dest", required=True, help="Folder to save processed .npz files into")
    args = parser.parse_args()

    source_root = Path(args.source)
    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(source_root.glob("*.mat"))
    print(f"Found {len(mat_files)} .mat files in {source_root}")

    succeeded = 0
    skipped_wrong_bw = 0
    failed = []

    for i, filepath in enumerate(mat_files, start=1):
        try:
            data_check = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
            if data_check["CSI"].shape[1] != 256:
                skipped_wrong_bw += 1
                continue  # this file is 20 MHz, not what this script handles

            amplitude, phase, labels = process_one_file(filepath)

            out_path = dest_root / (filepath.stem + "_processed.npz")
            np.savez(
                out_path,
                amplitude=amplitude,
                phase=phase,
                application=labels["Application"],
                people=labels["People"],
                machine=labels["Machine"],
                status=labels["Status"],
                n_people=labels["N_People"],
                n_machine=labels["N_Machine"],
            )
            succeeded += 1

        except Exception as e:
            failed.append((filepath.name, str(e)))

        if i % 25 == 0 or i == len(mat_files):
            print(f"  checked {i}/{len(mat_files)} -- {succeeded} processed (80MHz), {skipped_wrong_bw} skipped (20MHz)")

    print(f"\nDone. {succeeded} 80 MHz files processed successfully, saved to {dest_root}")
    print(f"{skipped_wrong_bw} 20 MHz files skipped (already processed separately earlier)")

    if failed:
        print(f"\n{len(failed)} files failed to process. First few:")
        for name, err in failed[:10]:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()