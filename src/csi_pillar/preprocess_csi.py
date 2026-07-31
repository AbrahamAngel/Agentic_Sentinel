"""
preprocess_csi.py (corrected version)

Implements TSFR-style preprocessing on raw EHUNAM CSI data:
1. Unwrap and remove linear phase drift ACROSS SUBCARRIERS, per individual
   reading (frequency axis) -- this corrects the per-packet timing offset
   (STO/SFO-type error), and is safe because phase varies smoothly across
   subcarriers within a single packet.
2. Savitzky-Golay filtering ACROSS TIME (per subcarrier, across readings)
   to smooth out noise between consecutive readings.
3. Threshold-based correction for any remaining large jumps over time.

NOTE: an earlier version of this script incorrectly unwrapped phase across
TIME instead of across subcarriers. Raw phase between independent packets
is not a continuous signal, so time-axis unwrapping caused runaway values
(the "processed" plot showing phase in the hundreds of radians instead of
a normal, bounded range). This version fixes that by unwrapping/detrending
per-reading across subcarriers instead, which is where phase is actually
physically continuous.

Usage:
    python preprocess_csi.py "data/mc3/MC3_01A_3_PCMAR_j_#_8_R_01.mat"
"""

import sys
import numpy as np
import scipy.io
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def load_csi(filepath):
    """Load the complex CSI matrix from an EHUNAM .mat file."""
    data = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    csi = data["CSI"]  # shape: (num_readings, num_subcarriers), complex128
    return csi


def extract_amplitude_phase(csi):
    """Split the complex CSI matrix into separate amplitude and phase arrays."""
    amplitude = np.abs(csi)
    phase = np.angle(csi)  # radians, wrapped into [-pi, pi]
    return amplitude, phase


def correct_phase_per_reading(phase):
    """
    For each individual CSI reading (row), unwrap phase ACROSS SUBCARRIERS
    (not across time), then fit and subtract a linear trend across
    subcarriers. This is the correct axis for this kind of correction,
    since phase varies smoothly with subcarrier frequency within a single
    packet, but is not guaranteed continuous from one packet to the next.
    """
    num_readings, num_subcarriers = phase.shape
    sc_idx = np.arange(num_subcarriers)
    corrected = np.zeros_like(phase)

    for i in range(num_readings):
        row = phase[i, :]

        # Unwrap along the subcarrier axis for this single reading
        row_unwrapped = np.unwrap(row)

        # Fit and remove a linear trend across subcarriers
        coeffs = np.polyfit(sc_idx, row_unwrapped, deg=1)
        trend = np.polyval(coeffs, sc_idx)
        corrected[i, :] = row_unwrapped - trend

    return corrected


def smooth_savgol_over_time(phase_corrected, window_length=11, polyorder=3):
    """
    Apply a Savitzky-Golay filter along the TIME axis (per subcarrier,
    across readings) to smooth out remaining noise between consecutive
    readings. This is safe now that each reading's phase has already
    been sanity-corrected across subcarriers.
    """
    if window_length >= phase_corrected.shape[0]:
        window_length = phase_corrected.shape[0] - 1
        if window_length % 2 == 0:
            window_length -= 1

    smoothed = savgol_filter(
        phase_corrected, window_length=window_length, polyorder=polyorder, axis=0
    )
    return smoothed


def correct_discontinuities(phase_smoothed, threshold=None):
    """
    Detect and correct any remaining large jumps between consecutive
    time samples (per subcarrier) that exceed a threshold.
    """
    corrected = phase_smoothed.copy()
    diffs = np.diff(corrected, axis=0)

    if threshold is None:
        threshold = 3 * np.std(diffs)

    for sc in range(corrected.shape[1]):
        col_diffs = diffs[:, sc]
        jump_indices = np.where(np.abs(col_diffs) > threshold)[0]
        for idx in jump_indices:
            corrected[idx + 1, sc] = corrected[idx, sc]

    return corrected


def plot_comparison(phase_raw, phase_processed, num_readings_to_show=10):
    """Before/after comparison plot, same style as EHUNAM's Fig. 8."""
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    num_subcarriers = phase_raw.shape[1]
    x = np.arange(num_subcarriers)
    y = np.arange(num_readings_to_show)
    X, Y = np.meshgrid(x, y)

    ax1.plot_surface(X, Y, phase_raw[:num_readings_to_show, :], cmap="viridis")
    ax1.set_title("Raw Phase")
    ax1.set_xlabel("Subcarrier")
    ax1.set_ylabel("CSI reading")
    ax1.set_zlabel("Phase (rad)")

    ax2.plot_surface(X, Y, phase_processed[:num_readings_to_show, :], cmap="viridis")
    ax2.set_title("Processed Phase (TSFR, corrected)")
    ax2.set_xlabel("Subcarrier")
    ax2.set_ylabel("CSI reading")
    ax2.set_zlabel("Phase (rad)")

    plt.tight_layout()
    plt.savefig("csi_phase_comparison.png", dpi=150)
    print("\nSaved comparison plot to csi_phase_comparison.png")


def main(filepath):
    print(f"Loading {filepath} ...")
    csi = load_csi(filepath)
    print(f"CSI shape: {csi.shape}")

    amplitude, phase_raw = extract_amplitude_phase(csi)

    print("Correcting phase per-reading across subcarriers (unwrap + detrend) ...")
    phase_corrected = correct_phase_per_reading(phase_raw)

    print("Applying Savitzky-Golay smoothing across time ...")
    phase_smoothed = smooth_savgol_over_time(phase_corrected)

    print("Correcting residual discontinuities over time ...")
    phase_final = correct_discontinuities(phase_smoothed)

    print(f"\nRaw phase range:       [{phase_raw.min():.2f}, {phase_raw.max():.2f}] rad")
    print(f"Processed phase range: [{phase_final.min():.2f}, {phase_final.max():.2f}] rad")
    print("(Processed range should now be a reasonable, bounded range --")
    print(" not hundreds of radians. If it still looks huge, something is")
    print(" still wrong and we should investigate further before trusting it.)")

    print("\nGenerating before/after comparison plot ...")
    plot_comparison(phase_raw, phase_final)

    out_path = filepath.replace(".mat", "_processed.npz")
    np.savez(out_path, amplitude=amplitude, phase=phase_final)
    print(f"Saved processed amplitude + phase to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python preprocess_csi.py "path/to/file.mat"')
        sys.exit(1)

    main(sys.argv[1])