"""
inspect_mat_file.py

Loads a single EHUNAM .mat file and prints out its structure so you can
see exactly what fields exist, their shapes, and sample values before
writing any preprocessing or training code.

Usage:
    python inspect_mat_file.py path/to/some_file.mat
"""

import sys
import numpy as np
import scipy.io


def inspect_mat_file(filepath):
    print(f"Loading: {filepath}\n")

    # squeeze_me and struct_as_record make the output easier to read
    data = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)

    # loadmat always includes some internal MATLAB housekeeping keys
    # that start with "__" -- skip those, they're not your actual data
    real_keys = [k for k in data.keys() if not k.startswith("__")]

    print(f"Top-level fields found ({len(real_keys)}):")
    for key in real_keys:
        value = data[key]
        print(f"\n--- {key} ---")
        print(f"  type: {type(value)}")

        if isinstance(value, np.ndarray):
            print(f"  shape: {value.shape}")
            print(f"  dtype: {value.dtype}")

            # If it's the CSI matrix, it'll likely be complex-valued
            if np.iscomplexobj(value):
                print("  (complex-valued -- likely the CSI matrix itself)")
                print(f"  sample value at [0][0]: {value.flat[0]}")
            else:
                # For small metadata fields, just print the value directly
                if value.size <= 10:
                    print(f"  value: {value}")
                else:
                    print(f"  first few values: {value.flat[:5]}")
        else:
            # Scalars, strings, etc.
            print(f"  value: {value}")

    print("\nDone. Look for a field with a complex dtype and a shape like")
    print("(num_CSI_readings, num_subcarriers) -- that's your main CSI data.")
    print("Other fields (Activity, Application, People, Machine, Status, etc.)")
    print("are the labels/metadata for this specific recording.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_mat_file.py path/to/file.mat")
        sys.exit(1)

    inspect_mat_file(sys.argv[1])