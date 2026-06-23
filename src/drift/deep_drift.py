"""HDF5 data loading for ann-benchmarks datasets (DEEP-1B, etc.)."""

import h5py
import numpy as np


def load_deep_hdf5(path, n_base=None, n_queries=None):
    """Load ann-benchmarks HDF5 dataset. Returns (train, test) as float32 arrays.

    n_base: cap on number of training vectors loaded (None = all).
    n_queries: cap on number of test vectors loaded (None = all).
    Vectors are NOT normalised — use Euclidean distance.
    """
    with h5py.File(path, "r") as f:
        train = np.array(f["train"][:n_base], dtype=np.float32)
        test = np.array(f["test"][:n_queries], dtype=np.float32)
    return train, test
