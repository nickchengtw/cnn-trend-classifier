import numpy as np


def normalize_windows(X):
    # Normalize per sample
    X_min = X.min(axis=(1, 2), keepdims=True)
    X_max = X.max(axis=(1, 2), keepdims=True)
    X = (X - X_min) / (X_max - X_min + 1e-8)

    # If needed, reshape to (num_samples, 15, 1, 1) for Conv2D
    X = X[..., np.newaxis]  # shape: (num_samples, 15, 1, 1)
    X = X.squeeze(-1)  # shape (num_samples, 15, 1)

    return X