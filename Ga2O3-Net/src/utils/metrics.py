"""
Regression evaluation metrics.
"""

import numpy as np


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, and R² for 1-D arrays.

    Args:
        pred: Predicted values, shape [N].
        target: Ground-truth values, shape [N].

    Returns:
        dict with keys: mae, rmse, r2
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    assert len(pred) == len(target), "pred and target must have the same length"

    mae = float(np.mean(np.abs(pred - target)))
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))

    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    return {"mae": mae, "rmse": rmse, "r2": float(r2)}


def print_metrics_table(results: dict, title: str = "Results"):
    """Pretty-print a nested metrics dict to stdout."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    for key, val in results.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k2, v2 in val.items():
                print(f"    {k2}: {v2:.4f}")
        else:
            print(f"  {key}: {val:.4f}")
    print(f"{'='*50}\n")
