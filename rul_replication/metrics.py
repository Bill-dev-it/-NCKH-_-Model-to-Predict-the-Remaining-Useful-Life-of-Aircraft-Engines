from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_pred - y_true
    score = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(score))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def rrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true = float(np.mean(y_true))
    return rmse(y_true, y_pred) / mean_true * 100.0 if mean_true else float("nan")


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    std_ratio = float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) else float("nan")
    mean_ratio = float(np.mean(y_pred) / np.mean(y_true)) if np.mean(y_true) else float("nan")
    return float(1.0 - np.sqrt((corr - 1.0) ** 2 + (std_ratio - 1.0) ** 2 + (mean_ratio - 1.0) ** 2))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "r2": r2_score_np(y_true, y_pred),
        "rrmse": rrmse(y_true, y_pred),
        "kge": kge(y_true, y_pred),
    }
