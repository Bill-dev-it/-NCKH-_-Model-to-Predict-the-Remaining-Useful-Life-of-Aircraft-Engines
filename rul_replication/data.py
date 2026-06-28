from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import COLUMN_NAMES, DEFAULT_DATA_DIR, SELECTED_SENSORS


@dataclass
class Scaler:
    mean: pd.Series
    std: pd.Series

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = df.copy()
        out[columns] = (out[columns] - self.mean) / self.std.replace(0, 1.0)
        return out


@dataclass
class CmapssData:
    train_windows: np.ndarray
    train_labels: np.ndarray
    test_windows: np.ndarray
    test_labels: np.ndarray
    test_units: np.ndarray
    feature_names: list[str]


def read_split(data_dir: Path, fd: str, split: str) -> pd.DataFrame:
    path = data_dir / f"{split}_{fd}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)


def read_rul(data_dir: Path, fd: str) -> np.ndarray:
    path = data_dir / f"RUL_{fd}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=np.float32)


def add_train_rul(df: pd.DataFrame, clip_rul: int | None) -> pd.DataFrame:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    out = df.copy()
    out["rul"] = max_cycle - out["cycle"]
    if clip_rul is not None:
        out["rul"] = out["rul"].clip(upper=clip_rul)
    return out


def fit_scaler(train_df: pd.DataFrame, columns: list[str]) -> Scaler:
    return Scaler(mean=train_df[columns].mean(), std=train_df[columns].std(ddof=0))


def make_train_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for _, unit_df in df.groupby("unit", sort=True):
        features = unit_df[feature_cols].to_numpy(dtype=np.float32)
        labels = unit_df["rul"].to_numpy(dtype=np.float32)
        if len(unit_df) < window_size:
            continue
        for end in range(window_size, len(unit_df) + 1):
            start = end - window_size
            xs.append(features[start:end])
            ys.append(float(labels[end - 1]))
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def make_test_windows(
    df: pd.DataFrame,
    ruls: np.ndarray,
    feature_cols: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    units: list[int] = []
    for idx, (unit, unit_df) in enumerate(df.groupby("unit", sort=True)):
        features = unit_df[feature_cols].to_numpy(dtype=np.float32)
        if len(features) >= window_size:
            window = features[-window_size:]
        else:
            pad = np.repeat(features[:1], window_size - len(features), axis=0)
            window = np.concatenate([pad, features], axis=0)
        xs.append(window)
        ys.append(float(ruls[idx]))
        units.append(int(unit))
    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.asarray(units, dtype=np.int32)


def load_cmapss(
    fd: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    window_size: int = 30,
    clip_rul: int | None = 125,
    include_settings: bool = False,
) -> CmapssData:
    feature_cols = (["setting_1", "setting_2", "setting_3"] if include_settings else []) + SELECTED_SENSORS
    train_df = add_train_rul(read_split(data_dir, fd, "train"), clip_rul)
    test_df = read_split(data_dir, fd, "test")
    scaler = fit_scaler(train_df, feature_cols)
    train_df = scaler.transform(train_df, feature_cols)
    test_df = scaler.transform(test_df, feature_cols)
    train_x, train_y = make_train_windows(train_df, feature_cols, window_size)
    test_x, test_y, test_units = make_test_windows(test_df, read_rul(data_dir, fd), feature_cols, window_size)
    return CmapssData(train_x, train_y, test_x, test_y, test_units, feature_cols)
