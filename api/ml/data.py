from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS, TARGET_COLUMNS


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


def load_feature_frame(climate_csv: Path, weather_csv: Path) -> pd.DataFrame:
    climate = pd.read_csv(climate_csv, low_memory=False)
    weather = pd.read_csv(weather_csv, low_memory=False)

    required_climate = {"%time", *CLIMATE_COLUMNS}
    required_weather = {"%time", *WEATHER_COLUMNS}
    missing_climate = sorted(required_climate.difference(climate.columns))
    missing_weather = sorted(required_weather.difference(weather.columns))
    if missing_climate:
        raise ValueError(f"Climate CSV is missing columns: {missing_climate}")
    if missing_weather:
        raise ValueError(f"Weather CSV is missing columns: {missing_weather}")

    climate = climate[["%time", *CLIMATE_COLUMNS]].copy()
    weather = weather[["%time", *WEATHER_COLUMNS]].copy()

    merged = climate.merge(weather, on="%time", how="inner")
    merged = merged.sort_values("%time").drop_duplicates(subset=["%time"], keep="last")

    frame = merged[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    all_nan_cols = [col for col in FEATURE_COLUMNS if frame[col].isna().all()]
    if all_nan_cols:
        raise ValueError(f"Columns are entirely non-numeric/missing after coercion: {all_nan_cols}")

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.interpolate(method="linear", limit_direction="both", axis=0)
    frame = frame.ffill().bfill()
    frame = frame.fillna(frame.median(numeric_only=True))
    frame = frame.astype(np.float32)
    if frame.empty:
        raise ValueError("No usable rows after cleaning/interpolation")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Cleaned feature frame still contains non-finite values")

    return frame


def _check_finite(arr: np.ndarray, name: str) -> None:
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values detected in {name}")


def _make_sequences_for_target_range(
    data: np.ndarray,
    lookback: int,
    horizon: int,
    target_col_indices: list[int],
    target_row_start: int,
    target_row_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) pairs where the prediction target falls strictly within
    [target_row_start, target_row_end) of the raw data array.

    The input window for a given target_row is:
        X = data[target_row - lookback - horizon + 1 : target_row - horizon + 1]
        y = data[target_row, target_col_indices]

    Val and test windows are allowed to look back into the training rows —
    that is fine because the targets are always future to training.
    """
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for target_row in range(target_row_start, target_row_end):
        x_start = target_row - lookback - horizon + 1
        if x_start < 0:
            continue
        xs.append(data[x_start : x_start + lookback])
        ys.append(data[target_row, target_col_indices])
    if not xs:
        raise ValueError(
            f"No sequences could be built for target range [{target_row_start}, {target_row_end}). "
            f"Need at least {lookback + horizon} rows before the first target."
        )
    return np.stack(xs, dtype=np.float32), np.stack(ys, dtype=np.float32)


def split_scale_frame(
    frame: pd.DataFrame,
    lookback: int,
    horizon: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> DatasetSplits:
    """
    Correct chronological pipeline:

    1. Split raw rows at the frame level (no sequence overlap between partitions).
    2. Fit feature_scaler on training rows ONLY — no future data leaks into scaling.
    3. Scale the entire row sequence with the training-fit scaler.
    4. Build sequences for each partition independently.  Val/test windows may
       include training-period context rows but their targets are strictly in the
       future of training.
    5. target_scaler is aligned to feature_scaler for the target columns so that
       inverse_transform is consistent between training and inference.
    """
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be in (0, 1) and sum to < 1")

    data_raw = frame.to_numpy(dtype=np.float32)
    n_rows = len(data_raw)
    target_col_indices = [FEATURE_COLUMNS.index(col) for col in TARGET_COLUMNS]

    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)

    if train_end < lookback + horizon:
        raise ValueError(
            f"Training portion ({train_end} rows) is too small for lookback={lookback}, horizon={horizon}."
        )
    if val_end >= n_rows:
        raise ValueError("val_ratio is too large; no rows remain for the test split.")

    # --- Fit scaler on training rows only ---
    feature_scaler = StandardScaler()
    feature_scaler.fit(data_raw[:train_end])

    # --- Scale the full timeline with the training-fit scaler ---
    data_scaled = feature_scaler.transform(data_raw).astype(np.float32)
    _check_finite(data_scaled, "scaled feature matrix")

    # --- Build sequences per partition ---
    x_train, y_train = _make_sequences_for_target_range(
        data_scaled, lookback, horizon, target_col_indices,
        target_row_start=lookback + horizon - 1,
        target_row_end=train_end,
    )
    x_val, y_val = _make_sequences_for_target_range(
        data_scaled, lookback, horizon, target_col_indices,
        target_row_start=train_end,
        target_row_end=val_end,
    )
    x_test, y_test = _make_sequences_for_target_range(
        data_scaled, lookback, horizon, target_col_indices,
        target_row_start=val_end,
        target_row_end=n_rows,
    )

    _check_finite(x_train, "x_train")
    _check_finite(y_train, "y_train")
    _check_finite(x_val, "x_val")
    _check_finite(y_val, "y_val")
    _check_finite(x_test, "x_test")
    _check_finite(y_test, "y_test")

    # --- Build target_scaler aligned to feature_scaler for the target columns ---
    # Targets are embedded inside x windows using feature_scaler normalization.
    # target_scaler must use the same mean/scale for those columns so that
    # inverse_transform at eval and inference time produces consistent raw-unit values.
    target_scaler = copy.deepcopy(feature_scaler)
    target_scaler.mean_ = feature_scaler.mean_[target_col_indices].copy()
    target_scaler.scale_ = feature_scaler.scale_[target_col_indices].copy()
    target_scaler.var_ = feature_scaler.var_[target_col_indices].copy()
    target_scaler.n_features_in_ = len(target_col_indices)

    return DatasetSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )


# ---------------------------------------------------------------------------
# Legacy helpers kept for backward compatibility with evaluate_mpc_scenarios.py
# and any external callers. New code should use split_scale_frame directly.
# ---------------------------------------------------------------------------

def create_sequences(frame: pd.DataFrame, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Legacy: create all sequences from the full frame without splitting."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    data = frame.to_numpy(dtype=np.float32)
    target_col_indices = [FEATURE_COLUMNS.index(col) for col in TARGET_COLUMNS]
    return _make_sequences_for_target_range(
        data, lookback, horizon, target_col_indices,
        target_row_start=lookback + horizon - 1,
        target_row_end=len(data),
    )


def split_scale_sequences(
    x_values: np.ndarray,
    y_values: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> DatasetSplits:
    """
    Legacy sequence-level split.  Kept for backward compatibility only.
    Prefer split_scale_frame, which fixes scaler leakage and split contamination.
    """
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be in (0, 1) and sum to < 1")

    total = len(x_values)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    x_train = x_values[:train_end]
    x_val = x_values[train_end:val_end]
    x_test = x_values[val_end:]
    y_train = y_values[:train_end]
    y_val = y_values[train_end:val_end]
    y_test = y_values[val_end:]

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Split produced an empty partition; adjust ratios or dataset size")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    x_train_s = feature_scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_val_s = feature_scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
    x_test_s = feature_scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)
    y_train_s = target_scaler.fit_transform(y_train)
    y_val_s = target_scaler.transform(y_val)
    y_test_s = target_scaler.transform(y_test)

    for name, arr in [
        ("x_train", x_train_s), ("x_val", x_val_s), ("x_test", x_test_s),
        ("y_train", y_train_s), ("y_val", y_val_s), ("y_test", y_test_s),
    ]:
        _check_finite(arr, name)

    return DatasetSplits(
        x_train=x_train_s, y_train=y_train_s,
        x_val=x_val_s, y_val=y_val_s,
        x_test=x_test_s, y_test=y_test_s,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )
