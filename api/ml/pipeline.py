from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS


@dataclass
class PipelineOptions:
    lookback: int
    merge_strategy: str = "inner"
    fill_method: str = "interpolate"
    scaling_mode: str = "trained"


@dataclass
class PreparedWindow:
    raw_window: np.ndarray
    scaled_window: np.ndarray
    scaler_used: StandardScaler
    metadata: dict[str, Any]


def _normalize_table_rows(rows: list[dict[str, Any]], source_name: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["%time"])
    frame = pd.DataFrame(rows)
    if "%time" not in frame.columns:
        if "timestamp" in frame.columns:
            frame = frame.rename(columns={"timestamp": "%time"})
        else:
            raise ValueError(f"{source_name} rows must include '%time' or 'timestamp'")
    frame["%time"] = frame["%time"].astype(str)
    return frame


def _normalize_merged_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["%time", *FEATURE_COLUMNS])

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if "%time" not in item:
            if "timestamp" in item:
                item["%time"] = item["timestamp"]
            else:
                raise ValueError("merged_rows entries must include '%time' or 'timestamp'")

        greenhouse = item.pop("greenhouse", None)
        weather = item.pop("weather", None)
        if isinstance(greenhouse, dict):
            item.update(greenhouse)
        if isinstance(weather, dict):
            item.update(weather)

        normalized_rows.append(item)

    frame = pd.DataFrame(normalized_rows)
    frame["%time"] = frame["%time"].astype(str)
    return frame


def _merge_sources(
    greenhouse_rows: list[dict[str, Any]],
    weather_rows: list[dict[str, Any]],
    merged_rows: list[dict[str, Any]],
    merge_strategy: str,
) -> pd.DataFrame:
    if merge_strategy not in {"inner", "left", "right", "outer"}:
        raise ValueError("merge_strategy must be one of: inner, left, right, outer")

    if merged_rows:
        merged = _normalize_merged_rows(merged_rows)
    else:
        greenhouse = _normalize_table_rows(greenhouse_rows, "greenhouse")
        weather = _normalize_table_rows(weather_rows, "weather")
        if greenhouse.empty and weather.empty:
            raise ValueError(
                "Provide either merged_rows or greenhouse_rows/weather_rows for automated pipeline"
            )
        if greenhouse.empty:
            merged = weather.copy()
        elif weather.empty:
            merged = greenhouse.copy()
        else:
            merged = greenhouse.merge(weather, on="%time", how=merge_strategy)

    merged = merged.sort_values("%time").drop_duplicates(subset=["%time"], keep="last")
    return merged


def _fill_features(frame: pd.DataFrame, fill_method: str) -> pd.DataFrame:
    cleaned = frame.copy()

    for col in FEATURE_COLUMNS:
        if col not in cleaned.columns:
            cleaned[col] = np.nan

    cleaned = cleaned[FEATURE_COLUMNS]
    for col in FEATURE_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    if fill_method == "interpolate":
        cleaned = cleaned.interpolate(method="linear", limit_direction="both", axis=0)
        cleaned = cleaned.ffill().bfill()
    elif fill_method == "ffill":
        cleaned = cleaned.ffill().bfill()
    elif fill_method == "bfill":
        cleaned = cleaned.bfill().ffill()
    elif fill_method == "zero":
        cleaned = cleaned.fillna(0.0)
    else:
        raise ValueError("fill_method must be one of: interpolate, ffill, bfill, zero")

    cleaned = cleaned.fillna(cleaned.median(numeric_only=True))
    cleaned = cleaned.fillna(0.0)
    cleaned = cleaned.astype(np.float32)

    if cleaned.empty:
        raise ValueError("No valid rows after automated preprocessing")
    if not np.isfinite(cleaned.to_numpy()).all():
        raise ValueError("Automated preprocessing generated non-finite feature values")

    return cleaned


def prepare_model_window(
    *,
    greenhouse_rows: list[dict[str, Any]],
    weather_rows: list[dict[str, Any]],
    merged_rows: list[dict[str, Any]],
    options: PipelineOptions,
    trained_feature_scaler,
) -> PreparedWindow:
    merged = _merge_sources(
        greenhouse_rows=greenhouse_rows,
        weather_rows=weather_rows,
        merged_rows=merged_rows,
        merge_strategy=options.merge_strategy,
    )
    cleaned = _fill_features(merged, fill_method=options.fill_method)

    if len(cleaned) < options.lookback:
        raise ValueError(
            f"Need at least {options.lookback} merged rows, got {len(cleaned)}"
        )

    raw_window = cleaned.tail(options.lookback).to_numpy(dtype=np.float32)

    if options.scaling_mode == "trained":
        scaler_used = trained_feature_scaler
        scaled_window = scaler_used.transform(raw_window)
    elif options.scaling_mode == "dynamic":
        scaler_used = StandardScaler()
        scaled_window = scaler_used.fit_transform(raw_window)
    else:
        raise ValueError("scaling_mode must be one of: trained, dynamic")

    if not np.isfinite(scaled_window).all():
        raise ValueError("Scaled pipeline window contains non-finite values")

    metadata = {
        "row_count_merged": int(len(merged)),
        "row_count_cleaned": int(len(cleaned)),
        "lookback": options.lookback,
        "merge_strategy": options.merge_strategy,
        "fill_method": options.fill_method,
        "scaling_mode": options.scaling_mode,
        "has_climate_input": bool(greenhouse_rows),
        "has_weather_input": bool(weather_rows),
        "has_merged_input": bool(merged_rows),
        "climate_columns": CLIMATE_COLUMNS,
        "weather_columns": WEATHER_COLUMNS,
    }

    return PreparedWindow(
        raw_window=raw_window,
        scaled_window=scaled_window.astype(np.float32),
        scaler_used=scaler_used,
        metadata=metadata,
    )
