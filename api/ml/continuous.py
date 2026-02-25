from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from keras import Model

from .constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS
from .data import create_sequences, load_feature_frame


def _model_inputs(architecture_kind: str, x_scaled: np.ndarray):
    if architecture_kind == "multi_input":
        climate_idx = [FEATURE_COLUMNS.index(c) for c in CLIMATE_COLUMNS]
        weather_idx = [FEATURE_COLUMNS.index(c) for c in WEATHER_COLUMNS]
        return [x_scaled[:, :, climate_idx], x_scaled[:, :, weather_idx]]
    return x_scaled


def run_incremental_update(
    *,
    checkpoint_dir: Path,
    model: Model,
    feature_scaler,
    target_scaler,
    climate_csv: Path,
    weather_csv: Path,
    lookback: int,
    architecture_kind: str,
    fine_tune_epochs: int,
    batch_size: int,
    new_rows_limit: int,
    dry_run: bool,
) -> dict[str, Any]:
    state_path = checkpoint_dir / "stream_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"last_seen_rows": 0}

    frame = load_feature_frame(climate_csv=climate_csv, weather_csv=weather_csv)
    total_rows = len(frame)
    start_idx = int(state.get("last_seen_rows", 0))

    if start_idx >= total_rows:
        return {
            "updated": False,
            "reason": "No new rows available",
            "total_rows": total_rows,
            "last_seen_rows": start_idx,
        }

    new_frame = frame.iloc[start_idx:]
    if len(new_frame) > new_rows_limit:
        new_frame = new_frame.iloc[-new_rows_limit:]

    if len(new_frame) < lookback + 1:
        return {
            "updated": False,
            "reason": f"Not enough new rows for incremental training (need at least {lookback + 1})",
            "new_rows": int(len(new_frame)),
            "total_rows": total_rows,
            "last_seen_rows": start_idx,
        }

    x_new_raw, y_new_raw = create_sequences(new_frame, lookback=lookback, horizon=1)

    feature_scaler.partial_fit(x_new_raw.reshape(-1, x_new_raw.shape[-1]))
    target_scaler.partial_fit(y_new_raw)

    x_new_scaled = feature_scaler.transform(x_new_raw.reshape(-1, x_new_raw.shape[-1])).reshape(x_new_raw.shape)
    y_new_scaled = target_scaler.transform(y_new_raw)

    x_fit = _model_inputs(architecture_kind, x_new_scaled)

    history = model.fit(
        x_fit,
        y_new_scaled,
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        verbose=0,
    )

    new_last_seen = total_rows

    if not dry_run:
        model.save(checkpoint_dir / "model.keras")
        joblib.dump(feature_scaler, checkpoint_dir / "feature_scaler.save")
        joblib.dump(target_scaler, checkpoint_dir / "target_scaler.save")
        state["last_seen_rows"] = new_last_seen
        state["updated_at_rows"] = total_rows
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    losses = history.history.get("loss", [])
    return {
        "updated": True,
        "dry_run": dry_run,
        "new_rows_used": int(len(new_frame)),
        "training_windows": int(len(x_new_raw)),
        "epochs": fine_tune_epochs,
        "loss_history": [float(v) for v in losses],
        "last_seen_rows_before": start_idx,
        "last_seen_rows_after": new_last_seen,
        "total_rows": total_rows,
    }
