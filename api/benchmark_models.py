from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from api.ml.constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS, TARGET_COLUMNS
    from api.ml.data import load_feature_frame, split_scale_frame
except ModuleNotFoundError:
    from ml.constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS, TARGET_COLUMNS
    from ml.data import load_feature_frame, split_scale_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark all trained models against each other and a naive persistence baseline."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoint",
    )
    parser.add_argument(
        "--climate-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "greenhouse_code" / "GreenhouseClimate.csv",
    )
    parser.add_argument(
        "--weather-csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "greenhouse_code" / "Weather.csv",
    )
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    safe = np.where(denom < 1e-8, 1e-8, denom)
    return float(np.mean(np.abs(y_true - y_pred) / safe) * 100.0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> dict:
    row: dict = {
        "rmse":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":   float(mean_absolute_error(y_true, y_pred)),
        "smape": smape(y_true, y_pred),
        "r2_variance_weighted": float(
            r2_score(y_true, y_pred, multioutput="variance_weighted")
        ),
    }
    for i, name in enumerate(target_names):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        row[f"{name}_mae"]  = float(mean_absolute_error(col_true, col_pred))
        row[f"{name}_rmse"] = float(np.sqrt(mean_squared_error(col_true, col_pred)))
        row[f"{name}_r2"]   = float(r2_score(col_true, col_pred))
        row[f"{name}_smape"] = smape(col_true[:, None], col_pred[:, None])
    return row


# ---------------------------------------------------------------------------
# Input routing
# ---------------------------------------------------------------------------

def model_inputs(arch_kind: str, x_scaled: np.ndarray):
    if arch_kind != "multi_input":
        return x_scaled
    climate_idx = [FEATURE_COLUMNS.index(c) for c in CLIMATE_COLUMNS]
    weather_idx  = [FEATURE_COLUMNS.index(c) for c in WEATHER_COLUMNS]
    return [x_scaled[:, :, climate_idx], x_scaled[:, :, weather_idx]]


# ---------------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------------

def persistence_prediction(
    x_test: np.ndarray,
    target_scaler,
    target_names: list[str],
) -> np.ndarray:
    """
    Naive persistence: predict the last observed value of each target.

    x_test shape: (n_samples, lookback, n_features).
    The last time step of the window (x_test[:, -1, :]) contains the most
    recent observed values.  We extract the target columns from that step
    and inverse-transform to raw units.
    """
    target_col_indices = [FEATURE_COLUMNS.index(name) for name in target_names]
    last_step = x_test[:, -1, target_col_indices]   # (n, n_targets) — already scaled
    return target_scaler.inverse_transform(last_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    frame = load_feature_frame(args.climate_csv, args.weather_csv)
    splits = split_scale_frame(
        frame,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    y_test = splits.target_scaler.inverse_transform(splits.y_test)
    results: list[dict] = []

    # --- Persistence baseline ---
    # If trained models cannot beat this, the model adds zero value.
    persist_pred = persistence_prediction(splits.x_test, splits.target_scaler, TARGET_COLUMNS)
    persist_row = {"architecture": "persistence_baseline"}
    persist_row.update(compute_metrics(y_test, persist_pred, TARGET_COLUMNS))
    results.append(persist_row)

    # --- Trained models ---
    for sub in sorted(args.checkpoint_dir.iterdir()):
        if not sub.is_dir():
            continue
        model_path = sub / "model.keras"
        meta_path  = sub / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            continue

        meta     = json.loads(meta_path.read_text(encoding="utf-8"))
        arch     = meta.get("architecture", sub.name)
        arch_kind = meta.get("architecture_kind", "single_input")

        model      = keras.models.load_model(model_path)
        pred_scaled = model.predict(model_inputs(arch_kind, splits.x_test), verbose=0)
        pred       = splits.target_scaler.inverse_transform(pred_scaled)

        row = {"architecture": arch}
        row.update(compute_metrics(y_test, pred, TARGET_COLUMNS))
        results.append(row)

    if len(results) <= 1:
        raise SystemExit("No trained model folders found in checkpoint directory.")

    df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    # Highlight whether each model beats the persistence baseline
    persist_rmse = float(df.loc[df["architecture"] == "persistence_baseline", "rmse"].iloc[0])
    df["beats_persistence"] = df["rmse"] < persist_rmse

    comparison = {
        "persistence_baseline_rmse": persist_rmse,
        "ranking": df.to_dict(orient="records"),
        "best_model": df[df["architecture"] != "persistence_baseline"].iloc[0].to_dict(),
    }

    out_json = args.checkpoint_dir / "benchmark_comparison.json"
    out_csv  = args.checkpoint_dir / "benchmark_comparison.csv"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
