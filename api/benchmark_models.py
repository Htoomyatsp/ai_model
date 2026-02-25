from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from api.ml.constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS
    from api.ml.data import create_sequences, load_feature_frame, split_scale_sequences
except ModuleNotFoundError:
    from ml.constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS
    from ml.data import create_sequences, load_feature_frame, split_scale_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs enhanced models by RMSE/MAPE.")
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
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def model_inputs(arch_kind: str, x_scaled: np.ndarray):
    if arch_kind != "multi_input":
        return x_scaled
    climate_idx = [FEATURE_COLUMNS.index(c) for c in CLIMATE_COLUMNS]
    weather_idx = [FEATURE_COLUMNS.index(c) for c in WEATHER_COLUMNS]
    return [x_scaled[:, :, climate_idx], x_scaled[:, :, weather_idx]]


def main() -> None:
    args = parse_args()
    frame = load_feature_frame(args.climate_csv, args.weather_csv)
    x_vals, y_vals = create_sequences(frame, lookback=args.lookback, horizon=args.horizon)
    splits = split_scale_sequences(x_vals, y_vals, args.train_ratio, args.val_ratio)

    y_test = splits.target_scaler.inverse_transform(splits.y_test)
    results = []

    for sub in sorted(args.checkpoint_dir.iterdir()):
        if not sub.is_dir():
            continue
        model_path = sub / "model.keras"
        meta_path = sub / "metadata.json"
        if not model_path.exists() or not meta_path.exists():
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arch = meta.get("architecture", sub.name)
        arch_kind = meta.get("architecture_kind", "single_input")

        model = keras.models.load_model(model_path)
        x_test_in = model_inputs(arch_kind, splits.x_test)
        pred_scaled = model.predict(x_test_in, verbose=0)
        pred = splits.target_scaler.inverse_transform(pred_scaled)

        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        mape_val = mape(y_test, pred)

        results.append({"architecture": arch, "architecture_kind": arch_kind, "rmse": rmse, "mae": mae, "mape": mape_val})

    if not results:
        raise SystemExit("No trained model folders found in checkpoint directory.")

    df = pd.DataFrame(results).sort_values("rmse")
    comparison = {
        "ranking": df.to_dict(orient="records"),
        "best": df.iloc[0].to_dict(),
    }

    baseline = df[df["architecture"] == "baseline_lstm"]
    enhanced = df[df["architecture"] != "baseline_lstm"]
    if not baseline.empty and not enhanced.empty:
        b = baseline.iloc[0]
        e = enhanced.iloc[0]
        comparison["baseline_vs_best_enhanced"] = {
            "baseline": b.to_dict(),
            "best_enhanced": e.to_dict(),
            "rmse_improvement_pct": float((b["rmse"] - e["rmse"]) / max(b["rmse"], 1e-9) * 100.0),
            "mape_improvement_pct": float((b["mape"] - e["mape"]) / max(b["mape"], 1e-9) * 100.0),
        }

    out_json = args.checkpoint_dir / "benchmark_comparison.json"
    out_csv = args.checkpoint_dir / "benchmark_comparison.csv"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    df.to_csv(out_csv, index=False)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
