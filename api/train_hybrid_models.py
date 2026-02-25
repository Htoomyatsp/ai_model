from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from api.ml.architectures import build_model
    from api.ml.constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS
    from api.ml.data import create_sequences, load_feature_frame, split_scale_sequences
except ModuleNotFoundError:
    from ml.architectures import build_model
    from ml.constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS
    from ml.data import create_sequences, load_feature_frame, split_scale_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train greenhouse forecasting models with baseline, hybrid, and multi-input architectures "
            "and compare RMSE/MAPE."
        )
    )
    parser.add_argument(
        "--architecture",
        choices=["baseline_lstm", "lstm_cnn", "bi_lstm", "multi_input_hybrid", "all", "enhanced"],
        default="all",
        help="Architecture to train. 'enhanced' trains all except baseline_lstm.",
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
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoint",
    )
    return parser.parse_args()


def split_scaled_inputs(x_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    climate_idx = [FEATURE_COLUMNS.index(c) for c in CLIMATE_COLUMNS]
    weather_idx = [FEATURE_COLUMNS.index(c) for c in WEATHER_COLUMNS]
    return x_scaled[:, :, climate_idx], x_scaled[:, :, weather_idx]


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def model_inputs(architecture: str, x: np.ndarray):
    if architecture == "multi_input_hybrid":
        x_climate, x_weather = split_scaled_inputs(x)
        return [x_climate, x_weather]
    return x


def architecture_kind(architecture: str) -> str:
    return "multi_input" if architecture == "multi_input_hybrid" else "single_input"


def train_one_architecture(
    architecture: str,
    splits,
    output_root: Path,
    lookback: int,
    horizon: int,
    epochs: int,
    batch_size: int,
) -> dict:
    model_dir = output_root / architecture
    model_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = model_dir / "best.keras"

    model = build_model(
        architecture=architecture,
        input_shape=(splits.x_train.shape[1], splits.x_train.shape[2]),
        output_dim=splits.y_train.shape[1],
        climate_feature_count=len(CLIMATE_COLUMNS),
        weather_feature_count=len(WEATHER_COLUMNS),
    )

    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=4, factor=0.5, min_lr=1e-5),
        ModelCheckpoint(filepath=best_checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    x_train_in = model_inputs(architecture, splits.x_train)
    x_val_in = model_inputs(architecture, splits.x_val)
    x_test_in = model_inputs(architecture, splits.x_test)

    history = model.fit(
        x_train_in,
        splits.y_train,
        validation_data=(x_val_in, splits.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    from keras import models as keras_models

    if not best_checkpoint_path.exists():
        raise ValueError(
            f"Training for {architecture} did not produce a valid checkpoint. "
            "This usually means non-finite data caused NaN loss."
        )
    best_model = keras_models.load_model(best_checkpoint_path)

    test_pred_scaled = best_model.predict(x_test_in, verbose=0)
    test_pred = splits.target_scaler.inverse_transform(test_pred_scaled)
    y_test = splits.target_scaler.inverse_transform(splits.y_test)

    metrics = evaluate_predictions(y_test, test_pred)
    if not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError(f"Non-finite evaluation metrics for {architecture}: {metrics}")

    best_model.save(model_dir / "model.keras")
    joblib.dump(splits.feature_scaler, model_dir / "feature_scaler.save")
    joblib.dump(splits.target_scaler, model_dir / "target_scaler.save")

    pd.DataFrame(history.history).to_csv(model_dir / "history.csv", index=False)

    metadata = {
        "architecture": architecture,
        "architecture_kind": architecture_kind(architecture),
        "lookback": lookback,
        "horizon": horizon,
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "climate_features": CLIMATE_COLUMNS,
        "weather_features": WEATHER_COLUMNS,
        "metrics": metrics,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "architecture": architecture,
        "architecture_kind": architecture_kind(architecture),
        "model_dir": str(model_dir),
        "metrics": metrics,
    }


def build_comparison(results: list[dict]) -> dict:
    ordered = sorted(results, key=lambda r: r["metrics"]["rmse"])
    baseline = next((r for r in results if r["architecture"] == "baseline_lstm"), None)
    enhanced = [r for r in results if r["architecture"] != "baseline_lstm"]
    best_enhanced = min(enhanced, key=lambda r: r["metrics"]["rmse"]) if enhanced else None

    comparison = {
        "ranking_by_rmse": [
            {
                "rank": i + 1,
                "architecture": r["architecture"],
                "rmse": r["metrics"]["rmse"],
                "mape": r["metrics"]["mape"],
            }
            for i, r in enumerate(ordered)
        ],
        "baseline": baseline,
        "best_enhanced": best_enhanced,
    }

    if baseline and best_enhanced:
        baseline_rmse = baseline["metrics"]["rmse"]
        baseline_mape = baseline["metrics"]["mape"]
        comparison["improvement_vs_baseline"] = {
            "rmse_delta": best_enhanced["metrics"]["rmse"] - baseline_rmse,
            "rmse_pct": ((baseline_rmse - best_enhanced["metrics"]["rmse"]) / max(baseline_rmse, 1e-9)) * 100.0,
            "mape_delta": best_enhanced["metrics"]["mape"] - baseline_mape,
            "mape_pct": ((baseline_mape - best_enhanced["metrics"]["mape"]) / max(baseline_mape, 1e-9)) * 100.0,
        }

    return comparison


def promote_best_model(results: list[dict], output_root: Path) -> dict:
    best = min(results, key=lambda r: r["metrics"]["rmse"])
    best_dir = Path(best["model_dir"])

    promoted_files = {
        "model.keras": best_dir / "model.keras",
        "feature_scaler.save": best_dir / "feature_scaler.save",
        "target_scaler.save": best_dir / "target_scaler.save",
        "model_metadata.json": best_dir / "metadata.json",
    }

    for target_name, source_path in promoted_files.items():
        shutil.copy2(source_path, output_root / target_name)

    comparison = build_comparison(results)

    summary = {
        "best_architecture": best["architecture"],
        "results": results,
        "comparison": comparison,
    }
    (output_root / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_root / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "architecture": r["architecture"],
                "rmse": r["metrics"]["rmse"],
                "mape": r["metrics"]["mape"],
                "mae": r["metrics"]["mae"],
                "r2": r["metrics"]["r2"],
            }
            for r in results
        ]
    ).sort_values("rmse").to_csv(output_root / "model_comparison.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_feature_frame(climate_csv=args.climate_csv, weather_csv=args.weather_csv)
    x_values, y_values = create_sequences(frame=frame, lookback=args.lookback, horizon=args.horizon)
    splits = split_scale_sequences(
        x_values=x_values,
        y_values=y_values,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    if args.architecture == "all":
        architectures = ["baseline_lstm", "lstm_cnn", "bi_lstm", "multi_input_hybrid"]
    elif args.architecture == "enhanced":
        architectures = ["lstm_cnn", "bi_lstm", "multi_input_hybrid"]
    else:
        architectures = [args.architecture]

    results = []
    for architecture in architectures:
        result = train_one_architecture(
            architecture=architecture,
            splits=splits,
            output_root=args.output_dir,
            lookback=args.lookback,
            horizon=args.horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results.append(result)

    summary = promote_best_model(results=results, output_root=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
