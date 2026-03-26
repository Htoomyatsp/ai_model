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
    from api.ml.constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS, TARGET_COLUMNS
    from api.ml.data import load_feature_frame, split_scale_frame
except ModuleNotFoundError:
    from ml.architectures import build_model
    from ml.constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS, TARGET_COLUMNS
    from ml.data import load_feature_frame, split_scale_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train greenhouse forecasting models and compare by RMSE / per-target metrics."
    )
    parser.add_argument(
        "--architecture",
        choices=[
            "baseline_lstm", "lstm_cnn", "bi_lstm", "temporal_conv",
            "multi_input_hybrid", "all", "enhanced",
        ],
        default="all",
        help=(
            "Architecture to train. "
            "'all' trains all five. "
            "'enhanced' skips baseline_lstm."
        ),
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
    parser.add_argument(
        "--lookback", type=int, default=48,
        help=(
            "Input window length. Default 48 steps. "
            "At 5-min sampling = 4 h; at 15-min = 12 h. "
            "Increase to 96 if your dataset has ≥ 5000 training rows."
        ),
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="Max epochs. EarlyStopping (patience=15) will halt earlier.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoint",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE.  Well-defined even when y_true is near or at zero
    (e.g., HumDef at 100 % RH), unlike standard MAPE.
    Range: [0, 200].
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    safe_denom = np.where(denom < 1e-8, 1e-8, denom)
    return float(np.mean(np.abs(y_true - y_pred) / safe_denom) * 100.0)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> dict:
    """
    Returns aggregate metrics plus a per-target breakdown.

    Aggregate R² uses variance_weighted multioutput so a target with naturally
    higher variance does not dominate the score unfairly.
    """
    aggregate = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": smape(y_true, y_pred),
        "r2_variance_weighted": float(
            r2_score(y_true, y_pred, multioutput="variance_weighted")
        ),
    }

    per_target: dict[str, dict] = {}
    for i, name in enumerate(target_names):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        per_target[name] = {
            "mae": float(mean_absolute_error(col_true, col_pred)),
            "rmse": float(np.sqrt(mean_squared_error(col_true, col_pred))),
            "smape": smape(col_true[:, None], col_pred[:, None]),
            "r2": float(r2_score(col_true, col_pred)),
        }

    return {"aggregate": aggregate, "per_target": per_target}


# ---------------------------------------------------------------------------
# Input routing
# ---------------------------------------------------------------------------

def model_inputs(architecture: str, x: np.ndarray):
    if architecture == "multi_input_hybrid":
        climate_idx = [FEATURE_COLUMNS.index(c) for c in CLIMATE_COLUMNS]
        weather_idx = [FEATURE_COLUMNS.index(c) for c in WEATHER_COLUMNS]
        return [x[:, :, climate_idx], x[:, :, weather_idx]]
    return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

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
    best_ckpt = model_dir / "best.keras"

    model = build_model(
        architecture=architecture,
        input_shape=(splits.x_train.shape[1], splits.x_train.shape[2]),
        output_dim=splits.y_train.shape[1],
        climate_feature_count=len(CLIMATE_COLUMNS),
        weather_feature_count=len(WEATHER_COLUMNS),
    )

    callbacks = [
        TerminateOnNaN(),
        # Patience 15: gives the model room to climb out of local plateaus,
        # especially important early in training with Huber loss.
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        # LR halves after 6 epochs without improvement; min_lr prevents it
        # from decaying into vanishing-gradient territory.
        ReduceLROnPlateau(monitor="val_loss", patience=6, factor=0.5, min_lr=1e-6),
        ModelCheckpoint(filepath=best_ckpt, monitor="val_loss", save_best_only=True),
    ]

    x_tr = model_inputs(architecture, splits.x_train)
    x_va = model_inputs(architecture, splits.x_val)
    x_te = model_inputs(architecture, splits.x_test)

    model.fit(
        x_tr,
        splits.y_train,
        validation_data=(x_va, splits.y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    from keras import models as km
    if not best_ckpt.exists():
        raise ValueError(
            f"No checkpoint was saved for {architecture}. "
            "NaN loss is likely — check the data pipeline."
        )
    best_model = km.load_model(best_ckpt)

    pred_scaled = best_model.predict(x_te, verbose=0)
    pred = splits.target_scaler.inverse_transform(pred_scaled)
    y_test_raw = splits.target_scaler.inverse_transform(splits.y_test)

    metrics = evaluate_predictions(y_test_raw, pred, TARGET_COLUMNS)
    if not all(np.isfinite(v) for v in metrics["aggregate"].values()):
        raise ValueError(f"Non-finite aggregate metrics for {architecture}: {metrics['aggregate']}")

    best_model.save(model_dir / "model.keras")
    joblib.dump(splits.feature_scaler, model_dir / "feature_scaler.save")
    joblib.dump(splits.target_scaler, model_dir / "target_scaler.save")

    pd.DataFrame(model.history.history).to_csv(model_dir / "history.csv", index=False)

    metadata = {
        "architecture": architecture,
        "architecture_kind": "multi_input" if architecture == "multi_input_hybrid" else "single_input",
        "lookback": lookback,
        "horizon": horizon,
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "target_features": TARGET_COLUMNS,
        "climate_features": CLIMATE_COLUMNS,
        "weather_features": WEATHER_COLUMNS,
        "metrics": metrics,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "architecture": architecture,
        "architecture_kind": metadata["architecture_kind"],
        "model_dir": str(model_dir),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Comparison and promotion
# ---------------------------------------------------------------------------

def build_ranking(results: list[dict]) -> list[dict]:
    ordered = sorted(results, key=lambda r: r["metrics"]["aggregate"]["rmse"])
    return [
        {
            "rank": i + 1,
            "architecture": r["architecture"],
            "rmse": r["metrics"]["aggregate"]["rmse"],
            "mae": r["metrics"]["aggregate"]["mae"],
            "smape": r["metrics"]["aggregate"]["smape"],
            "r2_variance_weighted": r["metrics"]["aggregate"]["r2_variance_weighted"],
            "per_target": r["metrics"]["per_target"],
        }
        for i, r in enumerate(ordered)
    ]


def promote_best_model(results: list[dict], output_root: Path) -> dict:
    best = min(results, key=lambda r: r["metrics"]["aggregate"]["rmse"])
    best_dir = Path(best["model_dir"])

    for target_name, source_path in [
        ("model.keras",          best_dir / "model.keras"),
        ("feature_scaler.save",  best_dir / "feature_scaler.save"),
        ("target_scaler.save",   best_dir / "target_scaler.save"),
        ("model_metadata.json",  best_dir / "metadata.json"),
    ]:
        shutil.copy2(source_path, output_root / target_name)

    ranking = build_ranking(results)
    summary = {
        "best_architecture": best["architecture"],
        "ranking_by_rmse": ranking,
    }
    (output_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {
            "architecture": r["architecture"],
            "rmse":  r["metrics"]["aggregate"]["rmse"],
            "mae":   r["metrics"]["aggregate"]["mae"],
            "smape": r["metrics"]["aggregate"]["smape"],
            "r2_variance_weighted": r["metrics"]["aggregate"]["r2_variance_weighted"],
            **{
                f"{tgt}_{k}": v
                for tgt, tmetrics in r["metrics"]["per_target"].items()
                for k, v in tmetrics.items()
            },
        }
        for r in results
    ]).sort_values("rmse").to_csv(output_root / "model_comparison.csv", index=False)

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_feature_frame(climate_csv=args.climate_csv, weather_csv=args.weather_csv)

    # Correct chronological split: raw rows first, then sequences, scaler fit
    # on training rows only.  See ml/data.py:split_scale_frame for details.
    splits = split_scale_frame(
        frame=frame,
        lookback=args.lookback,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print(
        f"Dataset: {len(frame)} rows → "
        f"train={len(splits.x_train)}, val={len(splits.x_val)}, test={len(splits.x_test)} sequences"
    )

    if args.architecture == "all":
        architectures = ["baseline_lstm", "lstm_cnn", "bi_lstm", "temporal_conv", "multi_input_hybrid"]
    elif args.architecture == "enhanced":
        architectures = ["lstm_cnn", "bi_lstm", "temporal_conv", "multi_input_hybrid"]
    else:
        architectures = [args.architecture]

    results = []
    for arch in architectures:
        print(f"\n{'='*60}\nTraining: {arch}\n{'='*60}")
        result = train_one_architecture(
            architecture=arch,
            splits=splits,
            output_root=args.output_dir,
            lookback=args.lookback,
            horizon=args.horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        results.append(result)
        # Quick per-target summary after each model
        pt = result["metrics"]["per_target"]
        for tgt, m in pt.items():
            print(f"  {tgt}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}")

    summary = promote_best_model(results=results, output_root=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
