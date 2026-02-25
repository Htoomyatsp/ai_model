from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import joblib
import keras

try:
    from api.ml.constants import FEATURE_COLUMNS
    from api.ml.mpc import run_mpc_feedback_loop
except ModuleNotFoundError:
    from ml.constants import FEATURE_COLUMNS
    from ml.mpc import run_mpc_feedback_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MPC performance across greenhouse scenarios.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(__file__).resolve().parent / "checkpoint")
    parser.add_argument("--sample-json", type=Path, default=Path(__file__).resolve().parent / "example_request1.json")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=80)
    return parser.parse_args()


def build_predict_fn(model, architecture_kind: str, feature_names: list[str]):
    if architecture_kind != "multi_input":
        return lambda scaled_window: model.predict(scaled_window.reshape(1, scaled_window.shape[0], scaled_window.shape[1]), verbose=0)

    climate = ["CO2air", "Cum_irr", "Tair", "Tot_PAR", "Ventwind", "AssimLight", "VentLee", "HumDef", "co2_dos", "PipeGrow", "EnScr", "BlackScr"]
    weather = ["Windsp", "Winddir", "Tout", "Rhout", "AbsHumOut", "PARout", "Iglob", "Pyrgeo", "RadSum"]
    c_idx = [feature_names.index(c) for c in climate if c in feature_names]
    w_idx = [feature_names.index(c) for c in weather if c in feature_names]

    def _predict(scaled_window):
        x = np.asarray(scaled_window, dtype=np.float32)
        x_c = x[:, c_idx].reshape(1, x.shape[0], len(c_idx))
        x_w = x[:, w_idx].reshape(1, x.shape[0], len(w_idx))
        return model.predict([x_c, x_w], verbose=0)

    return _predict


def main() -> None:
    args = parse_args()
    meta = json.loads((args.checkpoint_dir / "model_metadata.json").read_text(encoding="utf-8"))
    feature_names = meta.get("features", FEATURE_COLUMNS)
    model = keras.models.load_model(args.checkpoint_dir / "model.keras")
    feature_scaler = joblib.load(args.checkpoint_dir / "feature_scaler.save")
    target_scaler = joblib.load(args.checkpoint_dir / "target_scaler.save")
    predict_fn = build_predict_fn(model, meta.get("architecture_kind", "single_input"), feature_names)

    sample = json.loads(args.sample_json.read_text(encoding="utf-8"))
    base_window = np.array(sample["data"], dtype=float)

    scenarios = [
        {"name": "Balanced", "target_setpoints": {"Tair": 21.0, "CO2air": 500.0, "HumDef": 6.0}},
        {"name": "EnergySaver", "target_setpoints": {"Tair": 20.0, "CO2air": 470.0, "HumDef": 6.5}, "control_weights": {"_magnitude": 0.08}},
        {"name": "HighGrowth", "target_setpoints": {"Tair": 22.5, "CO2air": 650.0, "HumDef": 5.8}, "control_weights": {"_magnitude": 0.01}},
    ]

    results = []
    for s in scenarios:
        out = run_mpc_feedback_loop(
            initial_history=base_window.copy(),
            feature_names=feature_names,
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            steps=args.steps,
            horizon=args.horizon,
            candidate_sequences=args.candidates,
            random_seed=42,
            target_setpoints=s.get("target_setpoints", {}),
            control_weights=s.get("control_weights", {}),
            predict_fn=predict_fn,
        )
        results.append({
            "name": s["name"],
            "energy_efficiency_score": out["energy_efficiency_score"],
            "stability_index": out["stability_index"],
            "objective_last": out["objective_trace"][-1] if out["objective_trace"] else None,
        })

    results = sorted(results, key=lambda r: (-r["energy_efficiency_score"], r["stability_index"]))
    summary = {"ranking": results}
    out_path = args.checkpoint_dir / "mpc_scenario_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
