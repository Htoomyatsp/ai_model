from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from api.ml.data import load_feature_frame
except ModuleNotFoundError:
    from ml.data import load_feature_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate preprocessing/merging of greenhouse + weather CSVs.")
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
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoint" / "processed_merged_features.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoint" / "processed_merged_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    frame = load_feature_frame(climate_csv=args.climate_csv, weather_csv=args.weather_csv)
    frame.to_csv(args.output_csv, index=False)

    summary = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "output_csv": str(args.output_csv),
        "column_stats": {
            c: {
                "min": float(frame[c].min()),
                "max": float(frame[c].max()),
                "mean": float(frame[c].mean()),
            }
            for c in frame.columns
        },
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
