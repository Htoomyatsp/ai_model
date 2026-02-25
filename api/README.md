# Greenhouse AI Platform

This API + web dashboard provides:
- enhanced deep learning architectures (baseline, hybrid, multi-input)
- automated data processing and optional dynamic scaling
- MPC simulation and scenario benchmarking
- trend visualization and explainability heatmaps
- incremental continuous-learning updates

## Quick Start
```bash
cd /Users/nickcecchin/Desktop/ai_model
source /Users/nickcecchin/Desktop/ai_model/api/.venv/bin/activate
pip install -r /Users/nickcecchin/Desktop/ai_model/api/requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open: `http://127.0.0.1:8000`

## Model Training (Baseline + Enhanced)
Train and compare baseline vs enhanced models:
```bash
python /Users/nickcecchin/Desktop/ai_model/api/train_hybrid_models.py --architecture all
```

Supported architectures:
- `baseline_lstm`
- `lstm_cnn`
- `bi_lstm`
- `multi_input_hybrid` (separate greenhouse/weather branches)

Outputs in `/Users/nickcecchin/Desktop/ai_model/api/checkpoint/`:
- per-model folders (`baseline_lstm/`, `lstm_cnn/`, `bi_lstm/`, `multi_input_hybrid/`)
- `model.keras`, `feature_scaler.save`, `target_scaler.save` (best promoted model)
- `model_metadata.json`
- `training_summary.json`
- `model_comparison.json`, `model_comparison.csv` (RMSE + MAPE comparison)

## Benchmark Existing Checkpoints
```bash
python /Users/nickcecchin/Desktop/ai_model/api/benchmark_models.py
```
Produces `benchmark_comparison.json` and `benchmark_comparison.csv`.

## Automated Data Processing
Batch processing of `GreenhouseClimate.csv` + `Weather.csv`:
```bash
python /Users/nickcecchin/Desktop/ai_model/api/automated_data_pipeline.py
```
Produces cleaned merged features and summary stats in `checkpoint/`.

## Continuous Learning (Incremental Update)
API endpoint:
- `POST /continuous/update`

This ingests newly appended CSV rows, updates scalers incrementally, and fine-tunes the deployed model.

Example request:
```json
{
  "new_rows_limit": 4000,
  "fine_tune_epochs": 2,
  "batch_size": 64,
  "dry_run": false
}
```

## Core API Endpoints

### Prediction
- `POST /predict`

Supports two modes:
- direct mode: provide `data` (`lookback x feature_count`)
- automated mode: `use_automated_pipeline=true` + `automated_pipeline` payload

### Pipeline
- `GET /pipeline/default-config`
- `POST /pipeline/prepare`

Pipeline options:
- `merge_strategy`: `inner|left|right|outer`
- `fill_method`: `interpolate|ffill|bfill|zero`
- `scaling_mode`: `trained|dynamic`

### MPC
- `GET /mpc/default-config`
- `POST /mpc/simulate`
- `POST /mpc/evaluate-scenarios`

`/mpc/simulate` returns:
- `energy_efficiency_score`
- `stability_index`
- `objective_trace`
- predicted states and applied control actions

Scenario benchmarking endpoint ranks scenarios by efficiency/stability.

### Explainability
- `GET /explain/default-config`
- `POST /explain`

Methods:
- `gradient_attention` (attention-style gradient heatmap)
- `shap_approx` (perturbation SHAP-like attribution)

### Model Metadata / Comparison
- `GET /model-info`
- `GET /model/comparison`

## Visualization Dashboard
The web UI includes tabs for:
- Predict
- MPC Control
- Explainability
- Code Runner

Features:
- historical trend lines and feature cards
- explainability heatmap + top contributors
- MPC trajectories (controlled/predicted variables)
- scenario benchmark output

## Deployment Notes
- Use Python 3.13-compatible stack from `requirements.txt`.
- Keep `checkpoint/model_metadata.json` aligned with deployed model.
- For production deployment, run behind a process manager (e.g., systemd, supervisor, or container orchestration) and disable `--reload`.
