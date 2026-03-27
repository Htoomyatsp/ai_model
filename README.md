# Greenhouse AI Platform

A machine learning and control platform for greenhouse climate forecasting, model benchmarking, explainability, and model predictive control.

This project combines deep learning with operational decision support. It predicts key greenhouse climate variables from recent greenhouse and weather history, compares multiple model architectures, explains model behavior, and simulates control strategies through MPC.

## Overview

The platform is built around a few core ideas:

- forecast future greenhouse conditions from recent sensor and weather data
- compare multiple neural network architectures and deploy the best-performing model
- explain predictions with feature importance and timestep heatmaps
- simulate control strategies with model predictive control
- provide a browser-based dashboard for demos, inspection, and technical use

## Main Capabilities

- **Project Summary / Dashboard**
  - high-level overview of the system, deployed model, and key outputs

- **Prediction**
  - run next-step greenhouse climate predictions from model input windows

- **Model Comparison**
  - compare trained architectures using RMSE, MAE, and related metrics

- **MPC Control / Simulation**
  - simulate control trajectories and benchmark control scenarios

- **Explainability**
  - inspect feature importance and attribution heatmaps for predictions

- **Charts / Trends**
  - visualize recent greenhouse and weather behavior over time

- **Developer Tools**
  - browse and run project Python scripts from the dashboard

- **Environment / Metadata**
  - inspect model metadata, schema details, and preprocessing defaults

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **ML / Data:** TensorFlow / Keras, NumPy, scikit-learn, joblib
- **Control:** custom MPC simulation pipeline

## Repository Structure

```text
ai_model/
├── api/
│   ├── main.py
│   ├── ui.html
│   ├── train_hybrid_models.py
│   ├── benchmark_models.py
│   ├── automated_data_pipeline.py
│   ├── evaluate_mpc_scenarios.py
│   ├── ml/
│   └── requirements.txt
├── greenhouse_code/
│   ├── GreenhouseClimate.csv
│   ├── Weather.csv
│   └── ...
└── web-system/
