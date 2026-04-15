# TerraMind Inference Integration Guide (Standard + Federated + Ensemble)

This document explains how TerraMind executes and integrates three inference paths:

1. Standard model (Random Forest from `ml/`)
2. Federated model (PyTorch `AdvisorNet` saved from FL simulation)
3. Ensemble model (RF + Federated decision fusion)

The Standard and Federated paths are independent. If federated artifacts are missing, Standard still works.

## 1) Architecture at a Glance

- Standard path:
  - Uses existing artifacts in `ml/`.
  - Executed via `run_standard_pipeline(...)`.
  - No dependency on federated files.

- Federated path:
  - Trained by `python -m federated.run_simulation`.
  - Artifacts are saved to `federated/results/`.
  - Loaded by `FederatedAdvisor` in `federated/inference.py`.

- Ensemble path:
  - Implemented in `meta_learner/inference.py`.
  - Combines RF and Federated outputs.
  - Uses decision logic (agreement/weighted/gating) to produce final output.

## 2) Federated Artifacts Created by Simulation

After running simulation, these files are produced in `federated/results/`:

- `federated_advisor_final.pth` (final federated model weights)
- `federated_scaler.pkl` (fitted StandardScaler)
- `federated_label_encoder.pkl` (fitted LabelEncoder)
- `federated_model_metadata.json` (accuracy and model metadata)
- `comparison.json` (centralized vs federated comparison)

Command:

```bash
python -m federated.run_simulation --data_path "dataset before sowing/crop_dataset_rebuilt.csv" --rounds 20
```

## 3) Backend Execution

Start backend from repository root (important):

```powershell
$env:PYTHONPATH = "backend"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

API base:

- `http://127.0.0.1:8000/api/v1`

## 4) Model Routing in /advisor/predict

Request field:

- `model_mode` (optional)
  - `standard` (default)
  - `federated`

Behavior:

- If `model_mode` is missing or `standard`, route to existing RF pipeline.
- If `model_mode` is `federated`, route to `FederatedAdvisor`.
- If federated artifacts are unavailable, API returns `503` with a clear action message.

## 5) API Examples

### 5.1 Standard prediction

```bash
curl -X POST http://127.0.0.1:8000/api/v1/advisor/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"ph":6.5,"temperature":20,"humidity":82,"rainfall":202,"soil_type":"loamy","season":"kharif","state_name":"karnataka","district_name":"mysore","model_mode":"standard"}'
```

### 5.2 Federated prediction

```bash
curl -X POST http://127.0.0.1:8000/api/v1/advisor/predict \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"ph":6.5,"temperature":20,"humidity":82,"rainfall":202,"soil_type":"loamy","season":"kharif","state_name":"karnataka","district_name":"mysore","model_mode":"federated"}'
```

### 5.3 Side-by-side compare endpoint

```bash
curl -X POST http://127.0.0.1:8000/api/v1/advisor/compare \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"ph":6.5,"temperature":20,"humidity":82,"rainfall":202,"soil_type":"loamy","season":"kharif","state_name":"karnataka","district_name":"mysore"}'
```

## 6) Ensemble Execution (Direct Python)

If you want direct ensemble inference (outside API routing), run:

```python
from meta_learner.inference import EnsembleAdvisor

ens = EnsembleAdvisor()
result = ens.predict(90, 42, 43, 6.5, 20, 82, 202)
print(result)
```

## 7) Real Example with Confidence and Accuracy

Input used:

- N=90, P=42, K=43, pH=6.5, temperature=20, humidity=82, rainfall=202

Observed predictions:

| Model | Predicted Crop | Confidence | Accuracy Metric Source |
|---|---|---:|---|
| Standard (RF) | rice | 0.9250 (92.5%) | Pipeline confidence (`predict_proba`) |
| Federated (AdvisorNet) | rice | 0.9750 (97.5%) | Softmax top-1 confidence |
| Ensemble | rice | 0.9750 (97.5%) | Ensemble final confidence |

Observed metadata accuracy values from federated response:

- `federated_model_accuracy`: 0.970349
- `centralized_model_accuracy`: 0.9671
- `accuracy_gap_pct`: -0.336

## 8) Failure-Safe Behavior

- Standard path does not depend on federated artifacts.
- If federated files are missing or invalid:
  - `/advisor/predict` with `model_mode="federated"` returns `503`.
  - `/advisor/predict` with `model_mode="standard"` continues to work.

## 9) Quick Validation Checklist

- [ ] Standard request returns success without federated files.
- [ ] Federated request returns success after simulation artifacts are generated.
- [ ] Federated request returns `503` with action hint when artifacts are missing.
- [ ] Compare endpoint returns both standard and federated sections.
- [ ] Ensemble inference works via `meta_learner/inference.py`.
