# Proteome-X: Protein-Protein Interaction GNN Demo

Proteome-X is a full-stack demo that predicts whether two proteins interact. It pairs a graph neural network (exported to ONNX) with a FastAPI backend and a React + Vite frontend that visualizes predictions, hotspots, and validation metrics.

## What this demo shows

- **Interaction Probability**: model output mapped to a probability-like percentage (not guaranteed calibrated).
- **Evidence Score**: robustness + sanity heuristic with a breakdown (explicitly not model self-confidence).
- **Hotspots**: top residues by attention-like signal for interpretability.
- **Model Scores**: benchmark-style metrics with 95% bootstrap CIs.

For the full definition of metrics and outputs, see [MODEL_PARAMETERS_AND_METRICS_REPORT.txt](MODEL_PARAMETERS_AND_METRICS_REPORT.txt).

## Tech stack

- Model: GNN (PyTorch Geometric) exported to ONNX
- Backend: FastAPI + ONNX Runtime + Redis cache
- Frontend: React + Vite + Tailwind
- Validation: DB5.5 lightweight evaluation pipeline

## Quick start (Windows)

Start the backend:

```powershell
cd backend
.\venv312\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend (new PowerShell window):

```powershell
cd frontend
npm install
npm run dev
```

Open:
- Frontend: http://127.0.0.1:5173
- Backend health: http://127.0.0.1:8000/health

For more details and troubleshooting, see [RUNNING.md](RUNNING.md).

## Docker compose

```bash
docker compose up --build
```

Services:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000

## API (FastAPI)

### POST /predict

Request:

```json
{
  "pdb1": "1ABC",
  "pdb2": "2XYZ"
}
```

Response (shape):

```json
{
  "source": "cache",
  "data": {
    "pdb_pair": "1ABC_2XYZ",
    "interaction_probability": 42.13,
    "evidence_score": 61.5,
    "hotspots": [
      {"node_idx": 54, "attention": 0.91}
    ]
  }
}
```

See full field definitions and computation details in [MODEL_PARAMETERS_AND_METRICS_REPORT.txt](MODEL_PARAMETERS_AND_METRICS_REPORT.txt).

## Validation and model scores

Generate or refresh the benchmark metrics used by the UI:

```powershell
cd backend
.\venv312\Scripts\python.exe .\run_validation.py --max-positives 40 --n-boot 200
```

Outputs:
- [backend/validation/outputs/validation_summary.json](backend/validation/outputs/validation_summary.json)
- [backend/validation/outputs/validation_results.json](backend/validation/outputs/validation_results.json)

## Project layout

- [backend/](backend/) FastAPI app, inference, caching, validation pipeline
- [frontend/](frontend/) React UI
- [model_pipeline/](model_pipeline/) model training + ONNX export utilities

## Caveats

- This is a demo-oriented evaluation, not a production-calibrated predictor.
- Interaction Probability is a model score mapped to a percentage; it may be uncalibrated.
- Evidence Score is a stability + sanity heuristic, not correctness.

## Sharing and deployment

See [SHARING.md](SHARING.md) for quick demo sharing and deployment options.
