# Backend (FastAPI)

Entry points:

- `main.py`: FastAPI app (`/predict`, `/health`, `/validation/summary`)
- `inference.py`: ONNX + mock inference, Evidence Score, attention/hotspots for interpretability
- `cache.py`: caching layer used by `/predict`

Validation (benchmark-style, demo-oriented):

- `run_validation.py`: CLI to generate validation outputs for the UI
- `validation/runner.py`: orchestrates dataset parsing → split → inference → metrics → JSON outputs
- `validation/db55_xlsx.py`: downloads/parses DB5.5 XLSX into PDB pairs
- `validation/split.py`: leakage-safe deterministic group split (by receptor)
- `validation/metrics.py`: metrics + bootstrap CIs

Outputs:

- `validation/outputs/validation_summary.json`: served by `/validation/summary` for the frontend
- `validation/outputs/validation_results.json`: fuller artifact for debugging

Notes:

- The validation pipeline only needs probabilities; it uses `inference.predict_interaction_probability()` to avoid heavy attention outputs.
- The UI `/predict` path uses full inference (including hotspots + Evidence Score).
