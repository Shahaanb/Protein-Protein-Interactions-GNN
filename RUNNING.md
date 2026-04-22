# Running Proteome-X (Windows-first)

This repo contains a working Python environment at `backend/venv312` (Python 3.12).
If you accidentally created other `venv*` folders, you can delete them — the docs/UI assume `venv312`.

## 1) Start the backend (FastAPI)

Open a terminal in `proteome-x/` and run:

```powershell
cd backend
.\venv312\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Quick checks:

- Health: http://127.0.0.1:8000/health
- Validation summary: http://127.0.0.1:8000/validation/summary

## 2) Start the frontend (Vite)

Open a second terminal in `proteome-x/` and run:

```powershell
cd frontend
npm install
npm run dev
```

Frontend: http://127.0.0.1:5173

Optional: if your backend is not on port 8000, set `VITE_API_URL` before starting the frontend.

```powershell
$env:VITE_API_URL = "http://127.0.0.1:8000"
npm run dev
```

## 3) Generate / refresh benchmark metrics (Model Scores)

This creates/updates:
- `backend/validation/outputs/validation_results.json`
- `backend/validation/outputs/validation_summary.json`

Run:

```powershell
cd backend
.\venv312\Scripts\python.exe .\run_validation.py --max-positives 40 --n-boot 200
```

Then open the app and click the **Model Scores** side button → **Refresh**.

## Troubleshooting

### “Not Found” in Model Scores

That typically means you’re talking to an older/stale backend process.
Confirm you get JSON at:

- http://127.0.0.1:8000/validation/summary

If you don’t, stop whatever is running on port 8000 and restart the backend from `proteome-x/backend`.

### Port already in use

If `uvicorn` says port 8000 is in use, something else is running there.
Stop the old process (or change ports and set `VITE_API_URL`).
