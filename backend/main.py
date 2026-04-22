from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cache
import inference
import logging
import json
import os

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Proteome-X GNN API", description="Interpretable GNN backend for Protein Interactions")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    pdb1: str
    pdb2: str

@app.post("/predict")
async def predict_interaction(req: PredictRequest):
    if not req.pdb1 or not req.pdb2:
        raise HTTPException(status_code=400, detail="Both PDB IDs must be provided.")
    
    # Check cache
    cache_key = f"predict:{req.pdb1.upper()}_{req.pdb2.upper()}"
    cached_result = cache.get_cache(cache_key)
    
    if cached_result:
        # If older cached entries exist (from before Evidence Score rollout), recompute once.
        if "evidence_score" in cached_result and "evidence_breakdown" in cached_result:
            logging.info(f"Cache hit for {cache_key}")
            return {"source": "cache", "data": cached_result}
        logging.info(f"Stale cache entry for {cache_key} (missing evidence fields). Recomputing.")
    
    logging.info(f"Computing inference for {cache_key}")
    # Run Inference
    try:
        result = inference.run_inference(req.pdb1, req.pdb2)
        # Store in cache
        cache.set_cache(cache_key, result, ttl_seconds=3600)
        return {"source": "compute", "data": result}
    except Exception as e:
        logging.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "name": "Proteome-X GNN API",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "predict": "/predict",
            "validation_summary": "/validation/summary",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/validation/summary")
async def validation_summary():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base_dir, "validation", "outputs", "validation_summary.json")

    if not os.path.exists(summary_path):
        raise HTTPException(
            status_code=404,
            detail=(
                "No validation summary found. Run backend/run_validation.py to generate "
                "backend/validation/outputs/validation_summary.json."
            ),
        )

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read validation summary: {e}")
