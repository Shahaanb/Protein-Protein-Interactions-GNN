from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .db55_xlsx import DB55_XLSX_URL, Db55Row, download_db55_xlsx, extract_pairs_from_xlsx, write_pairs_csv
from .metrics import bootstrap_ci, compute_all_metrics, find_best_threshold_for_f1
from .split import group_split


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_examples(
    *,
    positives: List[Db55Row],
    negatives_per_positive: int,
    seed: int,
) -> List[dict]:
    # Positives: (pdb1, pdb2)
    examples: List[dict] = []
    for p in positives:
        examples.append({"pdb1": p.pdbid_1, "pdb2": p.pdbid_2, "label": 1, "group": p.pdbid_1})

    # Negatives: mismatch ligands across positives.
    rng = np.random.default_rng(seed)
    ligands = [p.pdbid_2 for p in positives]
    receptors = [p.pdbid_1 for p in positives]

    pos_set = {(p.pdbid_1, p.pdbid_2) for p in positives}

    for p in positives:
        added = 0
        attempts = 0
        while added < negatives_per_positive and attempts < 200:
            attempts += 1
            # Choose a ligand from another example.
            ligand = str(rng.choice(ligands))
            if ligand == p.pdbid_2:
                continue
            pair = (p.pdbid_1, ligand)
            if pair in pos_set:
                continue
            examples.append({"pdb1": p.pdbid_1, "pdb2": ligand, "label": 0, "group": p.pdbid_1})
            added += 1

    return examples


def _predict_prob(pdb1: str, pdb2: str) -> Tuple[float, Dict]:
    # Import locally so this module can be imported without backend deps.
    import inference

    if hasattr(inference, "predict_interaction_probability"):
        res = inference.predict_interaction_probability(pdb1, pdb2)
    else:
        res = inference.run_inference(pdb1, pdb2)
    prob_percent = float(res.get("interaction_probability", 0.0))
    prob = max(0.0, min(1.0, prob_percent / 100.0))
    return prob, res


def run_validation(
    *,
    workdir: str,
    max_positives: int = 40,
    negatives_per_positive: int = 1,
    seed: int = 1337,
    n_boot: int = 200,
) -> Dict:
    """Runs an evidence-based validation pass (B–D):

    B) leakage-safe split: group split by receptor (pdb1)
    C) metrics: accuracy/F1 + ROC-AUC/PR-AUC + calibration (Brier/ECE)
    D) uncertainty: bootstrap 95% CIs

    Notes:
    - This is not training; it evaluates the current model output as-is.
    - It uses DB5.5 positives + mismatched negatives.
    """

    datasets_dir = os.path.join(workdir, "datasets")
    outputs_dir = os.path.join(workdir, "outputs")
    _ensure_dir(datasets_dir)
    _ensure_dir(outputs_dir)

    xlsx_path = os.path.join(datasets_dir, "Table_BM5.5.xlsx")
    csv_path = os.path.join(datasets_dir, "db55_pairs.csv")

    if not os.path.exists(xlsx_path):
        download_db55_xlsx(xlsx_path, url=DB55_XLSX_URL)

    pairs = extract_pairs_from_xlsx(xlsx_path)
    write_pairs_csv(pairs, csv_path)

    if max_positives and len(pairs) > max_positives:
        pairs = pairs[: int(max_positives)]

    examples = _make_examples(positives=pairs, negatives_per_positive=int(negatives_per_positive), seed=seed)

    splits = group_split(items=examples, group_key="group", train_frac=0.70, val_frac=0.15, test_frac=0.15, seed="db55")

    # Predict probs for each split.
    timings = {"n": 0, "seconds_total": 0.0}

    def run_split(name: str) -> List[dict]:
        out = []
        t0 = time.time()
        for ex in splits[name]:
            p, raw = _predict_prob(ex["pdb1"], ex["pdb2"])
            out.append({**ex, "prob": float(p), "inference_mode": raw.get("inference_mode", ""), "raw": raw})
        timings["n"] += len(splits[name])
        timings["seconds_total"] += time.time() - t0
        return out

    pred_train = run_split("train")
    pred_val = run_split("val")
    pred_test = run_split("test")

    def mode_counts(items: List[dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for it in items:
            mode = str(it.get("inference_mode") or "unknown")
            counts[mode] = int(counts.get(mode, 0)) + 1
        return counts

    inference_modes = {
        "train": mode_counts(pred_train),
        "val": mode_counts(pred_val),
        "test": mode_counts(pred_test),
    }

    def to_arrays(items: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        y_true = np.asarray([int(it["label"]) for it in items], dtype=int)
        y_prob = np.asarray([float(it["prob"]) for it in items], dtype=float)
        return y_true, y_prob

    yv_true, yv_prob = to_arrays(pred_val)
    best = {"threshold": 0.5, "f1": 0.0}
    # Avoid overfitting/instability when validation is tiny.
    if len(pred_val) >= 20:
        best = find_best_threshold_for_f1(yv_true, yv_prob)
    threshold = float(best["threshold"]) if pred_val else 0.5

    yt_true, yt_prob = to_arrays(pred_test)
    metrics = compute_all_metrics(y_true=yt_true, y_prob=yt_prob, threshold=threshold)
    cis = bootstrap_ci(y_true=yt_true, y_prob=yt_prob, threshold=threshold, n_boot=int(n_boot), seed=seed)

    # Summaries
    def split_summary(items: List[dict]) -> Dict[str, int]:
        y = [int(i["label"]) for i in items]
        return {"n": len(y), "positives": int(sum(y)), "negatives": int(len(y) - sum(y))}

    out = {
        "dataset": {
            "name": "Protein-Protein Docking Benchmark 5.5",
            "source_url": "https://zlab.wenglab.org/benchmark/",
            "table_url": DB55_XLSX_URL,
            "local_xlsx": xlsx_path,
            "local_pairs_csv": csv_path,
            "max_positives": int(max_positives),
            "negatives_per_positive": int(negatives_per_positive),
            "note": "Pairs are derived by taking the first 4-char PDB code from DB5.5 'PDBid 1/2' columns; chain-level identities are ignored for demo simplicity.",
        },
        "split": {
            "strategy": "group_split_by_receptor",
            "group_key": "pdb1",
            "train": split_summary(pred_train),
            "val": split_summary(pred_val),
            "test": split_summary(pred_test),
        },
        "threshold_selection": {
            "strategy": "maximize_f1_on_val" if len(pred_val) >= 20 else "fixed_0.5_due_to_small_val",
            "threshold": float(threshold),
            "val_f1": float(best.get("f1", 0.0)),
        },
        "test_metrics": metrics,
        "test_metrics_ci95": {k: {"lo": float(v[0]), "hi": float(v[1])} for k, v in cis.items()},
        "inference": {
            "modes": inference_modes,
            "used_mock": bool(
                inference_modes.get("train", {}).get("mock", 0)
                or inference_modes.get("val", {}).get("mock", 0)
                or inference_modes.get("test", {}).get("mock", 0)
            ),
        },
        "runtime": {
            "examples_scored": int(timings["n"]),
            "seconds_total": float(round(timings["seconds_total"], 3)),
            "seconds_per_example": float(round(timings["seconds_total"] / max(1, timings["n"]), 3)),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Write output artifact
    out_path = os.path.join(outputs_dir, "validation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Save slim version that frontend can render quickly
    slim = {
        "generated_at": out["generated_at"],
        "dataset": {
            "name": out["dataset"]["name"],
            "source_url": out["dataset"]["source_url"],
            "table_url": out["dataset"]["table_url"],
            "max_positives": out["dataset"]["max_positives"],
            "negatives_per_positive": out["dataset"]["negatives_per_positive"],
        },
        "split": out["split"],
        "threshold_selection": out["threshold_selection"],
        "test_metrics": out["test_metrics"],
        "test_metrics_ci95": out["test_metrics_ci95"],
        "inference": out["inference"],
    }
    slim_path = os.path.join(outputs_dir, "validation_summary.json")
    with open(slim_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2)

    return out
