import os
import hashlib
import numpy as np
import logging
from functools import lru_cache
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "proteome_x_gat.onnx")
RCSB_URL = "https://files.rcsb.org/download/{pdb}.pdb"
DISTANCE_THRESHOLD = float(os.getenv("GRAPH_DISTANCE_THRESHOLD", "8.0"))
FEATURE_DIM = int(os.getenv("GRAPH_FEATURE_DIM", "64"))
EVIDENCE_RUNS = int(os.getenv("EVIDENCE_RUNS", "3"))
EVIDENCE_FEATURE_NOISE_STD = float(os.getenv("EVIDENCE_FEATURE_NOISE_STD", "0.005"))
EVIDENCE_PROB_STD_MAX = float(os.getenv("EVIDENCE_PROB_STD_MAX", "8.0"))  # percent points
EVIDENCE_MODE_FACTOR_ONNX = float(os.getenv("EVIDENCE_MODE_FACTOR_ONNX", "1.0"))
EVIDENCE_MODE_FACTOR_MOCK = float(os.getenv("EVIDENCE_MODE_FACTOR_MOCK", "0.15"))
ORT_LOG_SEVERITY_LEVEL = int(os.getenv("ORT_LOG_SEVERITY_LEVEL", "3"))  # 0=verbose,1=info,2=warning,3=error,4=fatal

def load_onnx_session():
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        try:
            so.log_severity_level = int(ORT_LOG_SEVERITY_LEVEL)
        except Exception:
            # If onnxruntime build doesn't expose this option, proceed without it.
            pass
        session = ort.InferenceSession(MODEL_PATH, sess_options=so)
        logger.info(f"Loaded ONNX model from {MODEL_PATH}")
        return session
    except Exception as e:
        logger.warning(f"Could not load ONNX model ({e}). Using mock inference.")
        return None

# We can initialize it lazily or globally
_session = None


def _seed_from_pair(pdb_pair: str) -> int:
    digest = hashlib.sha256(pdb_pair.encode("utf-8")).digest()
    # Use 8 bytes for a stable 64-bit seed.
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@lru_cache(maxsize=256)
def _fetch_pdb_text(pdb_id: str) -> str:
    pdb_id = str(pdb_id).upper()
    url = RCSB_URL.format(pdb=pdb_id)
    try:
        with urlopen(url, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except (URLError, HTTPError) as e:
        raise ValueError(f"Could not fetch PDB {pdb_id}: {e}")


@lru_cache(maxsize=512)
def _get_ca_atoms(pdb_id: str):
    pdb_id = str(pdb_id).upper()
    pdb_text = _fetch_pdb_text(pdb_id)
    return _parse_ca_atoms(pdb_text, pdb_id)


def _parse_ca_atoms(pdb_text: str, pdb_id: str):
    coords = []
    residues = []

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        alt_loc = line[16:17].strip()
        if atom_name != "CA":
            continue
        if alt_loc not in ("", "A"):
            continue
        try:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            resi = int(line[22:26].strip())
        except ValueError:
            continue

        coords.append([x, y, z])
        residues.append(resi)

    if not coords:
        raise ValueError(f"No C-alpha atoms found for PDB {pdb_id}")

    return np.asarray(coords, dtype=np.float32), residues


def _build_node_features(coords: np.ndarray, protein_flag: np.ndarray, feature_dim: int) -> np.ndarray:
    # Lightweight deterministic features from coordinates + protein identity.
    centered = coords - np.mean(coords, axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1, keepdims=True)
    scaled = centered / (np.std(centered, axis=0, keepdims=True) + 1e-6)
    sin_part = np.sin(centered / 10.0)
    cos_part = np.cos(centered / 10.0)
    flag_col = protein_flag.reshape(-1, 1)
    one_hot = np.stack([1.0 - protein_flag, protein_flag], axis=1)

    base = np.concatenate([scaled, radius, sin_part, cos_part, flag_col, one_hot], axis=1).astype(np.float32)
    if base.shape[1] >= feature_dim:
        return base[:, :feature_dim]

    repeat_count = int(np.ceil(feature_dim / base.shape[1]))
    tiled = np.tile(base, (1, repeat_count))
    return tiled[:, :feature_dim].astype(np.float32)


def _build_edge_index(coords: np.ndarray, threshold: float) -> np.ndarray:
    diffs = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    mask = dist < threshold
    np.fill_diagonal(mask, False)

    src, dst = np.nonzero(mask)
    if src.size == 0:
        # Fallback: connect each node to its next node so the graph is minimally connected.
        n = coords.shape[0]
        if n > 1:
            src = np.arange(0, n - 1, dtype=np.int64)
            dst = np.arange(1, n, dtype=np.int64)
            src = np.concatenate([src, dst])
            dst = np.concatenate([dst, src[: n - 1]])
        else:
            src = np.array([0], dtype=np.int64)
            dst = np.array([0], dtype=np.int64)

    return np.vstack([src.astype(np.int64), dst.astype(np.int64)])


def _build_pair_graph_inputs(pdb1: str, pdb2: str):
    c1, r1 = _get_ca_atoms(pdb1)
    c2, r2 = _get_ca_atoms(pdb2)

    # Center each protein to reduce absolute-coordinate bias from PDB reference frames.
    c1 = c1 - np.mean(c1, axis=0, keepdims=True)
    c2 = c2 - np.mean(c2, axis=0, keepdims=True)

    coords = np.vstack([c1, c2]).astype(np.float32)
    protein_flag = np.concatenate(
        [np.zeros(c1.shape[0], dtype=np.float32), np.ones(c2.shape[0], dtype=np.float32)]
    )

    x = _build_node_features(coords, protein_flag, FEATURE_DIM).astype(np.float32)
    edge_index = _build_edge_index(coords, DISTANCE_THRESHOLD).astype(np.int64)
    batch = np.zeros((coords.shape[0],), dtype=np.int64)

    residue_numbers = r1 + r2
    return x, edge_index, batch, residue_numbers


def _select_node_attention(outputs, num_nodes: int) -> np.ndarray:
    # The exported model may return attention in different shapes (e.g. [N], [N, heads], [N, 1]).
    # Prefer per-node scalar outputs over multi-head intermediates.
    candidates = []
    for idx, out in enumerate(outputs[1:], start=1):
        arr = np.asarray(out)
        if arr.size == 0:
            continue
        if arr.ndim >= 1 and arr.shape[0] == num_nodes:
            per_node = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
            # rank: 0 = already 1D, 1 = scalar-per-node via [N,1], 2+ = larger per-node tensors
            if arr.ndim == 1:
                shape_rank = 0
            elif per_node == 1:
                shape_rank = 1
            else:
                shape_rank = 2
            candidates.append((shape_rank, per_node, idx, arr))

    if not candidates:
        return np.ones((num_nodes,), dtype=np.float32)

    # Prefer lower shape_rank, then lower per_node. For ties, prefer later outputs (higher idx).
    candidates.sort(key=lambda t: (t[0], t[1], -t[2]))
    arr = candidates[0][3]
    if arr.ndim == 1:
        return arr.astype(np.float32)
    reduce_axes = tuple(range(1, arr.ndim))
    return np.mean(arr.astype(np.float32), axis=reduce_axes)


def _onnx_predict_from_inputs(x: np.ndarray, edge_index: np.ndarray, batch: np.ndarray):
    input_names = [i.name for i in _session.get_inputs()]
    feed = {}

    if "x" in input_names:
        feed["x"] = x
    if "edge_index" in input_names:
        feed["edge_index"] = edge_index
    if "batch" in input_names:
        feed["batch"] = batch

    if len(feed) != len(input_names):
        # Fallback to positional mapping for non-standard input names.
        ordered_values = [x, edge_index, batch]
        feed = {name: ordered_values[idx] for idx, name in enumerate(input_names)}

    outputs = _session.run(None, feed)
    prob_arr = np.asarray(outputs[0], dtype=np.float32)
    raw_prob = float(prob_arr.reshape(-1)[0])

    if 0.0 <= raw_prob <= 1.0:
        prob = raw_prob
    else:
        prob = 1.0 / (1.0 + np.exp(-raw_prob))

    attention_arr = _select_node_attention(outputs, int(x.shape[0]))
    return float(prob), attention_arr


def _onnx_predict_probability_from_inputs(x: np.ndarray, edge_index: np.ndarray, batch: np.ndarray) -> float:
    """Runs ONNX but fetches ONLY the first output (prob/logit).

    This avoids requesting every intermediate output that may be present in the exported model,
    which can produce extremely noisy ORT warnings and slow evaluation runs.
    """
    input_names = [i.name for i in _session.get_inputs()]
    feed = {}

    if "x" in input_names:
        feed["x"] = x
    if "edge_index" in input_names:
        feed["edge_index"] = edge_index
    if "batch" in input_names:
        feed["batch"] = batch

    if len(feed) != len(input_names):
        ordered_values = [x, edge_index, batch]
        feed = {name: ordered_values[idx] for idx, name in enumerate(input_names)}

    output_name = _session.get_outputs()[0].name
    outputs = _session.run([output_name], feed)
    prob_arr = np.asarray(outputs[0], dtype=np.float32)
    raw_prob = float(prob_arr.reshape(-1)[0])

    if 0.0 <= raw_prob <= 1.0:
        return float(raw_prob)
    return float(1.0 / (1.0 + np.exp(-raw_prob)))


def _top_hotspots(attention_arr: np.ndarray, residue_numbers, top_k: int = 5):
    if attention_arr.shape[0] != len(residue_numbers):
        attention_arr = np.resize(attention_arr, (len(residue_numbers),))

    k = min(top_k, len(residue_numbers))
    top_idx = np.argsort(-attention_arr)[:k]
    hotspots = [
        {"node_idx": int(residue_numbers[int(idx)]), "attention": float(attention_arr[int(idx)])}
        for idx in top_idx
    ]
    hotspot_set = {int(residue_numbers[int(idx)]) for idx in top_idx}
    return hotspots, hotspot_set


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b)) / float(len(union))


def _input_quality_score(num_nodes: int, num_edges: int) -> float:
    # Heuristic sanity checks; intended to detect obviously broken graphs.
    #  - nodes: expect ~100+ CA atoms for many proteins
    #  - edges: expect some connectivity (distance graph)
    if num_nodes <= 0:
        return 0.0

    node_score = min(1.0, float(num_nodes) / 100.0)
    # edge_index contains directed edges; treat edge count as columns.
    edge_target = max(1.0, float(num_nodes) * 4.0)
    edge_score = min(1.0, float(num_edges) / edge_target)
    return max(0.0, min(1.0, 0.5 * node_score + 0.5 * edge_score))


def _compute_evidence_score(
    *,
    pdb_pair: str,
    inference_mode: str,
    prob_runs_percent,
    hotspot_sets,
    input_quality: float,
    runs: int,
    feature_noise_std: float,
):
    # Stability: lower std dev => higher score.
    prob_std = float(np.std(prob_runs_percent)) if prob_runs_percent else 0.0
    s_prob = 1.0 - min(1.0, prob_std / max(EVIDENCE_PROB_STD_MAX, 1e-6))

    # Consistency: mean pairwise Jaccard over hotspot sets.
    if len(hotspot_sets) >= 2:
        jac = []
        for i in range(len(hotspot_sets)):
            for j in range(i + 1, len(hotspot_sets)):
                jac.append(_jaccard(hotspot_sets[i], hotspot_sets[j]))
        s_hot = float(np.mean(jac)) if jac else 1.0
    else:
        s_hot = 1.0

    s_input = float(max(0.0, min(1.0, input_quality)))

    weights = {
        "probability_stability": 0.35,
        "hotspot_consistency": 0.35,
        "input_quality": 0.30,
    }
    base = (
        weights["probability_stability"] * s_prob
        + weights["hotspot_consistency"] * s_hot
        + weights["input_quality"] * s_input
    )

    if inference_mode == "onnx":
        mode_factor = EVIDENCE_MODE_FACTOR_ONNX
    else:
        mode_factor = EVIDENCE_MODE_FACTOR_MOCK

    evidence = 100.0 * max(0.0, min(1.0, base)) * max(0.0, min(1.0, mode_factor))

    breakdown = {
        "metric_version": "evidence_v1",
        "pdb_pair": pdb_pair,
        "inference_mode": inference_mode,
        "runs": int(runs),
        "feature_noise_std": float(feature_noise_std),
        "prob_std_max_percent_points": float(EVIDENCE_PROB_STD_MAX),
        "weights": weights,
        "probability_std_percent_points": round(prob_std, 4),
        "probability_stability_score": round(float(s_prob), 4),
        "hotspot_consistency_score": round(float(s_hot), 4),
        "input_quality_score": round(float(s_input), 4),
        "mode_factor": round(float(mode_factor), 4),
        "notes": (
            "Evidence Score measures robustness + input sanity; it is NOT the model's self-confidence."
            if inference_mode == "onnx"
            else "Mock mode detected; Evidence Score is heavily down-weighted and should not be treated as validation."
        ),
    }

    return round(float(evidence), 2), breakdown


def _run_onnx_inference(pdb1: str, pdb2: str, pdb_pair: str):
    x, edge_index, batch, residue_numbers = _build_pair_graph_inputs(pdb1, pdb2)

    # Base prediction
    prob, attention_arr = _onnx_predict_from_inputs(x, edge_index, batch)
    hotspots_dict, hotspot_set = _top_hotspots(attention_arr, residue_numbers, top_k=5)

    # Evidence runs (stability under small feature perturbations)
    runs = max(1, int(EVIDENCE_RUNS))
    prob_runs_percent = [float(prob) * 100.0]
    hotspot_sets = [hotspot_set]

    if runs > 1:
        seed = _seed_from_pair(pdb_pair + ":evidence")
        rng = np.random.default_rng(seed)
        for _ in range(runs - 1):
            noise = rng.normal(0.0, float(EVIDENCE_FEATURE_NOISE_STD), size=x.shape).astype(np.float32)
            p_i, att_i = _onnx_predict_from_inputs(x + noise, edge_index, batch)
            prob_runs_percent.append(float(p_i) * 100.0)
            _, hs_set = _top_hotspots(att_i, residue_numbers, top_k=5)
            hotspot_sets.append(hs_set)

    input_quality = _input_quality_score(int(x.shape[0]), int(edge_index.shape[1]))
    evidence_score, evidence_breakdown = _compute_evidence_score(
        pdb_pair=pdb_pair,
        inference_mode="onnx",
        prob_runs_percent=prob_runs_percent,
        hotspot_sets=hotspot_sets,
        input_quality=input_quality,
        runs=runs,
        feature_noise_std=float(EVIDENCE_FEATURE_NOISE_STD),
    )

    return {
        "pdb_pair": pdb_pair,
        "pdb1": pdb1,
        "pdb2": pdb2,
        "interaction_probability": round(float(prob) * 100, 2),
        "hotspots": hotspots_dict,
        "inference_mode": "onnx",
        "evidence_score": evidence_score,
        "evidence_breakdown": evidence_breakdown,
    }


def _run_mock_inference(pdb1: str, pdb2: str, pdb_pair: str):
    seed = _seed_from_pair(pdb_pair)
    rng = np.random.default_rng(seed)

    # Generating deterministic interpretable outputs for the UI
    prob = float(rng.uniform(0.5, 0.9))  # range 50-90% interaction

    # Generate 5 realistic hotspot indices
    # Proteins generally have between 100-300 amino acids
    hotspots = rng.choice(np.arange(10, 150), 5, replace=False).tolist()
    scores = rng.random(5).tolist()

    hotspot_pairs = sorted(zip(hotspots, scores), key=lambda pair: pair[1], reverse=True)
    hotspots_dict = [{"node_idx": int(idx), "attention": float(score)} for idx, score in hotspot_pairs]

    evidence_score, evidence_breakdown = _compute_evidence_score(
        pdb_pair=pdb_pair,
        inference_mode="mock",
        prob_runs_percent=[round(prob * 100.0, 2)],
        hotspot_sets=[{int(h["node_idx"]) for h in hotspots_dict}],
        input_quality=0.0,
        runs=1,
        feature_noise_std=0.0,
    )

    return {
        "pdb_pair": pdb_pair,
        "pdb1": pdb1,
        "pdb2": pdb2,
        "interaction_probability": round(prob * 100, 2),  # percentage
        "hotspots": hotspots_dict,
        "inference_mode": "mock",
        "evidence_score": evidence_score,
        "evidence_breakdown": evidence_breakdown,
    }


def predict_interaction_probability(pdb1: str, pdb2: str) -> dict:
    """Lightweight prediction API for evaluation.

    Returns only `interaction_probability` (+ minimal metadata) and does not compute attention/hotspots.
    """
    global _session
    if _session is None and os.path.exists(MODEL_PATH):
        _session = load_onnx_session()

    pdb_pair = f"{pdb1.upper()}_{pdb2.upper()}"

    if _session:
        try:
            x, edge_index, batch, _residue_numbers = _build_pair_graph_inputs(pdb1, pdb2)
            prob = _onnx_predict_probability_from_inputs(x, edge_index, batch)
            return {
                "pdb_pair": pdb_pair,
                "pdb1": pdb1,
                "pdb2": pdb2,
                "interaction_probability": round(float(prob) * 100, 2),
                "inference_mode": "onnx",
            }
        except Exception as e:
            logger.warning(f"ONNX probability-only inference failed ({e}). Falling back to mock.")

    # Mock fallback, deterministic.
    seed = _seed_from_pair(pdb_pair)
    rng = np.random.default_rng(seed)
    prob = float(rng.uniform(0.5, 0.9))
    return {
        "pdb_pair": pdb_pair,
        "pdb1": pdb1,
        "pdb2": pdb2,
        "interaction_probability": round(float(prob) * 100, 2),
        "inference_mode": "mock",
    }

def run_inference(pdb1: str, pdb2: str):
    """
    Executes model inference. Since establishing the full pipeline without bio-dependencies 
    in the FastAPI container is complex, we mock the results gracefully if ONNX is missing.
    In a real scenario, we'll convert PDB pairs to Graphs and run ort_session.run().
    """
    global _session
    if _session is None and os.path.exists(MODEL_PATH):
        _session = load_onnx_session()

    pdb_pair = f"{pdb1.upper()}_{pdb2.upper()}"

    if _session:
        try:
            return _run_onnx_inference(pdb1, pdb2, pdb_pair)
        except Exception as e:
            logger.warning(f"ONNX inference failed ({e}). Falling back to mock inference.")

    return _run_mock_inference(pdb1, pdb2, pdb_pair)
