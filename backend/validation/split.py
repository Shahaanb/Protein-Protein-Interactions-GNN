from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple


def _stable_hash(s: str) -> int:
    d = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big", signed=False)


def group_split(
    *,
    items: Sequence[dict],
    group_key: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: str = "split_v1",
) -> Dict[str, List[dict]]:
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1")

    groups: Dict[str, List[dict]] = {}
    for it in items:
        g = str(it.get(group_key, ""))
        groups.setdefault(g, []).append(it)

    group_ids = sorted(groups.keys())
    # Deterministic shuffle via hash.
    group_ids.sort(key=lambda g: _stable_hash(seed + ":" + g))

    n = len(group_ids)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train_groups = set(group_ids[:n_train])
    val_groups = set(group_ids[n_train : n_train + n_val])
    test_groups = set(group_ids[n_train + n_val :])

    out = {"train": [], "val": [], "test": []}
    for g, its in groups.items():
        if g in train_groups:
            out["train"].extend(its)
        elif g in val_groups:
            out["val"].extend(its)
        else:
            out["test"].extend(its)

    return out
