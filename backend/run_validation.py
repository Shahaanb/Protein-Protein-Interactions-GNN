from __future__ import annotations

import argparse
import os

from validation.runner import run_validation


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DB5.5 lightweight evaluation and write validation_summary.json")
    ap.add_argument("--max-positives", type=int, default=40, help="Max DB5.5 positive pairs to score")
    ap.add_argument("--negatives-per-positive", type=int, default=1, help="Synthetic negatives per positive")
    ap.add_argument("--n-boot", type=int, default=200, help="Bootstrap resamples for CI")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed")
    args = ap.parse_args()

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    workdir = os.path.join(backend_dir, "validation")

    out = run_validation(
        workdir=workdir,
        max_positives=int(args.max_positives),
        negatives_per_positive=int(args.negatives_per_positive),
        seed=int(args.seed),
        n_boot=int(args.n_boot),
    )

    outputs_dir = os.path.join(workdir, "outputs")
    print(f"Wrote: {os.path.join(outputs_dir, 'validation_results.json')}")
    print(f"Wrote: {os.path.join(outputs_dir, 'validation_summary.json')}")
    print("Test metrics:")
    for k, v in out.get("test_metrics", {}).items():
        if k in {"tp", "tn", "fp", "fn"}:
            continue
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
