"""Command-line entry point for applying the method to observed directions."""

import argparse
import json
from pathlib import Path

import numpy as np

from .core import DirectionalLaw, compress_moments, covariance_subspace, information


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress a directional law and evaluate a fixed subspace."
    )
    parser.add_argument("directions", type=Path, help="CSV: one unit direction per row")
    parser.add_argument("--weights", type=Path, help="CSV: one nonnegative weight per row")
    parser.add_argument("--rank", type=int, required=True, help="measurement subspace rank")
    parser.add_argument("--order", type=int, required=True, help="even-moment matching order")
    parser.add_argument("--snr", type=float, required=True, help="linear SNR, not decibels")
    parser.add_argument("--seed", type=int, default=0, help="linear-program seed")
    parser.add_argument("--output", type=Path, default=Path("compressed_law.npz"))
    args = parser.parse_args()

    directions = np.loadtxt(args.directions, delimiter=",", ndmin=2)
    weights = (np.ones(len(directions)) if args.weights is None else
               np.loadtxt(args.weights, delimiter=",").reshape(-1))
    law = DirectionalLaw(directions, weights)
    compressed = compress_moments(law, args.order, seed=args.seed)
    basis, eigenvalues = covariance_subspace(law, args.rank)
    original_value = information(law, basis, args.snr)
    proxy_value = information(compressed.law, basis, args.snr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        directions=compressed.law.directions,
        weights=compressed.law.weights,
        original_indices=compressed.indices,
        covariance_basis=basis,
    )
    print(json.dumps({
        "original_directions": len(law.weights),
        "retained_directions": len(compressed.law.weights),
        "dimension": law.directions.shape[1],
        "rank": args.rank,
        "moment_order": args.order,
        "feature_count": compressed.feature_count,
        "max_moment_residual": compressed.max_moment_residual,
        "covariance_eigenvalues": eigenvalues.tolist(),
        "covariance_design_information_nats": original_value,
        "compressed_law_information_nats": proxy_value,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
