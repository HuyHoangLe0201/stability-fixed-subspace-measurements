"""Regenerate two representative numerical checks without bundled paper files."""

import json
from pathlib import Path

import numpy as np

from fixed_subspace import DirectionalLaw, candidate_transfer, compress_moments, covariance_subspace, information


def unit_vectors(angles: np.ndarray) -> np.ndarray:
    return np.column_stack((np.cos(angles), np.sin(angles)))


def rank_one_grid(size: int = 4096) -> np.ndarray:
    angles = np.linspace(0, np.pi, size, endpoint=False)
    return unit_vectors(angles)[:, :, None]


def main() -> None:
    # Finite-SNR information design can move away from the covariance design.
    four = DirectionalLaw(unit_vectors(np.array([0.0, 0.6, 1.3, 2.2])),
                          np.array([0.45, 0.25, 0.20, 0.10]))
    q0, spectrum = covariance_subspace(four, rank=1)
    grid = rank_one_grid()
    finite_values = np.array([information(four, q, 3.0) for q in grid])
    selected = grid[int(np.argmax(finite_values))]
    covariance_regret = float(np.max(finite_values) - information(four, q0, 3.0))

    # Positive even-moment compression transfers a finite candidate design.
    angles = np.pi * np.arange(41) / 41
    weights = np.exp(0.7 * np.cos(2 * angles)
                     + 0.8 * np.sin(4 * angles)
                     + 0.3 * np.cos(6 * angles))
    law = DirectionalLaw(unit_vectors(angles), weights)
    proxy = compress_moments(law, order=4, seed=20260919)
    transfer = candidate_transfer(law, proxy.law, grid, snr=1.0)
    assert transfer.candidate_regret <= 2 * transfer.uniform_candidate_error + 1e-12

    report = {
        "four_direction": {
            "covariance_eigengap": float(spectrum[0] - spectrum[1]),
            "finite_snr": 3.0,
            "projector_distance": float(np.sqrt(2) * np.sqrt(
                max(0.0, 1.0 - float((q0.T @ selected).item()) ** 2))),
            "covariance_design_regret_nats_on_grid": covariance_regret,
        },
        "moment_compression": {
            "original_directions": len(law.weights),
            "retained_directions": len(proxy.law.weights),
            "matching_order": proxy.order,
            "max_moment_residual": proxy.max_moment_residual,
            "candidate_information_error_nats": transfer.uniform_candidate_error,
            "candidate_transfer_regret_nats": transfer.candidate_regret,
            "candidate_count": len(grid),
        },
        "scope": "Grid and finite-candidate checks; no global optimizer certificate",
    }
    output = Path("outputs/numerical_checks.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
