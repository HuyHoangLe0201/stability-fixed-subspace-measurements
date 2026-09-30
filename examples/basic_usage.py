"""Apply the method to a new directional ensemble.

Replace the generated directions with rows from your own application.
"""

import numpy as np

from fixed_subspace import (
    DirectionalLaw,
    candidate_transfer,
    compress_moments,
    covariance_subspace,
    information,
)


def main() -> None:
    rng = np.random.default_rng(7)
    directions = rng.normal(size=(80, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    law = DirectionalLaw.empirical(directions)

    covariance_basis, _ = covariance_subspace(law, rank=1)
    proxy = compress_moments(law, order=2, seed=7)
    gamma = 1.0  # linear signal-to-noise ratio
    print("Covariance design, original law:", information(law, covariance_basis, gamma))
    print("Covariance design, compressed law:", information(proxy.law, covariance_basis, gamma))
    print("Retained directions:", len(proxy.law.weights), "of", len(law.weights))
    print("Largest matched-moment residual:", proxy.max_moment_residual)

    candidates = [covariance_basis]
    for _ in range(30):
        q, _ = np.linalg.qr(rng.normal(size=(3, 1)))
        candidates.append(q)
    transfer = candidate_transfer(law, proxy.law, np.stack(candidates), gamma)
    print("Regret within this finite candidate set:", transfer.candidate_regret)
    print("Upper bound from twice its uniform candidate error:",
          2 * transfer.uniform_candidate_error)


if __name__ == "__main__":
    main()
