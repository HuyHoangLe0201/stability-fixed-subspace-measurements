"""Information-optimal fixed-subspace tools for discrete directional laws.

Directions are unit vectors in R^d. A d-by-K matrix with orthonormal columns
represents a rank-K measurement subspace. The receiver knows the direction.
All information values are in nats; the built-in scalar channel is real
Gaussian with unit-variance input and noise.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog


@dataclass(frozen=True)
class DirectionalLaw:
    """A finite probability law on unit directions; antipodes are equivalent.

    ``directions`` has shape (n, d); ``weights`` has shape (n,). Rows must
    already have unit length. Nonnegative weights are normalized to sum to one.
    """

    directions: NDArray[np.float64]
    weights: NDArray[np.float64]

    def __post_init__(self) -> None:
        u = np.asarray(self.directions, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        if u.ndim != 2 or u.shape[0] == 0 or u.shape[1] < 2:
            raise ValueError("directions must have shape (n, d), n > 0, d >= 2")
        if w.shape != (len(u),):
            raise ValueError("weights must have one entry per direction")
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(w)):
            raise ValueError("directions and weights must be finite")
        if np.max(np.abs(np.linalg.norm(u, axis=1) - 1.0)) > 1e-8:
            raise ValueError("every direction must have unit Euclidean norm")
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be nonnegative with positive total")
        object.__setattr__(self, "directions", u.copy())
        object.__setattr__(self, "weights", (w / w.sum()).copy())

    @classmethod
    def empirical(cls, directions: ArrayLike) -> "DirectionalLaw":
        """Assign equal weights to observed unit directions."""
        u = np.asarray(directions, dtype=float)
        if u.ndim != 2 or len(u) == 0:
            raise ValueError("directions must be a nonempty matrix")
        return cls(u, np.ones(len(u), dtype=float))


def gaussian_information(snr: ArrayLike) -> NDArray[np.float64]:
    """Real Gaussian scalar mutual information: 0.5 log(1 + snr) nats."""
    s = np.asarray(snr, dtype=float)
    if not np.all(np.isfinite(s)) or np.any(s < 0):
        raise ValueError("SNR values must be finite and nonnegative")
    return 0.5 * np.log1p(s)


def _basis(q: ArrayLike, dimension: int) -> NDArray[np.float64]:
    q = np.asarray(q, dtype=float)
    if (q.ndim != 2 or q.shape[0] != dimension
            or not 1 <= q.shape[1] < dimension):
        raise ValueError("basis must have shape (d, K), 1 <= K < d")
    if not np.all(np.isfinite(q)) or not np.allclose(
        q.T @ q, np.eye(q.shape[1]), atol=1e-8, rtol=0
    ):
        raise ValueError("basis columns must be orthonormal")
    return q


def information(
    law: DirectionalLaw,
    basis: ArrayLike,
    snr: float,
    scalar_information: Callable[[ArrayLike], ArrayLike] = gaussian_information,
) -> float:
    """Expected retained scalar information under a fixed measurement basis.

    Supply a vectorized scalar-information function ``h(snr)`` for another
    unit-variance amplitude prior. The direction is assumed known to the decoder.
    """
    q = _basis(basis, law.directions.shape[1])
    if not np.isfinite(snr) or snr < 0:
        raise ValueError("snr must be finite and nonnegative")
    coverage = np.sum((law.directions @ q) ** 2, axis=1)
    values = np.asarray(scalar_information(snr * coverage), dtype=float)
    if values.shape != (len(law.weights),) or not np.all(np.isfinite(values)):
        raise ValueError("scalar_information must return one finite value per direction")
    return float(law.weights @ values)


def covariance_subspace(
    law: DirectionalLaw, rank: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Leading covariance eigenspace and all eigenvalues in descending order.

    This subspace optimizes the leading low-SNR term. It need not optimize
    information at finite SNR.
    """
    d = law.directions.shape[1]
    if not isinstance(rank, int) or not 1 <= rank < d:
        raise ValueError("rank must be an integer between 1 and d - 1")
    second_moment = law.directions.T @ (law.weights[:, None] * law.directions)
    values, vectors = np.linalg.eigh(second_moment)
    order = np.argsort(values)[::-1]
    return vectors[:, order[:rank]], values[order]


@dataclass(frozen=True)
class CompressionResult:
    law: DirectionalLaw
    indices: NDArray[np.int64]
    order: int
    feature_count: int
    max_moment_residual: float


def compress_moments(
    law: DirectionalLaw,
    order: int,
    *,
    seed: int = 0,
    tolerance: float = 1e-8,
    max_features: int = 5000,
) -> CompressionResult:
    """Find a positive subset matching homogeneous degree-2*order moments.

    On the unit sphere, matching this degree also matches every lower even
    degree. The linear program finds a feasible vertex; it is not a minimum-
    support solver. A residual is returned because floating-point equality is
    approximate. The support bound is C(d+2r-1, 2r), subject to LP tolerance.
    """
    d = law.directions.shape[1]
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    feature_count = comb(d + 2 * order - 1, 2 * order)
    if feature_count > max_features:
        raise ValueError(
            f"{feature_count} moment features exceed max_features={max_features}"
        )
    n = len(law.weights)
    if n <= feature_count:
        return CompressionResult(law, np.arange(n), order, feature_count, 0.0)
    features = np.empty((feature_count + 1, n), dtype=float)
    features[0] = 1.0
    for row, powers in enumerate(combinations_with_replacement(range(d), 2 * order), 1):
        features[row] = np.prod(law.directions[:, powers], axis=1)
    target = features @ law.weights
    scale = np.maximum(np.linalg.norm(features, axis=1), 1e-15)
    rng = np.random.default_rng(seed)
    result = linprog(
        rng.normal(size=n), A_eq=features / scale[:, None], b_eq=target / scale,
        bounds=(0, None), method="highs-ds",
        options={"primal_feasibility_tolerance": 1e-9,
                 "dual_feasibility_tolerance": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"moment-matching linear program failed: {result.message}")
    indices = np.flatnonzero(result.x > 1e-10)
    weights = result.x[indices]
    # Remove the small equality residual of the LP basis when the refined
    # solution remains nonnegative. Keep the feasible LP weights otherwise.
    refined, *_ = np.linalg.lstsq(
        features[:, indices] / scale[:, None], target / scale, rcond=None
    )
    if np.min(refined) >= -1e-12:
        weights = np.maximum(refined, 0.0)
    proxy = DirectionalLaw(law.directions[indices], weights)
    residual = float(np.max(np.abs(features[:, indices] @ proxy.weights - target)))
    if residual > tolerance:
        raise RuntimeError(f"moment residual {residual:.3g} exceeds {tolerance:.3g}")
    return CompressionResult(proxy, indices, order, feature_count, residual)


def gaussian_matching_error_bound(snr: float, order: int) -> float:
    """Uniform information-error bound for *exact* even-moment matching.

    Check ``CompressionResult.max_moment_residual`` before using this as a
    numerical certificate; the formula has no floating-point residual term.
    """
    if not np.isfinite(snr) or snr < 0:
        raise ValueError("snr must be finite and nonnegative")
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    q = snr / (snr + 2)
    return float(min(snr ** (order + 1) / (2 * (order + 1)),
                     q ** (order + 1) / ((order + 1) * (1 - q))))


@dataclass(frozen=True)
class TransferResult:
    source_best_index: int
    proxy_best_index: int
    source_best_information: float
    transferred_information: float
    uniform_candidate_error: float
    candidate_regret: float


def candidate_transfer(
    source: DirectionalLaw,
    proxy: DirectionalLaw,
    candidates: ArrayLike,
    snr: float,
    scalar_information: Callable[[ArrayLike], ArrayLike] = gaussian_information,
) -> TransferResult:
    """Evaluate design transfer over a supplied finite family of bases.

    This reports candidate-family regret, not a global optimizer certificate.
    """
    q = np.asarray(candidates, dtype=float)
    if q.ndim != 3 or len(q) == 0:
        raise ValueError("candidates must have shape (m, d, K), m > 0")
    if source.directions.shape[1] != proxy.directions.shape[1]:
        raise ValueError("source and proxy dimensions must agree")
    source_values = np.array([
        information(source, basis, snr, scalar_information) for basis in q
    ])
    proxy_values = np.array([
        information(proxy, basis, snr, scalar_information) for basis in q
    ])
    best_source = int(np.argmax(source_values))
    best_proxy = int(np.argmax(proxy_values))
    return TransferResult(
        source_best_index=best_source,
        proxy_best_index=best_proxy,
        source_best_information=float(source_values[best_source]),
        transferred_information=float(source_values[best_proxy]),
        uniform_candidate_error=float(np.max(np.abs(source_values - proxy_values))),
        candidate_regret=float(source_values[best_source] - source_values[best_proxy]),
    )
