"""End-to-end checks of moment preservation and candidate design transfer."""

import unittest

import numpy as np

from fixed_subspace import (
    DirectionalLaw,
    candidate_transfer,
    compress_moments,
    covariance_subspace,
    gaussian_matching_error_bound,
    information,
)


class WorkflowTest(unittest.TestCase):
    def test_compression_preserves_coverage_moments_across_ranks(self) -> None:
        rng = np.random.default_rng(41)
        u = rng.normal(size=(90, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        law = DirectionalLaw(u, rng.uniform(0.1, 1, len(u)))
        proxy = compress_moments(law, order=2, seed=19)
        self.assertLessEqual(len(proxy.law.weights), proxy.feature_count)
        self.assertLess(proxy.max_moment_residual, 1e-8)
        for rank in (1, 2):
            for _ in range(12):
                q, _ = np.linalg.qr(rng.normal(size=(3, rank)))
                original_coverage = np.sum((law.directions @ q) ** 2, axis=1)
                proxy_coverage = np.sum((proxy.law.directions @ q) ** 2, axis=1)
                for power in (1, 2):
                    a = law.weights @ original_coverage ** power
                    b = proxy.law.weights @ proxy_coverage ** power
                    self.assertAlmostEqual(a, b, delta=1e-8)

    def test_information_and_transfer_on_a_new_law(self) -> None:
        rng = np.random.default_rng(12)
        u = rng.normal(size=(60, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        law = DirectionalLaw.empirical(u)
        proxy = compress_moments(law, order=2, seed=12)
        q0, _ = covariance_subspace(law, rank=1)
        candidates = [q0]
        for _ in range(24):
            q, _ = np.linalg.qr(rng.normal(size=(3, 1)))
            candidates.append(q)
        gamma = 1.0
        report = candidate_transfer(law, proxy.law, np.stack(candidates), gamma)
        self.assertLessEqual(report.candidate_regret,
                             2 * report.uniform_candidate_error + 1e-12)
        self.assertLessEqual(report.uniform_candidate_error,
                             gaussian_matching_error_bound(gamma, 2) + 1e-8)
        self.assertGreaterEqual(information(law, q0, gamma), 0)

    def test_reject_nonunit_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "unit"):
            DirectionalLaw.empirical([[2, 0], [0, 1]])

    def test_rank_two_transfer_in_eight_dimensions(self) -> None:
        rng = np.random.default_rng(82)
        u = rng.normal(size=(120, 8))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        law = DirectionalLaw.empirical(u)
        proxy = compress_moments(law, order=1, seed=82)
        q0, _ = covariance_subspace(law, rank=2)
        candidates = [q0]
        for _ in range(12):
            q, _ = np.linalg.qr(rng.normal(size=(8, 2)))
            candidates.append(q)
        report = candidate_transfer(law, proxy.law, np.stack(candidates), snr=2.0)
        self.assertLessEqual(len(proxy.law.weights), proxy.feature_count)
        self.assertLessEqual(report.candidate_regret,
                             2 * report.uniform_candidate_error + 1e-12)
        self.assertAlmostEqual(information(law, q0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
