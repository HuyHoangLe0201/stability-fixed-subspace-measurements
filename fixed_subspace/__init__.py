"""Reusable fixed-subspace information design for directional laws."""

from .core import (
    CompressionResult,
    DirectionalLaw,
    TransferResult,
    candidate_transfer,
    compress_moments,
    covariance_subspace,
    gaussian_information,
    gaussian_matching_error_bound,
    information,
)

__all__ = [
    "CompressionResult",
    "DirectionalLaw",
    "TransferResult",
    "candidate_transfer",
    "compress_moments",
    "covariance_subspace",
    "gaussian_information",
    "gaussian_matching_error_bound",
    "information",
]
