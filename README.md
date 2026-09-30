# Fixed subspace information design

Python tools for choosing a fixed measurement subspace when the signal direction
varies and is known at decoding. The library works with a finite weighted law
on unit directions. It evaluates retained scalar-channel mutual information,
finds the leading covariance eigenspace, compresses the directional law while
matching even moments, and measures design transfer over a user-supplied finite
candidate set.

The default information function is for a **real Gaussian** scalar amplitude:
`0.5 * log(1 + SNR)` nats. Pass a vectorized `scalar_information` function to
`information` or `candidate_transfer` for another amplitude prior. SNR values
are linear, not decibels. The code does not assume the covariance subspace is
globally information-optimal at finite SNR.

## Install

```bash
python -m pip install .
```

## Apply to your own directions

Store one unit direction per row in a numeric CSV. For example, to compress a
3-dimensional empirical law through degree-four moments and evaluate a rank-one
covariance design at SNR 1:

```bash
fixed-subspace directions.csv --rank 1 --order 2 --snr 1 --output compressed_law.npz
```

Add `--weights weights.csv` for a weighted law. The output contains retained
directions, normalized weights, their original row indices, and the covariance
basis. The command prints the largest moment residual and information values.
It does not claim to find a global finite-SNR optimizer.

Python users can work directly with the library:

```python
from fixed_subspace import DirectionalLaw, compress_moments, covariance_subspace, information

law = DirectionalLaw.empirical(directions)  # shape (n, d); unit rows
basis, eigenvalues = covariance_subspace(law, rank=2)
proxy = compress_moments(law, order=2, seed=0)
value = information(law, basis, snr=1.0)
```

Run `python -m examples.basic_usage` for the full workflow, including candidate
design transfer. Run `python -m examples.reproduce_checks` to regenerate two
representative numerical checks with fixed settings. Neither example downloads
data or needs the manuscript. Run `python -m unittest discover -s tests` to
check the installation.

## Interpretation and limits

- `covariance_subspace` optimizes the leading low-SNR term. The finite-SNR
  optimum may differ.
- `compress_moments` solves a positive linear program on the supplied support.
  Degree `2 * order` matching on the unit sphere also matches lower even
  degrees. It reports a floating-point residual, and the number of moment
  features grows rapidly with dimension and order.
- `candidate_transfer` compares a **finite family** of subspaces. Its regret is
  relative to that family, not all possible subspaces.
- `gaussian_matching_error_bound` is the theoretical uniform bound for exact
  even-moment matching. Numerical residuals need separate consideration.

The repository contains reusable method code and small numerical examples.
It does not contain the manuscript, compiled paper, paper-specific plots, or
archived experiment outputs.

The code is available under the MIT License; retain the copyright notice when
reusing it.
