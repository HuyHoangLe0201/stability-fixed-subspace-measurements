# Stability of Information-Optimal Fixed Subspace Measurements Under Directional Uncertainty

This repository contains the manuscript source, compiled PDF, numerical data, and Python code needed to reproduce the figures and numerical checks. It accompanies the paper by Huy Hoang Le and Kim-Anh Nguyen.

## Manuscript

- `paper/main.pdf`: compiled one-column manuscript.
- `paper/main.tex`: entry point for the LaTeX source. All TikZ and PGFPlots mechanisms are defined there.
- `paper/*.tex`: proofs, numerical sections, and generated numerical constants.
- `paper/figures/*.pdf`: generated numerical figures.

The manuscript uses IEEEtran and TeX Live. On Windows, run `./build.ps1`; it finds `pdflatex.exe` on `PATH` or accepts an explicit `-TeXBin` directory. On Linux or macOS, run `make`. The repository includes `tex/IEEEtran.cls` for reproducibility.

## Reproduce the results

Install the packages in `requirements.txt`, then run:

```text
python scripts/reproduce.py
python scripts/reproduce_core.py
python scripts/reproduce_rank.py
python scripts/verify_theory.py
python scripts/verify_revision.py
```

Rebuild the manuscript after regenerating the data. The scripts use fixed seeds 20260918, 20260919, and 20260920, respectively. Finite-law expectations are direct weighted sums. The output files under `experiments/` include the numerical values, compressed directional laws, candidate projectors, and residuals used in the paper.

Angular searches, finite candidate families, and sampled suprema are numerical diagnostics; they do not certify global optimality over all projectors. Analytic bounds and exact identities are proved in the manuscript. The Gaussian and projective-Wasserstein checks are independent calculations, not substitutes for the proofs.

## Repository layout

- `scripts/reproduce_core.py`: four-direction optimizer, orientation, compression, and finite-sample experiments.
- `scripts/reproduce_rank.py`: rank-two design transfer in eight dimensions.
- `scripts/reproduce.py`: Haar and finite-design reference calculations.
- `scripts/verify_theory.py` and `scripts/verify_revision.py`: independent numerical checks.
- `experiments/core_revision/`, `experiments/rank_revision/`, and `experiments/reproduced/`: saved results and metadata.

The repository excludes historical manuscript drafts and experiments that are not used by the current paper.
