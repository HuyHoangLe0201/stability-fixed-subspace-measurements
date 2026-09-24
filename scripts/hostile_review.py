"""Static final audit for claims, references, and presentation invariants."""
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parents[1]
paper = root / "paper"

seen = []


def expand(path):
    seen.append(path)
    source = path.read_text(encoding="utf-8")
    return re.sub(r"\\input\{([^}]+)\}",
                  lambda match: expand(path.parent / match[1]), source)


source = expand(paper / "main.tex")
checks = {
    "three_main_theorems": source.count(r"\begin{theorem}") == 3,
    "projective_transport_defined": all(token in source for token in [
        r"\rho(u,v)=\min", r"W_1^{\rm proj}",
        r"\sqrt{2-2|u^{\top}v|}"
    ]),
    "sharp_projective_lipschitz_constant": all(token in source for token in [
        r"\le \gamma L_\gamma W_1^{\rm proj}",
        r"\le2\gamma L_\gamma W_1^{\rm proj}",
        r"\le\sin\theta"
    ]),
    "jensen_wording_projector_specific":
        "Both displayed choices give constant coverage" in source,
    "information_theory_framing":
        "quantified directly in mutual-information units" in source,
    "joint_stability_named": "Joint SNR--directional-law stability" in source,
    "ambient_dimension_remark": all(token in source for token in [
        "Ambient versus intrinsic covering dimension", "K(d-K)"
    ]),
    "quadratic_growth_scope": all(token in source for token in [
        "When quadratic growth is available", "Uniqueness alone does"
    ]),
    "gaussian_joint_specialization": r"\label{eq:joint_gaussian}" in source,
    "moment_transport_roles_distinguished": all(token in source for token in [
        "algebraic representation guarantee", "finite-SNR metric"
    ]),
    "weighted_cubature_not_unweighted_design":
        "need not be a classical unweighted design" in source,
    "bpsk_removed_from_manuscript": all(token not in source for token in [
        "Binary Phase-Shift Keying", "eq:bpsk", "h_{\\rm BPSK}"
    ]),
    "no_framed_formulas": not re.search(r"\\(?:boxed|fbox|fboxed)\b", source),
    "haar_average_wording_fixed": "has value no smaller than this average" in source,
    "acronyms_defined_at_first_use": all(token in source for token in [
        "signal-to-noise ratio (SNR)",
        "mutual-information--minimum-mean-square-error (I-MMSE)",
        "minimum mean-square error (MMSE)",
        "additive white Gaussian\nnoise (AWGN)",
        "comma-separated-value (CSV)",
    ]),
    "notation_defined_before_dependent_display": all(token in source for token in [
        r"Let $h_R(s)=I(R;\sqrt{s}R+Z_0)$ denote",
        r"let $\I_{\rm vertex}(\gamma)$ denote",
        r"$P_{N,r}$ under $\nu_{N,r}$",
        r"Let $T_\gamma$ denote the chart coordinate",
    ]),
    "covering_ball_center_explicit":
        "radius-$\\sqrt K$ ball centered at the origin" in source,
    "author_orcids_present": all(token in source for token in [
        r"\orcidlink{0009-0009-2138-5468}",
        r"\orcidlink{0000-0003-3408-847X}",
    ]),
}

labels = re.findall(r"\\label\{([^}]+)\}", source)
refs = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", source)
bib = re.findall(r"\\bibitem\{([^}]+)\}", source)
cites = [key for group in re.findall(r"\\cite\{([^}]+)\}", source)
         for key in group.split(",")]
checks.update({
    "unique_labels": len(labels) == len(set(labels)),
    "all_references_resolved": set(refs) <= set(labels),
    "all_bibliography_items_used": set(cites) == set(bib),
})

log = (paper / "main.log").read_text(errors="replace")
checks["clean_layout_log"] = not re.search(
    r"undefined references|undefined citations|Overfull \\[hv]box|^!", log, re.M)

revision = json.loads((root / "validation/revision_checks.json").read_text())
theory = json.loads((root / "validation/theory_checks.json").read_text())
checks["independent_numerical_checks"] = (
    revision.get("status") == "passed" and theory.get("status") == "passed")

assert all(checks.values()), {key: value for key, value in checks.items() if not value}
result = {"status": "passed", "checks": checks,
          "source_files": [str(path.relative_to(root)) for path in seen]}
(root / "validation/hostile_review.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
