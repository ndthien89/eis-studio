"""Software and literature that EIS Studio builds on - single source of truth
for the in-app 'How to cite' panels, CITATION.txt in exported ZIPs, and docs."""

from __future__ import annotations

SOFTWARE = {
    "impedance.py": {
        "role": "File readers, equivalent-circuit models and fitting, Lin-KK test",
        "authors": "M. D. Murbach, B. Gerwe, N. Dawson-Elli, L. Tsui",
        "url": "https://github.com/ECSHackWeek/impedance.py",
        "version": "1.7.1 (PyPI); Lin-KK evaluation back-ported from GitHub main",
        "license": "MIT",
        "paper": ("M. D. Murbach, B. Gerwe, N. Dawson-Elli, L. Tsui, impedance.py: "
                  "A Python package for electrochemical impedance analysis, "
                  "J. Open Source Softw. 5 (52), 2349 (2020). "
                  "https://doi.org/10.21105/joss.02349"),
    },
    "bayes_drt2": {
        "role": "Distribution of relaxation times (ridge, Bayesian MAP and HMC), "
                "Havriliak–Negami peak decomposition",
        "authors": "J. Huang",
        "url": "https://github.com/jdhuang-csm/bayes-drt2",
        "version": "0.2, commit 99d5b603d98469a6382ddde5210a1bfdc0d76b3d (2025-01-11)",
        "license": "BSD 3-Clause",
        "paper": ("J. Huang, M. Papac, R. O'Hayre, Towards robust autonomous impedance "
                  "spectroscopy analysis: a calibrated hierarchical Bayesian approach "
                  "for electrochemical impedance spectroscopy (EIS) inversion, "
                  "Electrochim. Acta 367, 137493 (2021). "
                  "https://doi.org/10.1016/j.electacta.2020.137493"),
    },
}

METHODS = {
    "lin_kk": ("M. Schönleber, D. Klotz, E. Ivers-Tiffée, A method for improving the "
               "robustness of linear Kramers-Kronig validity tests, Electrochim. Acta "
               "131, 20-27 (2014). https://doi.org/10.1016/j.electacta.2014.01.034"),
    "brug": ("G. J. Brug, A. L. G. van den Eeden, M. Sluyters-Rehbach, J. H. Sluyters, "
             "The analysis of electrode impedances complicated by the presence of a "
             "constant phase element, J. Electroanal. Chem. 176, 275-295 (1984). "
             "https://doi.org/10.1016/S0022-0728(84)80324-1"),
    "ridge_lambda": ("F. Ciucci, C. Chen, Analysis of electrochemical impedance "
                     "spectroscopy data using the distribution of relaxation times: "
                     "A Bayesian and hierarchical Bayesian approach, Electrochim. Acta "
                     "167, 439-454 (2015). https://doi.org/10.1016/j.electacta.2015.03.123"),
    "ridge_weights": ("M. B. Effat, F. Ciucci, Bayesian and hierarchical Bayesian based "
                      "regularization for deconvolving the distribution of relaxation "
                      "times from electrochemical impedance spectroscopy data, "
                      "Electrochim. Acta 247, 1117-1129 (2017). "
                      "https://doi.org/10.1016/j.electacta.2017.07.050"),
    "stan": ("B. Carpenter et al., Stan: A probabilistic programming language, "
             "J. Stat. Softw. 76 (1) (2017). https://doi.org/10.18637/jss.v076.i01"),
}


def drt_references(method: str) -> list[str]:
    """References to cite for a DRT result, following the bayes_drt2 README."""
    refs = [SOFTWARE["bayes_drt2"]["paper"]]
    if method == "ridge":
        refs.append(METHODS["ridge_lambda"])     # ridge_fit uses hyper_lambda=True
    else:
        refs.append(METHODS["stan"])
    return refs


def software_line(key: str) -> str:
    s = SOFTWARE[key]
    return f"{key} {s['version']} - {s['url']} ({s['license']} licence)"


def citation_text(drt_method: str | None = None, circuit: bool = False,
                  lin_kk: bool = False) -> str:
    """Plain-text citation block for exported results."""
    lines = ["Results produced with EIS Studio. Please cite the underlying software "
             "and methods:", "", "SOFTWARE"]
    if circuit or lin_kk:
        lines.append("  " + software_line("impedance.py"))
    if drt_method:
        lines.append("  " + software_line("bayes_drt2"))
    lines += ["", "REFERENCES"]
    refs = []
    if circuit or lin_kk:
        refs.append(SOFTWARE["impedance.py"]["paper"])
    if lin_kk:
        refs.append(METHODS["lin_kk"])
    if circuit:
        refs.append(METHODS["brug"])
    if drt_method:
        refs += drt_references(drt_method)
    lines += [f"  [{i}] {r}" for i, r in enumerate(refs, 1)]
    return "\n".join(lines) + "\n"
