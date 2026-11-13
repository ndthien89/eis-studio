"""Distribution of relaxation times (DRT) via bayes_drt2.

bayes_drt2 (J. Huang, M. Papac, R. O'Hayre, Electrochim. Acta 367, 137493,
2020; https://github.com/jdhuang-csm/bayes-drt2) offers three solvers:

* ``ridge`` - hyperparametric ridge regression (Ciucci & Chen 2015; Effat &
              Ciucci 2017). Needs only cvxopt. Fast, always available.
* ``map``   - calibrated hierarchical Bayesian model, maximum a posteriori
              estimate (L-BFGS in Stan).                 Needs cmdstan.
* ``hmc``   - the same model sampled with Hamiltonian Monte Carlo, giving a
              credible interval.                         Needs cmdstan.

Three adaptations are made here (checked against commit 99d5b60, Jan 2025):
0. NumPy 2 removed ``np.trapz``; it is aliased to ``np.trapezoid`` (below).
1. ``Inverter.fit_peaks`` calls ``predict_distribution(eval_tau, distribution,
   ...)`` positionally, but the signature is ``(name, tau, ...)``, so the tau
   array lands in ``name``. Peaks are therefore fitted by calling
   ``bayes_drt2.peak_fit.fit_peaks`` directly on a correctly evaluated DRT.
2. Stan compiles models next to the ``.stan`` files, i.e. inside
   site-packages, which is read-only on most deployments. The model files are
   copied to a writable cache and the library is pointed there.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# NumPy 2 removed np.trapz (renamed np.trapezoid). bayes_drt2 (matrices.py,
# peak_fit.py) still calls np.trapz, so alias it when missing. Same function.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

METHODS = {
    "ridge": "Ridge",
    "map": "Bayesian MAP",
    "hmc": "Bayesian HMC",
}
METHOD_DETAIL = {
    "ridge": "hyperparametric ridge regression",
    "map": "hierarchical Bayesian, MAP estimate",
    "hmc": "hierarchical Bayesian, HMC sampling, 95 % credible interval",
}

STAN_CACHE = os.environ.get(
    "EIS_STUDIO_STAN_CACHE", os.path.join(tempfile.gettempdir(), "eis_studio_stan"))


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------
@dataclass
class Availability:
    ridge: bool
    bayes: bool
    message: str = ""


def availability() -> Availability:
    try:
        import bayes_drt2.inversion  # noqa: F401
    except Exception as exc:  # not installed / broken install
        return Availability(False, False,
                            f"bayes_drt2 is not installed ({type(exc).__name__}). "
                            "Install it with: pip install -r requirements.txt")
    try:
        import cmdstanpy
        cmdstanpy.cmdstan_path()
    except Exception:
        return Availability(True, False,
                            "Bayesian solvers need CmdStan. Install once with "
                            "python -c \"import cmdstanpy; cmdstanpy.install_cmdstan()\" "
                            "(Windows: add compiler=True). Ridge works without it.")
    return Availability(True, True, "")


def _use_writable_stan_dir() -> None:
    """Point bayes_drt2 at a writable copy of its Stan model files."""
    import bayes_drt2.inversion as inv
    src = os.path.join(os.path.dirname(inv.__file__), "stan_model_files")
    dst = os.path.join(STAN_CACHE, "stan_model_files")
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith(".stan") and not os.path.exists(os.path.join(dst, name)):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
    inv.script_dir = STAN_CACHE


def precompile(models=("Series", "Series_pos", "Series_outliers",
                       "Series_pos_outliers")) -> None:
    """Compile the single-DRT Stan models ahead of time (used by Docker)."""
    from cmdstanpy import CmdStanModel
    _use_writable_stan_dir()
    for m in models:
        CmdStanModel(stan_file=os.path.join(STAN_CACHE, "stan_model_files", m + ".stan"))


# --------------------------------------------------------------------------
# options / results
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DRTOptions:
    method: str = "ridge"              # ridge | map | hmc
    nonneg: bool = True
    # ridge
    preset: str = "Huang"              # Huang | Ciucci | none
    fit_inductance: bool = True
    # Bayesian
    outliers: str = "auto"             # auto | on | off
    init_from_ridge: bool = True
    warmup: int = 200
    samples: int = 200
    chains: int = 2
    # peaks
    find_peaks: bool = True
    check_shoulders: bool = False      # shoulder search tends to add degenerate peaks
    prom_rthresh: float = 0.005
    R_rthresh: float = 0.005


@dataclass
class DRTResult:
    name: str
    method: str
    f: np.ndarray
    Z: np.ndarray
    Z_fit: np.ndarray
    tau: np.ndarray
    gamma: np.ndarray
    gamma_lo: np.ndarray | None
    gamma_hi: np.ndarray | None
    R_inf: float
    inductance: float
    Rp: float
    peaks: pd.DataFrame
    peak_curves: list[np.ndarray] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unit: str = "Ω"

    @property
    def res_real(self) -> np.ndarray:
        return (self.Z.real - self.Z_fit.real) / np.abs(self.Z)

    @property
    def res_imag(self) -> np.ndarray:
        return (self.Z.imag - self.Z_fit.imag) / np.abs(self.Z)

    @property
    def mean_rel_error(self) -> float:
        return float(np.mean(np.abs(self.Z - self.Z_fit) / np.abs(self.Z)) * 100)

    def distribution_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({"tau_s": self.tau, "freq_Hz": 1 / (2 * np.pi * self.tau),
                           "gamma": self.gamma})
        if self.gamma_lo is not None:
            df["gamma_2.5pct"] = self.gamma_lo
            df["gamma_97.5pct"] = self.gamma_hi
        for k, c in enumerate(self.peak_curves, 1):
            df[f"peak_{k}"] = c
        return df

    def fit_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "freq_Hz": self.f, "Z_real": self.Z.real, "Z_imag": self.Z.imag,
            "Zdrt_real": self.Z_fit.real, "Zdrt_imag": self.Z_fit.imag,
            "resid_real_pct": self.res_real * 100, "resid_imag_pct": self.res_imag * 100})

    def summary_row(self) -> dict:
        row = {"File": self.name, "Method": self.method, "R∞": self.R_inf,
               "Rp": self.Rp, "L (H)": self.inductance,
               "Peaks": len(self.peaks), "Mean |ΔZ|/|Z| %": self.mean_rel_error}
        for k, r in self.peaks.reset_index(drop=True).iterrows():
            row[f"R{k + 1}"] = r["R"]
            row[f"f{k + 1} (Hz)"] = r["f_peak (Hz)"]
        return row


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
def _tau_grid(inv) -> np.ndarray:
    basis_tau = inv.distributions["DRT"]["tau"]
    lo, hi = np.log10(basis_tau.min()) - 1, np.log10(basis_tau.max()) + 1
    return np.logspace(lo, hi, int(20 * (hi - lo) + 1))


def _peaks(tau, gamma, Rp, opt: DRTOptions, f_range=(0.0, np.inf)):
    from bayes_drt2 import peak_fit as pf
    if not opt.find_peaks or not np.isfinite(Rp) or Rp <= 0:
        return pd.DataFrame(), []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = pf.fit_peaks(tau, gamma, Rp, nonneg=bool(np.min(gamma) >= 0),
                         check_shoulders=opt.check_shoulders,
                         prom_rthresh=opt.prom_rthresh, R_rthresh=opt.R_rthresh)
    x = np.asarray(x, float)
    if x.size == 0:
        return pd.DataFrame(), []
    # An HN peak with alpha≈0 or beta≈0 has (almost) zero distribution: it is a
    # flat resistance the optimiser uses to satisfy its Rp constraint, not a
    # physical process. Drop those.
    keep = [k for k in range(x.size // 4)
            if x[4 * k + 2] >= 0.05 and x[4 * k + 3] >= 0.05]
    if not keep:
        return pd.DataFrame(), []
    order = sorted(keep, key=lambda k: x[4 * k + 1])
    rows, curves = [], []
    for k in order:
        xi = x[4 * k: 4 * k + 4]
        R, t0, a, b = xi[0], float(np.exp(xi[1])), xi[2], xi[3]
        fpk = 1 / (2 * np.pi * t0)
        rows.append({"Peak": len(rows) + 1, "R": float(R), "τ₀ (s)": t0,
                     "f_peak (Hz)": fpk, "α (HN)": float(a),
                     "β (HN)": float(b), "Share of Rp %": float(R / Rp * 100),
                     # peaks outside the measured window are extrapolations
                     "Inside data range": bool(f_range[0] <= fpk <= f_range[1])})
        curves.append(pf.evaluate_fit_distribution(xi, tau))
    return pd.DataFrame(rows), curves


def compute_drt(name: str, f, Z, opt: DRTOptions, unit: str = "Ω") -> DRTResult:
    """Run one DRT inversion. Raises RuntimeError with a friendly message."""
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    av = availability()
    if not av.ridge:
        raise RuntimeError(av.message)
    if opt.method in ("map", "hmc") and not av.bayes:
        raise RuntimeError(av.message)
    if f.size < 10:
        raise RuntimeError("DRT needs at least 10 data points.")

    from bayes_drt2.inversion import Inverter

    notes: list[str] = []
    inv = Inverter(fit_inductance=opt.fit_inductance)
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
        warnings.simplefilter("always")
        if opt.method == "ridge":
            kw = {"nonneg": opt.nonneg}
            if opt.preset in ("Huang", "Ciucci"):
                kw["preset"] = opt.preset
            inv.ridge_fit(f, Z, **kw)
        else:
            _use_writable_stan_dir()
            outliers = {"auto": "auto", "on": True, "off": False}[opt.outliers]
            kw = dict(nonneg=opt.nonneg, outliers=outliers,
                      init_from_ridge=opt.init_from_ridge)
            if opt.method == "map":
                inv.fit(f, Z, mode="optimize", **kw)
            else:
                inv.fit(f, Z, mode="sample", warmup=int(opt.warmup),
                        samples=int(opt.samples), chains=int(opt.chains), **kw)
    for w in caught:
        msg = str(w.message)
        if "outlier" in msg.lower() and msg not in notes:
            notes.append(msg)

    tau = _tau_grid(inv)
    gamma = np.asarray(inv.predict_distribution(name="DRT", tau=tau), float)
    lo = hi = None
    if opt.method == "hmc":
        lo = np.asarray(inv.predict_distribution(name="DRT", tau=tau, percentile=2.5), float)
        hi = np.asarray(inv.predict_distribution(name="DRT", tau=tau, percentile=97.5), float)
    Z_fit = np.asarray(inv.predict_Z(f), complex)
    Rp = float(inv.predict_Rp())
    R_inf = float(np.asarray(inv.R_inf).ravel()[0])
    L = float(np.asarray(inv.inductance).ravel()[0])
    peaks, curves = _peaks(tau, gamma, Rp, opt, (float(f.min()), float(f.max())))
    if not peaks.empty and not peaks["Inside data range"].all():
        notes.append("Some peaks lie outside the measured frequency range; they are "
                      "extrapolations (or artefacts of very sharp, ideal-RC processes) "
                      "and should not be interpreted as physical processes.")
    if not peaks.empty and Rp > 0:
        share = peaks["R"].sum() / Rp * 100
        if abs(share - 100) > 5:
            notes.append(f"The fitted peaks account for {share:.0f} % of Rp. The rest "
                         "sits in the broad tails of the DRT - typical for very sharp "
                         "(near-ideal RC) processes, which regularisation widens. Use Rp "
                         "for the total and the peak R values for relative comparisons.")
    return DRTResult(name, opt.method, f, Z, Z_fit, tau, gamma, lo, hi,
                     R_inf, L, Rp, peaks, curves, notes, unit)
