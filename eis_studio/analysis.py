"""Analysis engine for EIS Studio, built on impedance.py.

* preprocessing  - frequency window, inductive-point removal, sign flip,
                   point masking, area normalisation
* validation     - Lin-KK (Schönleber et al. 2014) via impedance.validation
* fitting        - equivalent-circuit fitting via impedance.models.circuits
* derived values - effective capacitance (Brug) and characteristic frequency
                   for every R‖C / R‖CPE pair
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from impedance.models.circuits import CustomCircuit
from impedance.models.circuits.elements import circuit_elements, get_element_from_name
from impedance.models.circuits.fitting import (
    buildCircuit,
    check_and_eval,
    extract_circuit_elements,
    set_default_bounds,
)
import impedance.validation as _kk
from impedance.validation import linKK


# --------------------------------------------------------------------------
# Compatibility: impedance 1.7.1 (PyPI) builds its Lin-KK eval string from
# NumPy scalars, which print as "np.float64(...)" under NumPy 2 and crash.
# This is the fixed implementation from impedance.py `main` on GitHub.
# --------------------------------------------------------------------------
def _eval_linKK(elements, ts, f):
    el, tl, fl = np.asarray(elements).tolist(), np.asarray(ts).tolist(), \
        np.asarray(f).tolist()
    s = f"s([R({[el[0]]},{fl}),"
    for Rk, tk in zip(el[1:], tl):
        s += f"K({[Rk, tk]},{fl}),"
    s += f"L({[el[-1]]},{fl}),"
    if len(el) == len(tl) + 3:
        s += f"C({[1 / el[-2]]},{fl}),"
    s = s.strip(",") + "])"
    return eval(s, circuit_elements)  # noqa: S307


_kk.eval_linKK = _eval_linKK

# --------------------------------------------------------------------------
# Circuit library
# --------------------------------------------------------------------------
ELEMENT_INFO: dict[str, str] = {
    "R": "Resistor",
    "C": "Capacitor",
    "L": "Inductor",
    "CPE": "Constant phase element (Q, α)",
    "W": "Semi-infinite Warburg",
    "Wo": "Finite-space (open) Warburg",
    "Ws": "Finite-length (short) Warburg",
    "La": "Modified inductance (L, α)",
    "G": "Gerischer",
    "Gs": "Finite-length Gerischer",
    "K": "RC element (R, τ)",
    "Zarc": "Zarc / Cole–Cole (R, τ, γ)",
    "TLMQ": "Transmission line, CPE (porous electrode)",
    "T": "Paasch porous electrode",
}

PRESETS: dict[str, dict[str, str]] = {
    "R‖C": {
        "circuit": "R0-p(R1,C1)",
        "about": "Ohmic resistance plus one ideal semicircle.",
    },
    "R‖Q  (one arc)": {
        "circuit": "R0-p(R1,CPE1)",
        "about": "Ohmic resistance plus one depressed semicircle.",
    },
    "Randles  (Wo, C)": {
        "circuit": "R0-p(R1-Wo1,C1)",
        "about": "Charge transfer in parallel with double layer, finite diffusion.",
    },
    "Randles  (Wo, CPE)": {
        "circuit": "R0-p(R1-Wo1,CPE1)",
        "about": "Randles circuit with a non-ideal double layer.",
    },
    "Two arcs  (R‖Q – R‖Q)": {
        "circuit": "R0-p(R1,CPE1)-p(R2,CPE2)",
        "about": "Two time constants, e.g. SEI / surface film + charge transfer.",
    },
    "Li-ion cell  (L-R-RQ-RQ-W)": {
        "circuit": "L0-R0-p(R1,CPE1)-p(R2,CPE2)-W1",
        "about": "Cable inductance, ohmic, SEI, charge transfer, solid diffusion.",
    },
    "Li-ion cell  (R-RQ-(R-Wo)Q)": {
        "circuit": "R0-p(R1,CPE1)-p(R2-Wo1,CPE2)",
        "about": "Surface film arc plus Randles-type charge transfer with diffusion.",
    },
    "Coated metal": {
        "circuit": "R0-p(CPE1,R1-p(R2,CPE2))",
        "about": "Coating capacitance and pore resistance, nested interface arc.",
    },
    "Gerischer  (SOFC cathode)": {
        "circuit": "R0-p(R1,CPE1)-G1",
        "about": "Co-limited surface exchange and bulk diffusion.",
    },
    "Porous electrode  (TLM)": {
        "circuit": "R0-TLMQ1",
        "about": "Blocking porous electrode, transmission-line model.",
    },
}

_SAFE_CIRCUIT = re.compile(r"^[A-Za-z0-9_,()\-\s]+$")


class CircuitError(ValueError):
    """Raised for an invalid circuit string."""


@dataclass
class Param:
    name: str
    unit: str
    element: str


def parse_circuit(circuit: str) -> list[Param]:
    """Validate a circuit string and return its parameters in fit order."""
    circuit = (circuit or "").replace(" ", "")
    if not circuit:
        raise CircuitError("Enter a circuit, e.g. R0-p(R1,C1).")
    if not _SAFE_CIRCUIT.match(circuit):
        raise CircuitError("Only element names, digits, '_', '-', ',' and "
                           "parentheses are allowed.")
    depth = 0
    for ch in circuit:
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            break
    if depth != 0:
        raise CircuitError("Parentheses are not balanced.")
    if "(" in circuit.replace("p(", ""):
        raise CircuitError("Use p(a,b) for parallel groups.")
    params: list[Param] = []
    seen = set()
    for elem in extract_circuit_elements(circuit):
        if not elem:
            raise CircuitError("Empty element. Check '-', ',' and parentheses.")
        raw = get_element_from_name(elem)
        if raw not in circuit_elements or raw in ("s", "p"):
            raise CircuitError(f"Unknown element '{elem}'. "
                               f"Allowed: {', '.join(ELEMENT_INFO)}.")
        if elem in seen:
            raise CircuitError(f"Element name '{elem}' is used twice; "
                               "number each element uniquely.")
        seen.add(elem)
        fn = check_and_eval(raw)
        if fn.num_params > 1:
            for j in range(fn.num_params):
                params.append(Param(f"{elem}_{j}", fn.units[j], elem))
        else:
            params.append(Param(elem, fn.units[0], elem))
    # final structural check: build and evaluate once
    try:
        simulate(circuit, {p.name: 1.0 for p in params}, np.array([1.0, 10.0]))
    except Exception as exc:  # malformed nesting
        raise CircuitError(f"Circuit could not be built ({exc}).") from exc
    return params


def simulate(circuit: str, values: dict[str, float], f: np.ndarray) -> np.ndarray:
    """Impedance of a circuit with all parameters given by name."""
    f = np.asarray(f, dtype=float)
    expr, _ = buildCircuit(circuit.replace(" ", ""), f,
                           constants={k: float(v) for k, v in values.items()},
                           eval_string="", index=0)
    with np.errstate(all="ignore"):
        return np.asarray(eval(expr, circuit_elements), dtype=complex)  # noqa: S307


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
@dataclass
class Preprocess:
    fmin: float | None = None
    fmax: float | None = None
    drop_inductive: bool = False      # remove points with Z″ > 0
    flip_imag: bool = False           # file stores −Z″
    area_cm2: float | None = None     # multiply Z by area → Ω·cm²
    exclude: tuple[int, ...] = ()     # indices into the raw (sorted) data

    @property
    def unit(self) -> str:
        return "Ω·cm²" if self.area_cm2 else "Ω"


def preprocess(f: np.ndarray, Z: np.ndarray, opt: Preprocess
               ) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(f, float).copy()
    Z = np.asarray(Z, complex).copy()
    keep = np.ones(f.size, bool)
    if opt.exclude:
        idx = [i for i in opt.exclude if 0 <= i < f.size]
        keep[idx] = False
    if opt.flip_imag:
        Z = Z.real - 1j * Z.imag
    if opt.fmin:
        keep &= f >= opt.fmin
    if opt.fmax:
        keep &= f <= opt.fmax
    if opt.drop_inductive:
        keep &= Z.imag < 0
    f, Z = f[keep], Z[keep]
    if opt.area_cm2:
        Z = Z * float(opt.area_cm2)
    return f, Z


def summary_stats(f: np.ndarray, Z: np.ndarray) -> dict[str, float]:
    i_hi, i_lo = int(np.argmax(f)), int(np.argmin(f))
    # high-frequency intercept: interpolate where Z″ crosses zero, else Z′ at f_max
    r_hf = float(Z.real[i_hi])
    cross = np.where(np.diff(np.sign(Z.imag)) != 0)[0]
    if cross.size:
        k = cross[0]
        z1, z2 = Z[k], Z[k + 1]
        if z2.imag != z1.imag:
            t = -z1.imag / (z2.imag - z1.imag)
            r_hf = float(z1.real + t * (z2.real - z1.real))
    i_pk = int(np.argmax(-Z.imag))
    return {
        "n": int(f.size),
        "f_max": float(f[i_hi]),
        "f_min": float(f[i_lo]),
        "r_hf": r_hf,
        "z_lf": float(np.abs(Z[i_lo])),
        "f_peak": float(f[i_pk]),
    }


# --------------------------------------------------------------------------
# Initial guesses and bounds
# --------------------------------------------------------------------------
def _suffix(elem: str) -> str:
    m = re.search(r"(\d+)$", elem)
    return m.group(1) if m else elem


def initial_guess(circuit: str, f: np.ndarray, Z: np.ndarray) -> dict[str, float]:
    """Physically sensible starting values estimated from the spectrum."""
    params = parse_circuit(circuit)
    st = summary_stats(f, Z)
    r_hf = max(st["r_hf"], 1e-6 * max(np.abs(Z).max(), 1e-12))
    span = float(Z.real.max() - r_hf)
    if not np.isfinite(span) or span <= 0:
        span = float(np.abs(Z).max()) or 1.0
    elements = list(dict.fromkeys(p.element for p in params))
    raw = {e: get_element_from_name(e) for e in elements}
    arcs = [e for e in elements if raw[e] in ("C", "CPE")]
    n_arc = max(len(arcs), 1)
    if len(arcs) > 1:
        f_arc = np.logspace(np.log10(st["f_max"]) - 1,
                            np.log10(st["f_min"]) + 1, len(arcs))
    else:
        f_arc = np.array([st["f_peak"]])
    arc_f = {e: float(fa) for e, fa in zip(arcs, f_arc)}
    by_suffix = {_suffix(e): arc_f[e] for e in arcs}
    series_r = [e for e in elements if raw[e] == "R"]
    first_r = series_r[0] if series_r else None
    r_arc = span / n_arc
    omega_min = 2 * np.pi * st["f_min"]
    imag_hi = Z.imag[np.argmax(f)]
    w_hi = 2 * np.pi * st["f_max"]

    g: dict[str, float] = {}
    for e in elements:
        kind = raw[e]
        fa = by_suffix.get(_suffix(e), st["f_peak"])
        tau = 1.0 / (2 * np.pi * fa)
        if kind == "R":
            g[e] = r_hf if (e == first_r and e.endswith("0")) else r_arc
        elif kind == "C":
            g[e] = tau / r_arc
        elif kind == "CPE":
            g[f"{e}_0"], g[f"{e}_1"] = tau / r_arc, 0.9
        elif kind == "L":
            g[e] = float(imag_hi / w_hi) if imag_hi > 0 else 1e-7
        elif kind == "La":
            g[f"{e}_0"], g[f"{e}_1"] = (float(imag_hi / w_hi) if imag_hi > 0 else 1e-7), 0.9
        elif kind == "W":
            g[e] = 0.2 * span * np.sqrt(omega_min)
        elif kind in ("Wo", "Ws"):
            g[f"{e}_0"], g[f"{e}_1"] = 0.3 * span, 1.0 / omega_min
        elif kind == "G":
            g[f"{e}_0"], g[f"{e}_1"] = 0.3 * span, tau
        elif kind == "Gs":
            g[f"{e}_0"], g[f"{e}_1"], g[f"{e}_2"] = 0.3 * span, tau, 1.0
        elif kind == "K":
            g[f"{e}_0"], g[f"{e}_1"] = r_arc, tau
        elif kind == "Zarc":
            g[f"{e}_0"], g[f"{e}_1"], g[f"{e}_2"] = r_arc, tau, 0.9
        elif kind == "TLMQ":
            g[f"{e}_0"], g[f"{e}_1"], g[f"{e}_2"] = span, tau / span, 0.9
        elif kind == "T":
            g[f"{e}_0"], g[f"{e}_1"], g[f"{e}_2"], g[f"{e}_3"] = span, span, 1.0, 1.0
    out = {}
    for p in params:
        v = float(g.get(p.name, 1.0))
        out[p.name] = v if np.isfinite(v) and v > 0 else 1.0
    return out


def default_bounds(circuit: str) -> dict[str, tuple[float, float]]:
    params = parse_circuit(circuit)
    lo, hi = set_default_bounds(circuit.replace(" ", ""), constants={})
    bounds = {p.name: (float(a), float(b)) for p, a, b in zip(params, lo, hi)}
    for p in params:   # exponents of Zarc / TLMQ also live in (0, 1]
        raw = get_element_from_name(p.element)
        if (raw == "Zarc" and p.name.endswith("_2")) or \
                (raw == "TLMQ" and p.name.endswith("_2")):
            bounds[p.name] = (0.0, 1.0)
    return bounds


def pretty_unit(u: str) -> str:
    """'Ohm^-1 sec^a' → 'Ω⁻¹ sᵃ' etc. (display only)."""
    if not u:
        return "–"
    for a, b in (("Ohm-m^2", "Ω·m²"), ("Ohm cm^2", "Ω·cm²"), ("Ohm^-1", "Ω⁻¹"),
                 ("Ohm", "Ω"), ("sec^-1/2", "s⁻¹ᐟ²"), ("sec^(gamma - 1)", "s^(γ−1)"),
                 ("sec^a", "sᵅ"), ("sec", "s")):
        u = u.replace(a, b)
    return u


def parameter_table(circuit: str, f: np.ndarray, Z: np.ndarray,
                    previous: pd.DataFrame | None = None) -> pd.DataFrame:
    """Editable parameter table: name, unit, initial, fixed, lower, upper."""
    params = parse_circuit(circuit)
    guess = initial_guess(circuit, f, Z)
    bounds = default_bounds(circuit)
    rows = []
    prev = previous.set_index("Parameter") if previous is not None and len(previous) else None
    for p in params:
        row = {
            "Parameter": p.name,
            "Unit": pretty_unit(p.unit),
            "Initial": guess[p.name],
            "Fixed": False,
            "Lower": bounds[p.name][0],
            "Upper": bounds[p.name][1],
        }
        if prev is not None and p.name in prev.index:
            for col in ("Initial", "Fixed", "Lower", "Upper"):
                row[col] = prev.loc[p.name, col]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Lin-KK validation
# --------------------------------------------------------------------------
KK_MODES = {
    "auto": "Auto (robust μ scan)",
    "schonleber": "Schönleber (library default)",
    "manual": "Manual M",
}


@dataclass
class KKResult:
    M: int
    mu: float
    Z_fit: np.ndarray
    res_real: np.ndarray
    res_imag: np.ndarray
    scan: pd.DataFrame | None = None       # M, mu, chi2 for every M tried

    @property
    def max_residual(self) -> float:
        return float(max(np.abs(self.res_real).max(), np.abs(self.res_imag).max()))

    @property
    def mean_residual(self) -> float:
        return float(np.mean(np.abs(np.r_[self.res_real, self.res_imag])))

    @property
    def verdict(self) -> tuple[str, str]:
        m = self.max_residual
        if m < 0.01:
            return "Valid", "green"
        if m < 0.05:
            return "Acceptable", "orange"
        return "Questionable", "red"


def run_linkk(f, Z, c: float = 0.85, max_M: int = 50, fit_type: str = "complex",
              add_cap: bool = False, mode: str = "auto", M: int | None = None
              ) -> KKResult:
    """Lin-KK test.

    ``schonleber`` - impedance.linKK: stop at the first M with μ < c.
    ``manual``     - fixed number of RC elements ``M``.
    ``auto``       - scan M = 1…max_M. At low M the under-fit itself produces
                     negative R_k, so μ dips below c and the library loop can
                     stop far too early; μ also oscillates with M. Here we only
                     consider M once χ² is within 100× of its best value (past
                     the under-fit region), then take the first M where μ < c
                     holds for two consecutive M (onset of over-fit, Schönleber's
                     criterion). If that never happens, the M with the lowest χ²
                     is used.
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    max_M = int(min(max_M, max(2, f.size - 2)))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(), \
            np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        if mode == "schonleber":
            Mf, mu, Z_fit, rr, ri = linKK(f, Z, c=c, max_M=max_M,
                                          fit_type=fit_type, add_cap=add_cap)
            return KKResult(int(Mf), float(mu), np.asarray(Z_fit),
                            np.asarray(rr), np.asarray(ri))
        if mode == "manual":
            Mf, mu, Z_fit, rr, ri = linKK(f, Z, c=None, max_M=int(M or max_M),
                                          fit_type=fit_type, add_cap=add_cap)
            return KKResult(int(Mf), float(mu), np.asarray(Z_fit),
                            np.asarray(rr), np.asarray(ri))

        rows, fits = [], {}
        for m in range(1, max_M + 1):
            ts = _kk.get_tc_distribution(f, m)
            try:
                el, mu = _kk.fit_linKK(f, ts, m, Z, fit_type, add_cap)
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                continue
            Zf = _eval_linKK(el, ts, f)
            r = (Z - Zf) / np.abs(Z)
            chi2 = float(np.sum(r.real ** 2 + r.imag ** 2))
            rows.append({"M": m, "mu": float(mu), "chi2": chi2})
            fits[m] = (float(mu), Zf, r)
    if not rows:
        raise ValueError("Lin-KK could not be computed for this spectrum.")
    scan = pd.DataFrame(rows).reset_index(drop=True)
    converged = scan.chi2 <= 100 * scan.chi2.min()
    start = int(converged.idxmax())
    chosen = None
    for i in range(start, len(scan) - 1):
        if scan.mu[i] < c and scan.mu[i + 1] < c:
            chosen = int(scan.M[i])
            break
    if chosen is None:
        chosen = int(scan.loc[scan.chi2.idxmin(), "M"])
    scan["chosen"] = scan.M == chosen
    mu, Zf, r = fits[chosen]
    return KKResult(chosen, mu, Zf, r.real, r.imag, scan)


# --------------------------------------------------------------------------
# Equivalent-circuit fitting
# --------------------------------------------------------------------------
@dataclass
class FitConfig:
    circuit: str
    table: pd.DataFrame                    # output of parameter_table (edited)
    weighting: str = "modulus"             # "modulus" | "none"
    global_opt: bool = False
    niter: int = 50
    max_nfev: int = 100_000

    def to_dict(self) -> dict:
        return {"circuit": self.circuit, "weighting": self.weighting,
                "global_opt": self.global_opt, "niter": self.niter,
                "parameters": self.table.to_dict(orient="records")}


@dataclass
class FitResult:
    name: str
    circuit: str
    names: list[str]
    units: list[str]
    values: dict[str, float]               # all parameters (fitted + fixed)
    errors: dict[str, float]               # 1σ; nan for fixed / unavailable
    fixed: dict[str, float]
    f: np.ndarray
    Z: np.ndarray
    Z_fit: np.ndarray
    unit: str = "Ω"
    config: dict = field(default_factory=dict)

    # ------------------------------------------------------------ metrics
    @property
    def res_real(self) -> np.ndarray:
        return (self.Z.real - self.Z_fit.real) / np.abs(self.Z)

    @property
    def res_imag(self) -> np.ndarray:
        return (self.Z.imag - self.Z_fit.imag) / np.abs(self.Z)

    @property
    def rmse(self) -> float:
        d = np.r_[self.Z.real - self.Z_fit.real, self.Z.imag - self.Z_fit.imag]
        return float(np.sqrt(np.mean(d ** 2)))

    @property
    def chi2(self) -> float:
        """Modulus-weighted χ² per degree of freedom (Boukamp)."""
        n_free = len(self.values) - len(self.fixed)
        dof = max(2 * self.f.size - n_free, 1)
        return float(np.sum(self.res_real ** 2 + self.res_imag ** 2) / dof)

    @property
    def mean_rel_error(self) -> float:
        return float(np.mean(np.abs(self.Z - self.Z_fit) / np.abs(self.Z)) * 100)

    def smooth(self, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
        """Model evaluated on a dense log grid spanning the data (for plots)."""
        fg = np.logspace(np.log10(self.f.max()), np.log10(self.f.min()), n)
        return fg, simulate(self.circuit, self.values, fg)

    # ------------------------------------------------------------ tables
    def param_frame(self) -> pd.DataFrame:
        rows = []
        for n, u in zip(self.names, self.units):
            v, e = self.values[n], self.errors.get(n, np.nan)
            rel = abs(e / v) * 100 if (np.isfinite(e) and v) else np.nan
            rows.append({"Parameter": n, "Value": v, "± 1σ": e,
                         "Error %": rel, "Unit": self.scale_unit(u),
                         "Status": "fixed" if n in self.fixed else "fitted"})
        return pd.DataFrame(rows)

    def scale_unit(self, u: str) -> str:
        if self.unit == "Ω·cm²" and u.startswith("Ohm") and not u.startswith("Ohm^-1"):
            u = u.replace("Ohm", "Ohm cm^2", 1)
        return pretty_unit(u)

    def derived(self) -> pd.DataFrame:
        return derived_quantities(self.circuit, self.values)

    def curve_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "freq_Hz": self.f,
            "Z_real": self.Z.real, "Z_imag": self.Z.imag,
            "Zfit_real": self.Z_fit.real, "Zfit_imag": self.Z_fit.imag,
            "resid_real_pct": self.res_real * 100,
            "resid_imag_pct": self.res_imag * 100,
        })

    def model_json(self) -> str:
        """impedance.py-compatible model file (CustomCircuit.load)."""
        free = [n for n in self.names if n not in self.fixed]
        return json.dumps({
            "Name": self.name,
            "Circuit String": self.circuit,
            "Initial Guess": [self.config.get("initial", {}).get(n, self.values[n])
                              for n in free],
            "Constants": self.fixed,
            "Fit": True,
            "Parameters": [self.values[n] for n in free],
            "Confidence": [self.errors.get(n, float("nan")) for n in free],
            "EIS Studio": {"rmse": self.rmse, "chi2": self.chi2,
                           "unit": self.unit, "config": self.config},
        }, indent=2, default=float)

    def summary_row(self) -> dict:
        row = {"File": self.name, "Circuit": self.circuit}
        for n in self.names:
            row[n] = self.values[n]
            row[f"{n} ±"] = self.errors.get(n, np.nan)
        row.update({"RMSE": self.rmse, "χ²": self.chi2,
                    "Mean |ΔZ|/|Z| %": self.mean_rel_error})
        return row


def fit_circuit(name: str, f, Z, cfg: FitConfig, unit: str = "Ω",
                initial_override: dict[str, float] | None = None) -> FitResult:
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    circuit = cfg.circuit.replace(" ", "")
    params = parse_circuit(circuit)
    tbl = cfg.table.set_index("Parameter")
    names = [p.name for p in params]
    units = [p.unit for p in params]
    missing = [n for n in names if n not in tbl.index]
    if missing:
        raise CircuitError(f"Parameter table is missing {', '.join(missing)}.")

    fixed = {n: float(tbl.loc[n, "Initial"]) for n in names if bool(tbl.loc[n, "Fixed"])}
    free = [n for n in names if n not in fixed]
    if not free:
        raise CircuitError("All parameters are fixed; unfix at least one.")

    lo = np.array([float(tbl.loc[n, "Lower"]) for n in free])
    hi = np.array([float(tbl.loc[n, "Upper"]) for n in free])
    lo = np.where(np.isfinite(lo), lo, -np.inf)
    hi = np.where(np.isnan(hi), np.inf, hi)
    if np.any(lo >= hi):
        bad = [n for n, a, b in zip(free, lo, hi) if a >= b]
        raise CircuitError(f"Lower bound ≥ upper bound for {', '.join(bad)}.")
    start = {n: float(tbl.loc[n, "Initial"]) for n in free}
    if initial_override:
        start.update({k: float(v) for k, v in initial_override.items() if k in start})
    x0 = []
    for n, a, b in zip(free, lo, hi):
        v = float(np.clip(start[n], a, b))
        width = (b - a) if np.isfinite(b - a) else max(abs(v), 1.0)
        if v <= a:                       # keep strictly inside (Q=0 → ÷0)
            v = a + 1e-6 * width
        if v >= b:
            v = b - 1e-6 * width
        x0.append(float(v))

    model = CustomCircuit(circuit, initial_guess=x0, constants=fixed, name=name)
    kwargs: dict = {}
    if cfg.global_opt:
        kwargs["global_opt"] = True
        kwargs["niter"] = int(cfg.niter)
    else:
        kwargs["maxfev"] = int(cfg.max_nfev)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        model.fit(f, Z, bounds=(lo.tolist(), hi.tolist()),
                  weight_by_modulus=(cfg.weighting == "modulus" and not cfg.global_opt),
                  **kwargs)
    popt = np.asarray(model.parameters_, float)
    conf = model.conf_ if model.conf_ is not None else [np.nan] * len(popt)
    values = dict(fixed)
    errors = {}
    for n, v, e in zip(free, popt, conf):
        values[n] = float(v)
        errors[n] = float(e) if e is not None and np.isfinite(e) else float("nan")
    for n in fixed:
        errors[n] = float("nan")
    values = {n: values[n] for n in names}
    Z_fit = simulate(circuit, values, f)
    config = cfg.to_dict()
    config["initial"] = dict(zip(free, x0))
    return FitResult(name, circuit, names, units, values, errors, fixed,
                     f, Z, Z_fit, unit, config)


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------
def _parallel_groups(circuit: str) -> list[list[str]]:
    """Top-level element lists of every p(...) group (leaf elements only)."""
    groups, stack = [], []
    i = 0
    while i < len(circuit):
        if circuit.startswith("p(", i):
            stack.append(i + 2)
            i += 2
            continue
        if circuit[i] == ")" and stack:
            start = stack.pop()
            inner = circuit[start:i]
            depth, branches, cur = 0, [], ""
            for ch in inner:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                if ch == "," and depth == 0:
                    branches.append(cur)
                    cur = ""
                else:
                    cur += ch
            branches.append(cur)
            groups.append(branches)
        i += 1
    return groups


def derived_quantities(circuit: str, values: dict[str, float]) -> pd.DataFrame:
    """Effective capacitance and peak frequency for R‖C and R‖CPE pairs."""
    rows = []
    for branches in _parallel_groups(circuit.replace(" ", "")):
        simple = [b for b in branches if re.fullmatch(r"[A-Za-z]+\d*(_\d+)?", b)]
        rs = [b for b in simple if get_element_from_name(b) == "R"]
        cs = [b for b in simple if get_element_from_name(b) in ("C", "CPE")]
        if len(rs) != 1 or len(cs) != 1 or len(branches) != 2:
            continue
        r, c = rs[0], cs[0]
        R = values.get(r)
        if get_element_from_name(c) == "C":
            C = values.get(c)
            if R and C:
                rows.append({"Pair": f"{r} ‖ {c}", "R": R, "C_eff (F)": C,
                             "f_peak (Hz)": 1 / (2 * np.pi * R * C),
                             "τ (s)": R * C})
        else:
            Q, a = values.get(f"{c}_0"), values.get(f"{c}_1")
            if R and Q and a:
                tau = (R * Q) ** (1 / a)
                rows.append({"Pair": f"{r} ‖ {c}", "R": R,
                             "C_eff (F)": tau / R,          # Brug et al. (1984)
                             "f_peak (Hz)": 1 / (2 * np.pi * tau),
                             "τ (s)": tau})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Batch helpers
# --------------------------------------------------------------------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def number_in_name(s: str) -> float | None:
    nums = re.findall(r"[-+]?\d*\.?\d+", s.rsplit(".", 1)[0])
    return float(nums[-1]) if nums else None
