"""Data loading for EIS Studio.

Instrument files are parsed with the readers in ``impedance.preprocessing``
(impedance.py). Uploaded bytes are written to a temporary file because those
readers expect a path. ``auto`` mode guesses the instrument from the file
extension and header, then falls back to trying every reader, and finally to
the generic table importer.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from impedance import preprocessing

INSTRUMENTS: dict[str, str] = {
    "auto": "Auto-detect",
    "csv": "CSV  (f, Z′, Z″)",
    "gamry": "Gamry  (.DTA)",
    "biologic": "BioLogic EC-Lab  (.mpt)",
    "autolab": "Metrohm Autolab  (.txt)",
    "parstat": "Parstat  (.txt)",
    "zplot": "Scribner ZPlot  (.z)",
    "versastudio": "VersaStudio  (.par)",
    "powersuite": "PowerSuite  (.txt)",
    "chinstruments": "CH Instruments  (.txt)",
    "generic": "Generic table  (choose columns)",
}

UPLOAD_TYPES = ["csv", "txt", "dta", "mpt", "z", "par", "dat", "tsv"]

_READER_ORDER = ["csv", "gamry", "biologic", "zplot", "chinstruments",
                 "autolab", "parstat", "powersuite", "versastudio"]

_NUM = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


@dataclass
class Spectrum:
    """One impedance spectrum. ``Z`` uses the Z′ + jZ″ convention (Z″ < 0
    for capacitive behaviour)."""

    name: str
    f: np.ndarray
    Z: np.ndarray
    source: str
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.f.size)


@dataclass
class TableMapping:
    """Column mapping for the generic table importer."""

    f_col: str
    a_col: str
    b_col: str
    mode: str = "re_im"          # "re_im" or "mod_phase"
    negate_b: bool = False       # column holds −Z″ (or −phase)


class LoadError(Exception):
    """Raised when a file cannot be parsed into a spectrum."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean(f, Z) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(f, dtype=float).ravel()
    Z = np.asarray(Z, dtype=complex).ravel()
    ok = np.isfinite(f) & np.isfinite(Z.real) & np.isfinite(Z.imag) & (f > 0)
    f, Z = f[ok], Z[ok]
    order = np.argsort(f)[::-1]          # high → low frequency
    return f[order], Z[order]


def guess_instrument(filename: str, text: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    head = text[:4000]
    if ext == ".dta" or "ZCURVE" in text:
        return "gamry"
    if ext == ".mpt" or head.startswith("EC-Lab ASCII FILE"):
        return "biologic"
    if ext == ".z":
        return "zplot"
    if ext == ".par":
        return "versastudio"
    lines = head.splitlines()
    if any(line.startswith("Freq/Hz") for line in lines):
        return "chinstruments"
    first = lines[0] if lines else ""
    if "Potential (V)" in first and "Frequency (Hz)" in first:
        return "parstat"
    if first.strip().startswith("Frequency") and "\t" in first:
        return "powersuite"
    if "Z60W" in first or ("Data File" in first and "Freq" in text):
        return "autolab"
    if ext == ".csv":
        return "csv"
    return "generic"


def _read_with(instrument: str, path: str) -> tuple[np.ndarray, np.ndarray]:
    # readers print nothing, but guard anyway
    with contextlib.redirect_stdout(io.StringIO()):
        if instrument == "csv":
            f, Z = preprocessing.readCSV(path)
        else:
            f, Z = preprocessing.readFile(path, instrument=instrument)
    return _clean(f, Z)


def _is_valid(f: np.ndarray) -> bool:
    return f.size >= 3


# --------------------------------------------------------------------------
# generic table import
# --------------------------------------------------------------------------
def _split(line: str, sep: str | None) -> list[str]:
    if sep is None:
        return line.split()
    return [t.strip() for t in line.split(sep)]


def read_table(raw: bytes) -> pd.DataFrame:
    """Parse any delimited numeric table. Finds the first block of numeric
    rows, detects the delimiter and uses the line above as header if it has
    the same number of fields."""
    text = decode(raw)
    lines = text.splitlines()
    start, sep = None, None
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        for cand in ("\t", ",", ";", None):
            toks = [t for t in _split(line, cand) if t != ""]
            if len(toks) >= 3 and sum(bool(_NUM.match(t)) for t in toks) >= 3 \
                    and all(_NUM.match(t) for t in toks[:3]):
                start, sep = i, cand
                break
        if start is not None:
            break
    if start is None:
        raise LoadError("No numeric table with at least 3 columns was found.")

    first = _split(lines[start], sep)
    ncol = len(first)
    header = None
    for j in range(start - 1, max(start - 4, -1), -1):
        if not lines[j].strip():
            continue
        cand = [t.strip().strip('"') for t in _split(lines[j], sep)]
        if len(cand) == ncol and not all(_NUM.match(t) for t in cand if t):
            header = cand
        break
    rows = []
    for line in lines[start:]:
        if not line.strip():
            continue
        toks = _split(line, sep)
        if len(toks) < 3:
            break
        rows.append(toks[:ncol] + [""] * (ncol - len(toks)))
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    names = header or [f"col {k + 1}" for k in range(ncol)]
    # de-duplicate / fill blank names
    seen: dict[str, int] = {}
    clean = []
    for k, nm in enumerate(names):
        nm = nm or f"col {k + 1}"
        if nm in seen:
            seen[nm] += 1
            nm = f"{nm} ({seen[nm]})"
        else:
            seen[nm] = 0
        clean.append(nm)
    df.columns = clean
    return df.dropna(how="all")


def suggest_mapping(df: pd.DataFrame) -> TableMapping:
    """Best-effort guess of frequency / real / imaginary columns by name."""
    cols = list(df.columns)
    low = [c.lower() for c in cols]

    def find(*keys, exclude=()):
        for k in keys:
            for c, lc in zip(cols, low):
                if k in lc and not any(e in lc for e in exclude):
                    return c
        return None

    f_col = find("freq", "hz") or cols[0]
    mod = find("|z|", "mod", "zmod")
    phase = find("phase", "phz", "deg")
    re_col = find("re(z)", "zre", "z'", "zreal", "real", "re ", exclude=("|z|",))
    im_col = find("-im", "im(z)", "zim", 'z"', "z''", "zimag", "imag")
    if re_col and im_col and re_col != im_col:
        neg = im_col.lower().startswith("-") or "-im" in im_col.lower()
        return TableMapping(f_col, re_col, im_col, "re_im", neg)
    if mod and phase:
        return TableMapping(f_col, mod, phase, "mod_phase", False)
    rest = [c for c in cols if c != f_col]
    return TableMapping(f_col, rest[0], rest[1], "re_im", False)


def spectrum_from_table(df: pd.DataFrame, m: TableMapping
                        ) -> tuple[np.ndarray, np.ndarray]:
    f = df[m.f_col].to_numpy(dtype=float)
    a = df[m.a_col].to_numpy(dtype=float)
    b = df[m.b_col].to_numpy(dtype=float)
    if m.negate_b:
        b = -b
    if m.mode == "mod_phase":
        Z = a * np.exp(1j * np.deg2rad(b))
    else:
        Z = a + 1j * b
    return _clean(f, Z)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def load_spectrum(name: str, raw: bytes, instrument: str = "auto",
                  mapping: TableMapping | None = None) -> Spectrum:
    """Parse uploaded bytes into a :class:`Spectrum`."""
    text = decode(raw)

    if instrument == "generic":
        df = read_table(raw)
        m = mapping or suggest_mapping(df)
        if mapping is not None and not {m.f_col, m.a_col, m.b_col} <= set(df.columns):
            m = suggest_mapping(df)
        f, Z = spectrum_from_table(df, m)
        if not _is_valid(f):
            raise LoadError("Fewer than 3 valid points after parsing.")
        return Spectrum(name, f, Z, "generic",
                        {"mapping": m.__dict__, "columns": list(df.columns)})

    if instrument == "auto":
        first = guess_instrument(name, text)
        order = [first] + [r for r in _READER_ORDER if r != first]
    else:
        order = [instrument]

    suffix = os.path.splitext(name)[1] or ".txt"
    fd, path = tempfile.mkstemp(suffix=suffix)
    errors = []
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        for inst in order:
            if inst == "generic":
                continue
            try:
                f, Z = _read_with(inst, path)
            except Exception as exc:  # reader-specific parse failures
                errors.append(f"{inst}: {type(exc).__name__}")
                continue
            if _is_valid(f):
                return Spectrum(name, f, Z, inst)
            errors.append(f"{inst}: no valid points")
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)

    if instrument == "auto":
        try:
            df = read_table(raw)
            m = mapping or suggest_mapping(df)
            f, Z = spectrum_from_table(df, m)
            if _is_valid(f):
                return Spectrum(name, f, Z, "generic",
                                {"mapping": m.__dict__,
                                 "columns": list(df.columns)})
        except Exception as exc:
            errors.append(f"generic: {exc}")

    raise LoadError("Could not read this file. Tried " + "; ".join(errors))


def spectrum_to_csv(f: np.ndarray, Z: np.ndarray) -> bytes:
    df = pd.DataFrame({"freq_Hz": f, "Z_real": Z.real, "Z_imag": Z.imag})
    return df.to_csv(index=False).encode()
