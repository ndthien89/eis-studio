"""EIS Studio - electrochemical impedance analysis in the browser.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from eis_studio import __version__
from eis_studio import analysis as an
from eis_studio import citations as cite
from eis_studio import drt as drt
from eis_studio import io as eio
from eis_studio import plots as pl
from eis_studio import ui

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"
SERIES = ["Circuit1_EIS_1.z", "Circuit1_EIS_2.z", "Circuit2_EIS_1.z",
          "Circuit2_EIS_2.z", "Circuit3_EIS_1.z", "Circuit3_EIS_2.z"]

st.set_page_config(
    page_title="EIS Studio",
    page_icon=":material/ssid_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": f"EIS Studio {__version__} - a Streamlit front end for "
                 "impedance.py (Murbach et al., JOSS 2020).",
        "Get help": "https://impedancepy.readthedocs.io",
    },
)
ui.inject_css()

ss = st.session_state
for key, default in {
    "raw": {},            # name -> {"bytes": b, "origin": "upload" | "example"}
    "excluded": {},       # name -> tuple of excluded point indices
    "fits": {},           # name -> FitResult (single mode)
    "drt": {},            # name -> DRTResult (single mode)
    "batch": None,        # dict with results / errors / kk / config
    "batch_drt": None,    # dict with DRT results / errors for batch mode
    "load_errors": {},    # name -> message
    "uploader_key": 0,
}.items():
    ss.setdefault(key, default)


# ==========================================================================
# cached workers
# ==========================================================================
@st.cache_data(show_spinner=False, max_entries=512)
def parse(name: str, raw: bytes, instrument: str, mapping: tuple | None):
    m = eio.TableMapping(*mapping) if mapping else None
    return eio.load_spectrum(name, raw, instrument, m)


@st.cache_data(show_spinner=False, max_entries=64)
def table_preview(raw: bytes) -> pd.DataFrame:
    return eio.read_table(raw)


@st.cache_data(show_spinner=False, max_entries=256)
def kk_cached(f, Z, c, max_M, fit_type, add_cap, mode, M):
    return an.run_linkk(f, Z, c=c, max_M=max_M, fit_type=fit_type,
                        add_cap=add_cap, mode=mode, M=M)


def show(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, theme="streamlit", config=pl.CONFIG, key=key)


def load_examples(names: list[str]) -> None:
    for n in names:
        p = SAMPLE_DIR / n
        if p.exists():
            ss.raw[n] = {"bytes": p.read_bytes(), "origin": "example"}


def clear_all() -> None:
    ss.raw, ss.excluded, ss.fits, ss.batch, ss.load_errors = {}, {}, {}, None, {}
    ss.drt, ss.batch_drt = {}, None
    ss.uploader_key += 1


@st.cache_resource(show_spinner=False)
def drt_availability() -> drt.Availability:
    return drt.availability()


@st.cache_data(show_spinner=False, max_entries=128)
def drt_cached(name, f, Z, opt: drt.DRTOptions, unit):
    return drt.compute_drt(name, f, Z, opt, unit)


# ==========================================================================
# sidebar - library, format, preprocessing
# ==========================================================================
with st.sidebar:
    ui.brand(__version__)

    files = st.file_uploader(
        "Impedance files", type=eio.UPLOAD_TYPES, accept_multiple_files=True,
        key=f"uploader_{ss.uploader_key}",
        help="Gamry .DTA, BioLogic .mpt, ZPlot .z, VersaStudio .par, Autolab / "
             "Parstat / PowerSuite / CH Instruments .txt, or any CSV / text table.")
    current = {uf.name: uf.getvalue() for uf in files or []}
    for n in [n for n, r in ss.raw.items() if r["origin"] == "upload" and n not in current]:
        del ss.raw[n]
    for n, b in current.items():
        if n not in ss.load_errors:
            ss.raw[n] = {"bytes": b, "origin": "upload"}

    with st.popover("Example data", icon=":material/science:", width="stretch"):
        st.caption("Datasets shipped with impedance.py (MIT licence).")
        if st.button("Load a 6-file series", icon=":material/stacks:", width="stretch",
                     help="Three circuits × two repeats, ideal for batch mode."):
            load_examples(SERIES)
            st.rerun()
        picks = st.multiselect("Or pick files", sorted(p.name for p in SAMPLE_DIR.iterdir()
                                                       if p.is_file()),
                               placeholder="Choose example files")
        if st.button("Load selected", disabled=not picks, width="stretch"):
            load_examples(picks)
            st.rerun()

    instrument = st.selectbox("File format", list(eio.INSTRUMENTS),
                              format_func=eio.INSTRUMENTS.get, key="instrument",
                              help="Auto-detect reads the extension and header, then "
                                   "tries every reader in turn.")
    if ss.get("_last_instrument") != instrument:      # new format → retry failed files
        ss.load_errors = {}
        ss._last_instrument = instrument
        for n, b in current.items():
            ss.raw[n] = {"bytes": b, "origin": "upload"}

    mapping_key = None
    uploads = [n for n, r in ss.raw.items() if r["origin"] == "upload"]
    if instrument == "generic" and uploads:
        with st.container(border=True):
            st.caption("COLUMN MAPPING  ·  applied to all uploaded files")
            try:
                ref = table_preview(ss.raw[uploads[0]]["bytes"])
                cols = list(ref.columns)
                sug = eio.suggest_mapping(ref)
                for k, v in (("map_f", sug.f_col), ("map_a", sug.a_col), ("map_b", sug.b_col)):
                    if ss.get(k) not in cols:
                        ss[k] = v
                ss.setdefault("map_mode", "Z′ / Z″" if sug.mode == "re_im" else "|Z| / phase")
                ss.setdefault("map_neg", sug.negate_b)
                mode = st.segmented_control("Columns hold", ["Z′ / Z″", "|Z| / phase"],
                                            key="map_mode", required=True)
                a_lbl, b_lbl = ("Z′", "Z″") if mode == "Z′ / Z″" else ("|Z|", "Phase (°)")
                fcol = st.selectbox("Frequency", cols, key="map_f")
                acol = st.selectbox(a_lbl, cols, key="map_a")
                bcol = st.selectbox(b_lbl, cols, key="map_b")
                neg = st.toggle("Column is negated (−Z″ / −phase)", key="map_neg")
                mapping_key = (fcol, acol, bcol,
                               "re_im" if mode == "Z′ / Z″" else "mod_phase", neg)
                with st.expander("Preview", icon=":material/table:"):
                    st.dataframe(ref.head(8), hide_index=True, height=240)
            except eio.LoadError as exc:
                st.error(str(exc))

    # ---- parse everything (cached) -----------------------------------
    spectra: dict[str, eio.Spectrum] = {}
    for n, r in list(ss.raw.items()):
        inst = instrument if r["origin"] == "upload" else "auto"
        try:
            spectra[n] = parse(n, r["bytes"], inst,
                               mapping_key if inst == "generic" else None)
        except Exception as exc:  # parse failure → report once, drop file
            ss.load_errors[n] = str(exc)
            del ss.raw[n]
    for n in list(ss.load_errors):
        if n not in current:
            ss.load_errors.pop(n)
    for n, msg in ss.load_errors.items():
        st.error(f"**{n}** - {msg}\n\nTry *Generic table* format.",
                 icon=":material/error:")

    with st.expander("Preprocessing", icon=":material/tune:"):
        st.caption("Applied to every spectrum before analysis.")
        c1, c2 = st.columns(2)
        fmin = c1.number_input("f min (Hz)", min_value=0.0, value=0.0, format="%g",
                               help="0 = no lower limit")
        fmax = c2.number_input("f max (Hz)", min_value=0.0, value=0.0, format="%g",
                               help="0 = no upper limit")
        drop_ind = st.toggle("Remove inductive points (Z″ > 0)",
                             help="Same as impedance.preprocessing.ignoreBelowX.")
        flip = st.toggle("File stores −Z″ (flip sign)",
                         help="Use if your semicircles appear below the axis.")
        use_area = st.toggle("Normalise by electrode area")
        area = st.number_input("Area (cm²)", min_value=1e-6, value=1.0, format="%g",
                               disabled=not use_area)

    prep_base = dict(fmin=fmin or None, fmax=fmax or None, drop_inductive=drop_ind,
                     flip_imag=flip, area_cm2=area if use_area else None)
    UNIT = "Ω·cm²" if use_area else "Ω"

    if spectra:
        ui.section("Library", f"{len(spectra)} spectra")
        for n, sp in spectra.items():
            ui.file_row(n, f"{sp.n} pts · {sp.source}")
        st.button("Clear all", icon=":material/delete:", on_click=clear_all,
                  type="tertiary")

    with st.expander("About & citations", icon=":material/info:"):
        st.caption("EIS STUDIO IS BUILT ON")
        for key, s in cite.SOFTWARE.items():
            st.markdown(f"**[{key}]({s['url']})** — {s['role']}.  \n"
                        f"{s['authors']} · {s['license']} · {s['version']}")
        st.caption("PLEASE CITE")
        st.markdown("\n".join(f"- {r}" for r in
                              [s["paper"] for s in cite.SOFTWARE.values()]
                              + [cite.METHODS["lin_kk"], cite.METHODS["brug"]]))


def prepared(name: str) -> tuple[np.ndarray, np.ndarray]:
    sp = spectra[name]
    # the exclusion widget's own state is newest; fall back to the stored copy
    excl = ss.get(f"excl_{name}", ss.excluded.get(name, ()))
    ss.excluded[name] = tuple(excl)
    opt = an.Preprocess(**prep_base, exclude=tuple(excl))
    return an.preprocess(sp.f, sp.Z, opt)


# ==========================================================================
# shared components
# ==========================================================================
def circuit_setup(prefix: str, f: np.ndarray, Z: np.ndarray,
                  fitted: an.FitResult | None = None) -> tuple[str | None, pd.DataFrame | None]:
    """Preset picker, circuit string, parameter table. Returns (circuit, table)."""
    ck, tk, vk, ek = (f"{prefix}_circuit", f"{prefix}_tbl", f"{prefix}_ver",
                      f"{prefix}_edited")
    ss.setdefault(ck, an.PRESETS["Randles  (Wo, CPE)"]["circuit"])
    ss.setdefault(vk, 0)

    def on_preset():
        p = ss[f"{prefix}_preset"]
        if p in an.PRESETS:
            ss[ck] = an.PRESETS[p]["circuit"]

    presets = list(an.PRESETS) + ["Custom"]
    match = next((k for k, v in an.PRESETS.items() if v["circuit"] == ss[ck]), "Custom")
    ss.setdefault(f"{prefix}_preset", match)
    st.selectbox("Model", presets, key=f"{prefix}_preset", on_change=on_preset)
    if ss[f"{prefix}_preset"] in an.PRESETS:
        ui.note(an.PRESETS[ss[f"{prefix}_preset"]]["about"])

    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    circuit = c1.text_input("Circuit string", key=ck,
                            placeholder="e.g. R0-p(R1,CPE1)-W1")
    with c2.popover("", icon=":material/help:", help="Element reference"):
        st.markdown("**Syntax** `R0-p(R1,CPE1)` → R0 in series with (R1 ‖ CPE1)")
        st.dataframe(pd.DataFrame({"Element": list(an.ELEMENT_INFO),
                                   "Meaning": list(an.ELEMENT_INFO.values())}),
                     hide_index=True)

    try:
        an.parse_circuit(circuit)
    except an.CircuitError as exc:
        st.error(str(exc), icon=":material/error:")
        return None, None

    circuit = circuit.replace(" ", "")
    if ss.get(f"{tk}_circ") != circuit or tk not in ss:
        ss[tk] = an.parameter_table(circuit, f, Z, previous=ss.get(ek))
        ss[f"{tk}_circ"] = circuit
        ss[vk] += 1

    def reestimate():
        ss[tk] = an.parameter_table(circuit, f, Z)
        ss[vk] += 1

    def use_fitted():
        t = ss[tk].copy()
        t["Initial"] = [fitted.values.get(p, v) for p, v in zip(t.Parameter, t.Initial)]
        ss[tk] = t
        ss[vk] += 1

    b1, b2 = st.columns(2)
    b1.button("Estimate from data", icon=":material/auto_fix_high:", on_click=reestimate,
              width="stretch", key=f"{prefix}_est")
    b2.button("Start from last fit", icon=":material/history:", on_click=use_fitted,
              width="stretch", key=f"{prefix}_usefit",
              disabled=fitted is None or fitted.circuit != circuit)

    bounds = st.toggle("Edit bounds", key=f"{prefix}_bounds",
                       help="Defaults: 0 … ∞, and 0 … 1 for CPE / Zarc exponents.")
    edited = st.data_editor(
        ss[tk], key=f"{prefix}_editor_{ss[vk]}", hide_index=True,
        disabled=["Parameter", "Unit"],
        column_order=["Parameter", "Initial", "Unit", "Fixed"]
        + (["Lower", "Upper"] if bounds else []),
        column_config={
            "Parameter": st.column_config.TextColumn(width="small"),
            "Unit": st.column_config.TextColumn(width="small"),
            "Initial": st.column_config.NumberColumn(format="%.4g", required=True),
            "Fixed": st.column_config.CheckboxColumn(help="Hold constant during the fit"),
            "Lower": st.column_config.NumberColumn(format="%.3g"),
            "Upper": st.column_config.NumberColumn(format="%.3g"),
        })
    ss[ek] = edited
    return circuit, edited


def optimizer_options(prefix: str) -> dict:
    with st.expander("Optimizer", icon=":material/settings:"):
        w = st.segmented_control("Weighting", ["Modulus |Z|", "Unweighted"],
                                 default="Modulus |Z|", required=True,
                                 key=f"{prefix}_w",
                                 help="Modulus weighting is the usual choice when "
                                      "measurement variances are unknown.")
        g = st.toggle("Global search (basin-hopping)", key=f"{prefix}_g",
                      help="Slower, less sensitive to starting values. "
                           "Uses unweighted RMSE.")
        n = st.slider("Basin-hopping iterations", 5, 300, 40, key=f"{prefix}_n",
                      disabled=not g)
    return dict(weighting="modulus" if w == "Modulus |Z|" else "none",
                global_opt=g, niter=n)


def fit_figures(res: an.FitResult, key: str, bode_mode: str = "magphase") -> None:
    fg, Zg = res.smooth()
    traces = [dict(name="Data", f=res.f, Z=res.Z, kind="data", color=pl.BLUE),
              dict(name="Fit", f=fg, Z=Zg, kind="fit", color=pl.ORANGE)]
    c1, c2 = st.columns(2)
    with c1:
        show(pl.nyquist(traces, UNIT, height=420), key=f"{key}_nyq")
    with c2:
        show(pl.bode(traces, UNIT, height=420, mode=bode_mode), key=f"{key}_bode")
    show(pl.residuals(res.f, res.res_real, res.res_imag, title="Fit residuals"),
         key=f"{key}_res")


def result_tables(res: an.FitResult) -> None:
    ui.tiles([
        ("χ² (modulus-weighted)", ui.fmt(res.chi2, 3), "per degree of freedom"),
        ("RMSE", ui.fmt(res.rmse, 3), UNIT),
        ("Mean |ΔZ| / |Z|", f"{res.mean_rel_error:.2f} %", None),
        ("Parameters", f"{len(res.names) - len(res.fixed)} fitted",
         f"{len(res.fixed)} fixed" if res.fixed else None),
    ])
    pf = res.param_frame()
    disp = pf.assign(**{
        "Value": pf["Value"].map(ui.fmt),
        "± 1σ": pf["± 1σ"].map(ui.fmt),
        "Error %": pf["Error %"].map(lambda v: "–" if v != v else f"{v:.2f}"),
    })
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(disp, hide_index=True)
    with c2:
        der = res.derived()
        if not der.empty:
            st.caption("DERIVED  ·  effective capacitance (Brug) and peak frequency")
            st.dataframe(der.assign(**{c: der[c].map(ui.fmt) for c in der.columns
                                       if c != "Pair"}), hide_index=True)
        bad = pf[(pf["Error %"] > 100) & (pf.Status == "fitted")]
        if not bad.empty:
            st.warning("Poorly determined: " + ", ".join(bad.Parameter) +
                       ". Consider fixing them or simplifying the circuit.",
                       icon=":material/warning:")


def single_downloads(res: an.FitResult, stem: str) -> None:
    c1, c2, c3 = st.columns(3)
    c1.download_button("Parameters (CSV)", res.param_frame().to_csv(index=False),
                       f"{stem}_parameters.csv", "text/csv",
                       icon=":material/table_view:", width="stretch")
    c2.download_button("Fitted curve (CSV)", res.curve_frame().to_csv(index=False),
                       f"{stem}_fit.csv", "text/csv",
                       icon=":material/show_chart:", width="stretch")
    c3.download_button("Model (JSON)", res.model_json(), f"{stem}_model.json",
                       "application/json", icon=":material/data_object:",
                       width="stretch",
                       help="Loads in impedance.py with CustomCircuit().load(path).")


def drt_controls(prefix: str, allow_hmc: bool = True) -> drt.DRTOptions | None:
    """Method + options for the DRT. Returns None if DRT is unavailable."""
    av = drt_availability()
    if not av.ridge:
        st.warning(av.message, icon=":material/extension_off:")
        return None
    methods = ["ridge", "map", "hmc"] if allow_hmc else ["ridge", "map"]
    with st.container(border=True):
        c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
        method = c1.segmented_control(
            "Solver", methods, format_func=drt.METHODS.get, default="ridge",
            required=True, key=f"{prefix}_method",
            help="Ridge: fast, no compiler needed. MAP: calibrated Bayesian estimate. "
                 "HMC: Bayesian sampling with a 95 % credible interval (slow).")
        nonneg = c2.toggle("Non-negative DRT", value=True, key=f"{prefix}_nonneg",
                           help="Constrain γ ≥ 0 (recommended unless you expect "
                                "inductive loops).")
        if method in ("map", "hmc") and not av.bayes:
            st.info(av.message, icon=":material/info:")
        opt = dict(method=method, nonneg=nonneg)
        with st.expander("Solver options", icon=":material/tune:"):
            if method == "ridge":
                opt["preset"] = st.segmented_control(
                    "Ridge preset", ["Huang", "Ciucci", "none"], default="Huang",
                    required=True, key=f"{prefix}_preset_r",
                    help="Huang: integral penalty, modulus weights. "
                         "Ciucci: discrete penalty, cross-validated λ.")
                opt["fit_inductance"] = st.toggle("Fit inductance", value=True,
                                                  key=f"{prefix}_ind")
            else:
                opt["outliers"] = st.segmented_control(
                    "Outlier-robust error model", ["auto", "on", "off"], default="auto",
                    required=True, key=f"{prefix}_out")
                opt["init_from_ridge"] = st.toggle("Initialise from ridge solution",
                                                   value=True, key=f"{prefix}_init")
                if method == "hmc":
                    a, b, c = st.columns(3)
                    opt["warmup"] = a.number_input("Warm-up", 50, 2000, 200, 50,
                                                   key=f"{prefix}_wu")
                    opt["samples"] = b.number_input("Samples", 50, 2000, 200, 50,
                                                    key=f"{prefix}_sa")
                    opt["chains"] = c.number_input("Chains", 1, 8, 2, key=f"{prefix}_ch")
            p1, p2 = st.columns(2)
            opt["find_peaks"] = p1.toggle("Decompose into peaks", value=True,
                                          key=f"{prefix}_peaks",
                                          help="Havriliak–Negami peak fit of the DRT.")
            opt["check_shoulders"] = p1.toggle(
                "Detect shoulder peaks", value=False, key=f"{prefix}_shoulders",
                disabled=not opt["find_peaks"],
                help="Also look for peaks hidden in the flank of a larger one. Can "
                     "split one broad process into several - check the result.")
            opt["R_rthresh"] = p2.slider("Min. peak size (% of Rp)", 0.1, 10.0, 0.5, 0.1,
                                         key=f"{prefix}_rth",
                                         disabled=not opt["find_peaks"]) / 100
            opt["prom_rthresh"] = opt["R_rthresh"]
    disabled = method in ("map", "hmc") and not av.bayes
    return None if disabled else drt.DRTOptions(**opt)


def cite_panel(drt_method: str | None = None, circuit: bool = False,
               lin_kk: bool = False) -> None:
    with st.expander("How to cite", icon=":material/format_quote:"):
        st.code(cite.citation_text(drt_method, circuit, lin_kk), language=None,
                wrap_lines=True)


def drt_results(res: drt.DRTResult, key: str, axis: str = "tau") -> None:
    for n in res.notes:
        st.info(n, icon=":material/info:")
    ui.tiles([
        ("R∞  (ohmic)", ui.fmt(res.R_inf, 4), res.unit),
        ("Rp  (polarisation)", ui.fmt(res.Rp, 4), res.unit),
        ("Peaks", f"{len(res.peaks)}", "Havriliak–Negami"),
        ("Inductance", ui.fmt(res.inductance, 3), "H"),
        ("Mean |ΔZ| / |Z|", f"{res.mean_rel_error:.2f} %", drt.METHOD_DETAIL[res.method]),
    ])
    show(pl.drt(res, axis=axis, title="Distribution of relaxation times"), key=f"{key}_g")
    if not res.peaks.empty:
        pk = res.peaks.copy()
        for c in ("R", "τ₀ (s)", "f_peak (Hz)"):
            pk[c] = pk[c].map(ui.fmt)
        for c in ("α (HN)", "β (HN)"):
            pk[c] = pk[c].map(lambda v: f"{v:.3f}")
        pk["Share of Rp %"] = pk["Share of Rp %"].map(lambda v: f"{v:.1f}")
        pk["Inside data range"] = pk["Inside data range"].map({True: "yes",
                                                               False: "no - extrapolated"})
        pk = pk.rename(columns={"R": f"R ({res.unit})"})
        st.dataframe(pk, hide_index=True)
    tr = [dict(name="Data", f=res.f, Z=res.Z, kind="data", color=pl.BLUE),
          dict(name="DRT model", f=res.f, Z=res.Z_fit, kind="fit", color=pl.ORANGE)]
    c1, c2 = st.columns(2)
    with c1:
        show(pl.nyquist(tr, res.unit, height=380, decade_labels=False,
                        title="DRT reconstruction"), key=f"{key}_nyq")
    with c2:
        show(pl.residuals(res.f, res.res_real, res.res_imag, height=380,
                          title="Residuals"), key=f"{key}_res")


def drt_view(name: str, f: np.ndarray, Z: np.ndarray) -> None:
    left, right = st.columns([5, 2], vertical_alignment="bottom")
    with left:
        opt = drt_controls("drt")
    with right:
        axis = st.segmented_control("x-axis", ["τ", "f"], default="τ", required=True,
                                    key="drt_axis", help="Plot against time constant "
                                                         "or frequency (f = 1/2πτ).")
        run = st.button("Compute DRT", type="primary", icon=":material/bar_chart:",
                        width="stretch", disabled=opt is None, key="drt_go")
    if run and opt is not None:
        slow = opt.method != "ridge"
        msg = ("Compiling and running Stan (first run compiles the model, ~1 min)…"
               if slow else "Inverting…")
        with st.spinner(msg):
            try:
                ss.drt[name] = drt_cached(name, f, Z, opt, UNIT)
            except Exception as exc:
                st.error(f"DRT failed: {exc}", icon=":material/error:")
    res = ss.drt.get(name)
    if res is None:
        ui.note("The DRT resolves overlapping processes by their time constant without "
                "assuming a circuit. Start with Ridge; use Bayesian HMC when you need "
                "uncertainty bands.")
        return
    if res.f.size != f.size:
        ui.note("Data changed since this DRT was computed — press Compute DRT to update.")
    drt_results(res, "drt", axis="tau" if axis == "τ" else "freq")
    stem = Path(name).stem
    c1, c2, c3 = st.columns(3)
    c1.download_button("DRT γ(τ) (CSV)", res.distribution_frame().to_csv(index=False),
                       f"{stem}_drt.csv", "text/csv", icon=":material/bar_chart:",
                       width="stretch")
    c2.download_button("Peaks (CSV)", res.peaks.to_csv(index=False), f"{stem}_drt_peaks.csv",
                       "text/csv", icon=":material/table_view:", width="stretch",
                       disabled=res.peaks.empty)
    c3.download_button("Reconstruction (CSV)", res.fit_frame().to_csv(index=False),
                       f"{stem}_drt_fit.csv", "text/csv", icon=":material/show_chart:",
                       width="stretch")
    cite_panel(drt_method=res.method)


# ==========================================================================
# main canvas
# ==========================================================================
ui.hero("Electrochemical impedance", "EIS Studio",
        "Load spectra, check them with Kramers–Kronig, and fit equivalent "
        "circuits — one file at a time or a whole series.")

if not spectra:
    ui.empty_state("Add impedance data to begin",
                   "Drop instrument files into the sidebar, or start with the example "
                   "series that ships with impedance.py.",
                   ["1  Load", "2  Validate", "3  Fit", "4  Export"])
    _, mid, _ = st.columns([2, 1.2, 2])
    if mid.button("Try example data", type="primary", icon=":material/play_arrow:",
                  width="stretch"):
        load_examples(SERIES + ["exampleData.csv"])
        st.rerun()
    st.stop()

mode = st.segmented_control("Mode", ["Single spectrum", "Batch"], default="Single spectrum",
                            required=True, key="mode", label_visibility="collapsed")


# --------------------------------------------------------------------------
# SINGLE SPECTRUM
# --------------------------------------------------------------------------
def single_view() -> None:
    names = list(spectra)
    c1, c2 = st.columns([3, 4], vertical_alignment="bottom")
    name = c1.selectbox("Spectrum", names, key="single_name")
    view = c2.segmented_control("View", ["Data", "Kramers–Kronig", "Circuit fit", "DRT"],
                                default="Data", required=True, key="single_view",
                                label_visibility="collapsed")
    f, Z = prepared(name)
    if f.size < 3:
        st.warning("Fewer than 3 points remain after preprocessing. Widen the "
                   "frequency window or restore points.", icon=":material/warning:")
        return
    s = an.summary_stats(f, Z)

    # ---------------------------------------------------------------- data
    if view == "Data":
        ui.tiles([
            ("Points", f"{s['n']}", f"{spectra[name].n - s['n']} removed"
             if spectra[name].n != s["n"] else spectra[name].source),
            ("Frequency range", f"{ui.fmt_freq(s['f_min'])} – {ui.fmt_freq(s['f_max'])}",
             f"{np.log10(s['f_max'] / s['f_min']):.1f} decades"),
            ("High-freq. intercept", ui.fmt(s["r_hf"], 4), UNIT),
            ("|Z| at f min", ui.fmt(s["z_lf"], 4), UNIT),
            ("Peak −Z″", ui.fmt_freq(s["f_peak"]), "characteristic frequency"),
        ])
        o1, o2, _ = st.columns([2, 2, 3])
        bmode = o1.segmented_control("Bode", ["|Z| · phase", "Z′ · Z″"],
                                     default="|Z| · phase", required=True, key="bode_mode",
                                     label_visibility="collapsed")
        labels = o2.toggle("Frequency labels", value=True, key="decades")
        tr = [dict(name=name, f=f, Z=Z, kind="data", color=pl.BLUE)]
        c1, c2 = st.columns(2)
        with c1:
            show(pl.nyquist(tr, UNIT, decade_labels=labels), key="d_nyq")
        with c2:
            show(pl.bode(tr, UNIT, mode="magphase" if bmode == "|Z| · phase" else "reim"),
                 key="d_bode")

        with st.expander("Points & table", icon=":material/table_rows:"):
            sp = spectra[name]
            ss.setdefault(f"excl_{name}", list(ss.excluded.get(name, ())))
            st.multiselect(
                "Exclude points", list(range(sp.n)),
                format_func=lambda i: f"#{i}  ·  {ui.fmt_freq(sp.f[i])}",
                placeholder="Choose outliers to drop from every analysis",
                key=f"excl_{name}")
            st.dataframe(pd.DataFrame({"f (Hz)": f, f"Z′ ({UNIT})": Z.real,
                                       f"Z″ ({UNIT})": Z.imag,
                                       f"|Z| ({UNIT})": np.abs(Z),
                                       "Phase (°)": np.angle(Z, deg=True)}),
                         hide_index=True, height=300)
            st.download_button("Download processed data (CSV)", eio.spectrum_to_csv(f, Z),
                               f"{Path(name).stem}_processed.csv", "text/csv",
                               icon=":material/download:")

    # ---------------------------------------------------------------- KK
    elif view == "Kramers–Kronig":
        with st.container(border=True):
            k1, k2, k3, k4 = st.columns([2, 2, 2, 2])
            kmode = k1.selectbox("Method", list(an.KK_MODES), format_func=an.KK_MODES.get,
                                 key="kk_mode")
            fit_type = k2.segmented_control("Fit", ["complex", "real", "imag"],
                                            default="complex", required=True, key="kk_ft")
            c = k3.slider("μ cut-off c", 0.50, 0.99, 0.85, 0.01, key="kk_c",
                          disabled=kmode == "manual")
            if kmode == "manual":
                M = k4.number_input("RC elements M", 2, 100, 20, key="kk_M")
                max_M = M
            else:
                max_M = k4.number_input("Max M", 5, 100, 50, key="kk_maxM")
                M = None
            add_cap = st.toggle("Add series capacitance (spectra without a low-frequency "
                                "intercept, e.g. blocking electrodes)", key="kk_cap")
        try:
            kk = kk_cached(f, Z, c, int(max_M), fit_type, add_cap, kmode, M)
        except Exception as exc:
            st.error(f"Lin-KK failed: {exc}", icon=":material/error:")
            return
        verdict, color = kk.verdict
        ui.show_pill(f"{verdict}  ·  max residual {kk.max_residual * 100:.2f} %", color)
        ui.tiles([("RC elements M", f"{kk.M}", None), ("μ", f"{kk.mu:.3f}", f"c = {c:.2f}"),
                  ("Max residual", f"{kk.max_residual * 100:.2f} %", None),
                  ("Mean residual", f"{kk.mean_residual * 100:.2f} %", None)])
        c1, c2 = st.columns(2)
        tr = [dict(name="Data", f=f, Z=Z, kind="data", color=pl.BLUE),
              dict(name="Lin-KK", f=f, Z=kk.Z_fit, kind="kk", color=pl.GREEN)]
        with c1:
            show(pl.residuals(f, kk.res_real, kk.res_imag, height=380,
                              title="Kramers–Kronig residuals"), key="kk_res")
        with c2:
            show(pl.nyquist(tr, UNIT, height=380, decade_labels=False), key="kk_nyq")
        if kk.scan is not None:
            show(pl.kk_scan(kk.scan, c), key="kk_scan")
        ui.note("Residuals inside ±1 % (green band) indicate linear, causal, stable data. "
                "Systematic trends at low frequency usually mean drift or a non-steady state.")
        cite_panel(lin_kk=True)

    # ---------------------------------------------------------------- DRT
    elif view == "DRT":
        drt_view(name, f, Z)

    # ---------------------------------------------------------------- fit
    else:
        prev = ss.fits.get(name)
        left, right = st.columns([5, 7], gap="large")
        with left:
            ui.section("Model", "Pick a preset or type your own circuit.")
            circuit, table = circuit_setup("single", f, Z, prev)
            opts = optimizer_options("single")
            run = st.button("Fit circuit", type="primary", icon=":material/bolt:",
                            width="stretch", disabled=circuit is None, key="fit_btn")
        if run and circuit is not None:
            if table["Initial"].isna().any():
                st.error("Every parameter needs an initial value.")
            else:
                with st.spinner("Fitting…"):
                    try:
                        cfg = an.FitConfig(circuit, table, **opts)
                        ss.fits[name] = an.fit_circuit(name, f, Z, cfg, unit=UNIT)
                        prev = ss.fits[name]
                        st.toast("Fit complete", icon=":material/check_circle:")
                    except Exception as exc:
                        st.error(f"Fit failed: {exc}", icon=":material/error:")
        with right:
            stale = prev is not None and (prev.circuit != circuit or prev.f.size != f.size)
            if prev is not None and not stale:
                fg, Zg = prev.smooth()
                tr = [dict(name="Data", f=f, Z=Z, kind="data", color=pl.BLUE),
                      dict(name="Fit", f=fg, Z=Zg, kind="fit", color=pl.ORANGE)]
                ttl = "Fit"
            else:
                tr = [dict(name="Data", f=f, Z=Z, kind="data", color=pl.BLUE)]
                ttl = "Preview  ·  initial guess"
                if circuit is not None and not table["Initial"].isna().any():
                    fg = np.logspace(np.log10(f.max()), np.log10(f.min()), 250)
                    try:
                        Zg = an.simulate(circuit, dict(zip(table.Parameter,
                                                           table.Initial.astype(float))), fg)
                        if np.isfinite(Zg).all():
                            tr.append(dict(name="Initial guess", f=fg, Z=Zg,
                                           kind="guess", color=pl.GRAY))
                    except Exception:
                        pass
            pv = st.segmented_control("Plot", ["Nyquist", "Bode"], default="Nyquist",
                                      required=True, key="fit_plot",
                                      label_visibility="collapsed")
            if pv == "Nyquist":
                show(pl.nyquist(tr, UNIT, height=520, decade_labels=False, title=ttl),
                     key="f_nyq")
            else:
                show(pl.bode(tr, UNIT, height=520, title=ttl), key="f_bode")
            if stale:
                ui.note("The last fit used a different circuit or data window - "
                        "press Fit circuit to update.")

        if prev is not None and not stale:
            ui.section("Results", f"{prev.circuit}  ·  {name}")
            result_tables(prev)
            show(pl.residuals(prev.f, prev.res_real, prev.res_imag, title="Fit residuals"),
                 key="f_res")
            single_downloads(prev, Path(name).stem)
            cite_panel(circuit=True)


# --------------------------------------------------------------------------
# BATCH
# --------------------------------------------------------------------------
def batch_view() -> None:
    names_all = list(spectra)
    with st.container(border=True):
        c1, c2, c3 = st.columns([4, 2, 2], vertical_alignment="bottom")
        chosen = c1.multiselect("Spectra", names_all, default=names_all, key="b_names")
        order = c2.segmented_control("Order", ["Name", "Loaded"], default="Name",
                                     required=True, key="b_order")
        xaxis = c3.segmented_control("Trend x-axis", ["Index", "# in name"],
                                     default="Index", required=True, key="b_x",
                                     help="'# in name' uses the last number in each file "
                                          "name, e.g. cycle_25.mpt → 25.")
    if not chosen:
        st.info("Select at least one spectrum.")
        return
    if order == "Name":
        chosen = sorted(chosen, key=an.natural_key)
    data = {n: prepared(n) for n in chosen}
    short = [n for n, (f, _) in data.items() if f.size < 3]
    if short:
        st.warning(f"Skipped (fewer than 3 points): {', '.join(short)}")
        chosen = [n for n in chosen if n not in short]
        if not chosen:
            return
    colors = dict(zip(chosen, pl.colors_for(len(chosen))))

    ui.section("Overview", f"{len(chosen)} spectra")
    tr = [dict(name=n, f=data[n][0], Z=data[n][1], kind="data", color=colors[n])
          for n in chosen]
    c1, c2 = st.columns(2)
    with c1:
        show(pl.nyquist(tr, UNIT, decade_labels=False, height=420), key="b_nyq")
    with c2:
        show(pl.bode(tr, UNIT, height=420), key="b_bode")

    task = st.segmented_control("Batch analysis", ["Circuit fit", "DRT"],
                                default="Circuit fit", required=True, key="b_task")
    if task == "DRT":
        batch_drt_view(chosen, data, colors, xaxis)
        return

    ui.section("Batch fit", "One circuit, applied to every selected spectrum.")
    left, right = st.columns([5, 7], gap="large")
    with left:
        ref = st.selectbox("Reference spectrum for starting values", chosen, key="b_ref")
        fr, Zr = data[ref]
        last = (ss.batch or {}).get("results", {}).get(ref)
        circuit, table = circuit_setup("batch", fr, Zr, last)
    with right:
        start = st.segmented_control(
            "Starting values", ["Table", "Chain previous fit", "Estimate per file"],
            default="Chain previous fit", required=True, key="b_start",
            help="Chain: each fit starts from the previous file's result - robust for "
                 "slowly evolving series (cycling, SOC, temperature).")
        run_kk = st.toggle("Also run Lin-KK on each spectrum", value=True, key="b_kk")
        opts = optimizer_options("batch")
        run_batch = st.button(f"Fit {len(chosen)} spectra", type="primary",
                              icon=":material/bolt:", width="stretch",
                              disabled=circuit is None, key="b_go")

    if run_batch and circuit is not None:
        results, errors, kks = {}, {}, {}
        bar = st.progress(0.0, text="Starting…")
        prev_vals = None
        for i, n in enumerate(chosen):
            bar.progress(i / len(chosen), text=f"Fitting {n}  ({i + 1}/{len(chosen)})")
            f, Z = data[n]
            tbl = table.copy()
            override = None
            try:
                if start == "Estimate per file":
                    est = an.initial_guess(circuit, f, Z)
                    tbl["Initial"] = [v if fx else est[p] for p, v, fx in
                                      zip(tbl.Parameter, tbl.Initial, tbl.Fixed)]
                elif start == "Chain previous fit" and prev_vals:
                    override = prev_vals
                cfg = an.FitConfig(circuit, tbl, **opts)
                res = an.fit_circuit(n, f, Z, cfg, unit=UNIT, initial_override=override)
                results[n] = res
                prev_vals = {k: v for k, v in res.values.items() if k not in res.fixed}
            except Exception as exc:
                errors[n] = str(exc)
            if run_kk:
                try:
                    kks[n] = kk_cached(f, Z, 0.85, 50, "complex", False, "auto", None)
                except Exception:
                    pass
        bar.progress(1.0, text="Done")
        bar.empty()
        ss.batch = dict(results=results, errors=errors, kk=kks, circuit=circuit,
                        order=chosen, unit=UNIT,
                        config=an.FitConfig(circuit, table, **opts).to_dict()
                        | {"starting_values": start},
                        time=datetime.now().isoformat(timespec="seconds"))
        st.toast(f"{len(results)} fitted, {len(errors)} failed",
                 icon=":material/check_circle:")

    b = ss.batch
    if not b:
        return
    results: dict[str, an.FitResult] = b["results"]
    order = [n for n in b["order"] if n in results]
    ui.section("Batch results", f"{b['circuit']}  ·  {b['time']}")
    if b["errors"]:
        with st.expander(f"{len(b['errors'])} spectra failed", icon=":material/error:"):
            for n, msg in b["errors"].items():
                st.markdown(f"**{n}** - {msg}")
    if not order:
        return
    summary = pd.DataFrame([results[n].summary_row() for n in order])
    if b["kk"]:
        summary["KK max resid %"] = [b["kk"][n].max_residual * 100 if n in b["kk"]
                                     else np.nan for n in order]
    summary.insert(1, "Index", range(1, len(order) + 1))
    summary.insert(2, "# in name", [an.number_in_name(n) for n in order])

    chi = summary["χ²"]
    ui.tiles([("Fitted", f"{len(order)}", f"of {len(order) + len(b['errors'])}"),
              ("Median χ²", ui.fmt(float(chi.median()), 3), None),
              ("Worst χ²", ui.fmt(float(chi.max()), 3),
               summary.loc[chi.idxmax(), "File"]),
              ("Mean |ΔZ|/|Z|", f"{summary['Mean |ΔZ|/|Z| %'].mean():.2f} %", None)])

    pnames = results[order[0]].names
    free = [p for p in pnames if p not in results[order[0]].fixed]
    default = [p for p in free if p.startswith("R")][:4] or free[:4]
    sel = st.multiselect("Parameters to plot", free, default=default, key="b_trend")
    xcol = "Index" if xaxis == "Index" else "# in name"
    if xcol == "# in name" and summary[xcol].isna().any():
        st.caption("Some names contain no number - using index instead.")
        xcol = "Index"
    if sel:
        show(pl.trends(summary, sel, xcol, "File index" if xcol == "Index"
                       else "Number in file name"), key="b_trends")

    st.dataframe(summary, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.4g")
                                for c in summary.columns
                                if c not in ("File", "Circuit", "Index", "# in name")})

    ui.section("Inspect", "Overlay of data and fits, or drill into one spectrum.")
    ov = st.segmented_control("Show", ["All fits", "One spectrum"], default="All fits",
                              required=True, key="b_show", label_visibility="collapsed")
    cols = dict(zip(order, pl.colors_for(len(order))))
    if ov == "All fits":
        tr = []
        for n in order:
            r = results[n]
            fg, Zg = r.smooth(200)
            tr += [dict(name=n, f=r.f, Z=r.Z, kind="data", color=cols[n], group=n),
                   dict(name=f"{n} fit", f=fg, Z=Zg, kind="fit", color=cols[n], group=n,
                        legend=False)]
        ui.note("Markers = data, lines = fits. Click a legend entry to hide a spectrum "
                "and its fit.")
        c1, c2 = st.columns(2)
        with c1:
            show(pl.nyquist(tr, b["unit"], decade_labels=False, height=460), key="b_ov_nyq")
        with c2:
            show(pl.bode(tr, b["unit"], height=460), key="b_ov_bode")
    else:
        n = st.selectbox("Spectrum", order, key="b_inspect")
        fit_figures(results[n], key="b_one")
        result_tables(results[n])

    # ---- downloads
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.csv", summary.to_csv(index=False))
        zf.writestr("batch_config.json", json.dumps(b["config"], indent=2, default=str))
        zf.writestr("CITATION.txt", cite.citation_text(circuit=True, lin_kk=bool(b["kk"])))
        for n in order:
            stem = Path(n).stem
            zf.writestr(f"fits/{stem}_fit.csv", results[n].curve_frame().to_csv(index=False))
            zf.writestr(f"fits/{stem}_parameters.csv",
                        results[n].param_frame().to_csv(index=False))
            zf.writestr(f"models/{stem}_model.json", results[n].model_json())
    c1, c2, _ = st.columns([2, 2, 3])
    c1.download_button("Summary (CSV)", summary.to_csv(index=False), "batch_summary.csv",
                       "text/csv", icon=":material/table_view:", width="stretch")
    c2.download_button("Everything (ZIP)", buf.getvalue(), "eis_batch_results.zip",
                       "application/zip", icon=":material/folder_zip:", width="stretch",
                       type="primary")
    cite_panel(circuit=True, lin_kk=bool(b["kk"]))


def batch_drt_view(chosen: list[str], data: dict, colors: dict, xaxis: str) -> None:
    ui.section("Batch DRT", "The same inversion applied to every selected spectrum.")
    opt = drt_controls("bdrt", allow_hmc=False)
    run = st.button(f"Compute DRT for {len(chosen)} spectra", type="primary",
                    icon=":material/bar_chart:", width="stretch",
                    disabled=opt is None, key="bdrt_go")
    if run and opt is not None:
        results, errors = {}, {}
        bar = st.progress(0.0, text="Starting…")
        for i, n in enumerate(chosen):
            bar.progress(i / len(chosen), text=f"DRT {n}  ({i + 1}/{len(chosen)})")
            f, Z = data[n]
            try:
                results[n] = drt_cached(n, f, Z, opt, UNIT)
            except Exception as exc:
                errors[n] = str(exc)
        bar.empty()
        ss.batch_drt = dict(results=results, errors=errors, order=chosen,
                            method=opt.method,
                            time=datetime.now().isoformat(timespec="seconds"))
        st.toast(f"{len(results)} inverted, {len(errors)} failed",
                 icon=":material/check_circle:")

    b = ss.batch_drt
    if not b:
        return
    res: dict[str, drt.DRTResult] = b["results"]
    order = [n for n in b["order"] if n in res]
    if b["errors"]:
        with st.expander(f"{len(b['errors'])} spectra failed", icon=":material/error:"):
            for n, msg in b["errors"].items():
                st.markdown(f"**{n}** - {msg}")
    if not order:
        return
    ui.section("DRT results", f"{drt.METHODS[b['method']]}  ·  {b['time']}")
    axis = st.segmented_control("x-axis", ["τ", "f"], default="τ", required=True,
                                key="bdrt_axis")
    cols = pl.colors_for(len(order))
    show(pl.drt_overlay([res[n] for n in order], cols,
                        axis="tau" if axis == "τ" else "freq"), key="bdrt_overlay")

    summary = pd.DataFrame([res[n].summary_row() for n in order])
    summary.insert(1, "Index", range(1, len(order) + 1))
    summary.insert(2, "# in name", [an.number_in_name(n) for n in order])
    xcol = "Index" if xaxis == "Index" or summary["# in name"].isna().any() else "# in name"
    numeric = [c for c in summary.columns if c not in ("File", "Method", "Index", "# in name")
               and pd.api.types.is_numeric_dtype(summary[c])]
    default = [c for c in ("R∞", "Rp") if c in numeric]
    sel = st.multiselect("Quantities to plot", numeric, default=default, key="bdrt_trend")
    if sel:
        show(pl.trends(summary, sel, xcol, "File index" if xcol == "Index"
                       else "Number in file name"), key="bdrt_trends")
    st.dataframe(summary, hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.4g")
                                for c in numeric})

    one = st.selectbox("Inspect spectrum", order, key="bdrt_one")
    drt_results(res[one], "bdrt_one", axis="tau" if axis == "τ" else "freq")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("drt_summary.csv", summary.to_csv(index=False))
        zf.writestr("CITATION.txt", cite.citation_text(drt_method=b["method"]))
        for n in order:
            stem = Path(n).stem
            zf.writestr(f"drt/{stem}_drt.csv", res[n].distribution_frame().to_csv(index=False))
            zf.writestr(f"drt/{stem}_peaks.csv", res[n].peaks.to_csv(index=False))
            zf.writestr(f"drt/{stem}_fit.csv", res[n].fit_frame().to_csv(index=False))
    c1, c2, _ = st.columns([2, 2, 3])
    c1.download_button("DRT summary (CSV)", summary.to_csv(index=False),
                       "drt_batch_summary.csv", "text/csv", icon=":material/table_view:",
                       width="stretch")
    c2.download_button("Everything (ZIP)", buf.getvalue(), "eis_drt_batch.zip",
                       "application/zip", icon=":material/folder_zip:", width="stretch",
                       type="primary")
    cite_panel(drt_method=b["method"])


if mode == "Single spectrum":
    single_view()
else:
    batch_view()
