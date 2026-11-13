"""Core tests: loading every sample format, validation, fitting, exports."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eis_studio import analysis as an
from eis_studio import io as eio

DATA = Path(__file__).resolve().parents[1] / "sample_data"

EXPECTED = {
    "exampleData.csv": "csv",
    "exampleDataGamry.DTA": "gamry",
    "exampleDataBioLogic.mpt": "biologic",
    "exampleDataZPlot.z": "zplot",
    "exampleDataCHInstruments.txt": "chinstruments",
    "exampleDataAutolab.txt": "autolab",
    "exampleDataParstat.txt": "parstat",
    "exampleDataPowersuite.txt": "powersuite",
    "exampleDataVersaStudio.par": "versastudio",
}


@pytest.mark.parametrize("fname,inst", EXPECTED.items())
def test_autodetect_loads_every_format(fname, inst):
    sp = eio.load_spectrum(fname, (DATA / fname).read_bytes(), "auto")
    assert sp.source == inst
    assert sp.n >= 3
    assert np.all(np.diff(sp.f) < 0)          # sorted high → low
    assert np.isfinite(sp.Z).all()


def test_generic_table_mod_phase():
    f = np.logspace(5, -1, 30)
    Z = an.simulate("R0-p(R1,C1)", {"R0": 10, "R1": 50, "C1": 1e-5}, f)
    txt = "Frequency (Hz);|Z| (Ohm);Phase (deg)\n" + "\n".join(
        f"{a};{b};{c}" for a, b, c in zip(f, np.abs(Z), np.angle(Z, deg=True)))
    df = eio.read_table(txt.encode())
    m = eio.suggest_mapping(df)
    assert m.mode == "mod_phase"
    sp = eio.load_spectrum("x.txt", txt.encode(), "generic")
    np.testing.assert_allclose(sp.Z, Z[np.argsort(f)[::-1]], rtol=1e-9)


def test_circuit_validation():
    assert [p.name for p in an.parse_circuit("R0-p(R1,CPE1)")] == \
        ["R0", "R1", "CPE1_0", "CPE1_1"]
    for bad in ["", "R0-p(R1,C1", "R0-X1", "__import__('os')", "R0-(R1,C1)", "R1-R1"]:
        with pytest.raises(an.CircuitError):
            an.parse_circuit(bad)


@pytest.mark.parametrize("preset", list(an.PRESETS))
def test_presets_have_sane_guesses(preset):
    sp = eio.load_spectrum("exampleData.csv", (DATA / "exampleData.csv").read_bytes())
    circuit = an.PRESETS[preset]["circuit"]
    tbl = an.parameter_table(circuit, sp.f, sp.Z)
    assert (tbl["Initial"] > 0).all()
    assert (tbl["Initial"] <= tbl["Upper"]).all()


def test_fit_recovers_synthetic_parameters():
    f = np.logspace(5, -2, 60)
    true = {"R0": 0.02, "R1": 0.05, "CPE1_0": 2.0, "CPE1_1": 0.85}
    Z = an.simulate("R0-p(R1,CPE1)", true, f)
    tbl = an.parameter_table("R0-p(R1,CPE1)", f, Z)
    res = an.fit_circuit("synthetic", f, Z, an.FitConfig("R0-p(R1,CPE1)", tbl))
    for k, v in true.items():
        assert res.values[k] == pytest.approx(v, rel=1e-3)
    assert res.chi2 < 1e-8
    assert not res.derived().empty


def test_fit_real_data_with_fixed_param_and_exports():
    sp = eio.load_spectrum("exampleData.csv", (DATA / "exampleData.csv").read_bytes())
    f, Z = an.preprocess(sp.f, sp.Z, an.Preprocess(drop_inductive=True))
    circuit = "R0-p(R1,CPE1)-p(R2-Wo1,CPE2)"
    tbl = an.parameter_table(circuit, f, Z)
    tbl.loc[tbl.Parameter == "R0", ["Initial", "Fixed"]] = [0.0145, True]
    res = an.fit_circuit("ex", f, Z, an.FitConfig(circuit, tbl))
    assert res.fixed == {"R0": 0.0145}
    assert res.mean_rel_error < 5
    assert len(res.param_frame()) == 9
    assert "Circuit String" in res.model_json()
    assert set(res.curve_frame().columns) >= {"freq_Hz", "Zfit_real"}


def test_global_optimizer_runs():
    f = np.logspace(4, -1, 30)
    Z = an.simulate("R0-p(R1,C1)", {"R0": 5, "R1": 20, "C1": 1e-4}, f)
    tbl = an.parameter_table("R0-p(R1,C1)", f, Z)
    res = an.fit_circuit("g", f, Z, an.FitConfig("R0-p(R1,C1)", tbl,
                                                 global_opt=True, niter=5))
    assert res.values["R1"] == pytest.approx(20, rel=0.05)


def test_linkk_on_clean_data_is_valid():
    f = np.logspace(5, -2, 50)
    Z = an.simulate("R0-p(R1,CPE1)", {"R0": 1, "R1": 10, "CPE1_0": 1e-4,
                                      "CPE1_1": 0.9}, f)
    kk = an.run_linkk(f, Z)
    assert kk.verdict[0] == "Valid"


def test_preprocess_area_and_window():
    f = np.array([1e3, 1e2, 1e1, 1.0])
    Z = np.array([1 + 0.1j, 2 - 1j, 3 - 2j, 4 - 1j])
    f2, Z2 = an.preprocess(f, Z, an.Preprocess(fmin=5, drop_inductive=True,
                                               area_cm2=2.0, exclude=(2,)))
    assert f2.tolist() == [1e2]
    assert Z2[0] == pytest.approx(4 - 2j)


def test_css_survives_html_sanitiser():
    """st.html (DOMPurify) drops a <style> whose body contains '<'."""
    from eis_studio import ui
    body = ui._CSS.split("<style>", 1)[1].rsplit("</style>", 1)[0]
    assert "<" not in body and "__WAVE_MASK__" not in body
    assert "data:image/svg+xml," in body


def test_batch_helpers():
    names = ["cell_10.z", "cell_2.z", "cell_1.z"]
    assert sorted(names, key=an.natural_key) == ["cell_1.z", "cell_2.z", "cell_10.z"]
    assert an.number_in_name("cycle_25.mpt") == 25
    assert isinstance(pd.DataFrame([{"a": 1}]), pd.DataFrame)
