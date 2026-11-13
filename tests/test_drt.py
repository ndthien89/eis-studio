"""DRT (bayes_drt2) integration tests."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from eis_studio import analysis as an
from eis_studio import drt

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_availability_never_raises():
    av = drt.availability()
    assert isinstance(av.ridge, bool) and isinstance(av.bayes, bool)
    if not av.ridge:
        assert "bayes_drt2" in av.message


def _two_arcs(a=0.85):
    """Two depressed arcs (R‖CPE): 20 Ω at 1 kHz and 40 Ω at 1 Hz."""
    f = np.logspace(5, -2, 71)
    vals = {"R0": 5.0,
            "R1": 20.0, "CPE1_0": (1 / (2 * np.pi * 1e3)) ** a / 20.0, "CPE1_1": a,
            "R2": 40.0, "CPE2_0": (1 / (2 * np.pi * 1.0)) ** a / 40.0, "CPE2_1": a}
    return f, an.simulate("R0-p(R1,CPE1)-p(R2,CPE2)", vals, f)


def test_ridge_recovers_two_processes():
    pytest.importorskip("bayes_drt2")
    f, Z = _two_arcs()
    res = drt.compute_drt("synthetic", f, Z, drt.DRTOptions(method="ridge"))
    assert res.R_inf == pytest.approx(5.0, rel=0.05)
    assert res.Rp == pytest.approx(60.0, rel=0.03)
    assert res.mean_rel_error < 1.0
    pk = res.peaks.sort_values("f_peak (Hz)").reset_index(drop=True)
    assert len(pk) == 2 and pk["Inside data range"].all()
    assert pk.loc[0, "f_peak (Hz)"] == pytest.approx(1.0, rel=0.1)
    assert pk.loc[0, "R"] == pytest.approx(40.0, rel=0.05)
    assert pk.loc[1, "f_peak (Hz)"] == pytest.approx(1e3, rel=0.1)
    assert pk.loc[1, "R"] == pytest.approx(20.0, rel=0.05)
    assert set(res.distribution_frame().columns) >= {"tau_s", "gamma", "peak_1"}


def test_no_degenerate_peaks_on_single_arc():
    """Circuit1 example = one arc. Shoulder search used to add alpha=0 'peaks'."""
    pytest.importorskip("bayes_drt2")
    from eis_studio import io as eio
    raw = (Path(APP).parent / "sample_data" / "Circuit1_EIS_1.z").read_bytes()
    sp = eio.load_spectrum("Circuit1_EIS_1.z", raw)
    res = drt.compute_drt("c1", sp.f, sp.Z, drt.DRTOptions())
    assert len(res.peaks) == 1
    assert (res.peaks["α (HN)"] >= 0.05).all() and (res.peaks["β (HN)"] >= 0.05).all()
    assert any("% of Rp" in n for n in res.notes)        # honest about the gap


def test_too_few_points_is_a_clear_error():
    pytest.importorskip("bayes_drt2")
    f, Z = _two_arcs()
    with pytest.raises(RuntimeError, match="at least 10"):
        drt.compute_drt("x", f[:5], Z[:5], drt.DRTOptions())


def test_drt_view_renders():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    next(b for b in at.button if b.label == "Try example data").click().run()
    at.session_state["single_view"] = "DRT"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if drt.availability().ridge:
        next(b for b in at.button if b.label == "Compute DRT").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert not at.error, [e.value for e in at.error]
        assert at.session_state["drt"]
        # batch DRT
        at.session_state["mode"] = "Batch"
        at.run()
        at.session_state["b_task"] = "DRT"
        at.run()
        btn = next(b for b in at.button if b.label.startswith("Compute DRT for"))
        btn.click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert len(at.session_state["batch_drt"]["results"]) == 7
    else:
        assert any("bayes_drt2" in w.value for w in at.warning)
