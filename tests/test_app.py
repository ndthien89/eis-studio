"""Headless end-to-end test of the Streamlit app (streamlit.testing)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


@pytest.fixture()
def loaded():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    _ok(at)
    _button(at, "Try example data").click().run()
    _ok(at)
    return at


def test_empty_state_renders():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    _ok(at)
    assert any(b.label == "Try example data" for b in at.button)


def test_single_spectrum_workflow(loaded):
    at = loaded
    assert at.selectbox(key="single_name").value is not None
    at.selectbox(key="single_name").set_value("exampleData.csv").run()
    _ok(at)

    at.session_state["single_view"] = "Kramers–Kronig"
    at.run()
    _ok(at)
    for mode in ("schonleber", "manual", "auto"):
        at.selectbox(key="kk_mode").set_value(mode).run()
        _ok(at)

    at.session_state["single_view"] = "Circuit fit"
    at.run()
    _ok(at)
    at.selectbox(key="single_preset").set_value("Two arcs  (R‖Q – R‖Q)").run()
    _ok(at)
    assert at.text_input(key="single_circuit").value == "R0-p(R1,CPE1)-p(R2,CPE2)"
    _button(at, "Fit circuit").click().run()
    _ok(at)
    assert "exampleData.csv" in at.session_state["fits"]
    assert len(at.get("download_button")) >= 3

    # custom circuit + invalid circuit message
    at.text_input(key="single_circuit").set_value("R0-p(R1,C1").run()
    assert at.error                     # friendly validation message, no crash
    assert not at.exception
    at.text_input(key="single_circuit").set_value("R0-p(R1,C1)").run()
    _ok(at)


def test_preprocessing_and_exclusion(loaded):
    at = loaded
    at.toggle[0].set_value(True).run()       # remove inductive points
    _ok(at)
    at.session_state["excl_Circuit1_EIS_1.z"] = [0, 1, 2]
    at.run()
    _ok(at)


def test_batch_workflow(loaded):
    at = loaded
    at.session_state["mode"] = "Batch"
    at.run()
    _ok(at)
    at.selectbox(key="batch_preset").set_value("R‖Q  (one arc)").run()
    _ok(at)
    btn = next(b for b in at.button if b.label.startswith("Fit ") and b.label.endswith("spectra"))
    btn.click().run()
    _ok(at)
    batch = at.session_state["batch"]
    assert len(batch["results"]) == 7
    assert not batch["errors"]
    at.session_state["b_show"] = "One spectrum"
    at.run()
    _ok(at)
    at.session_state["b_x"] = "# in name"
    at.run()
    _ok(at)
