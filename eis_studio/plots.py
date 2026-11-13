"""Plotly figures for EIS Studio.

Figures are rendered with ``theme="streamlit"`` so they follow the app's
light/dark appearance; only trace colours are set here (Apple system colours).
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

BLUE, ORANGE, GREEN, PURPLE, PINK = "#007AFF", "#FF9500", "#34C759", "#AF52DE", "#FF2D55"
TEAL, INDIGO, RED, GRAY = "#5AC8FA", "#5856D6", "#FF3B30", "#8E8E93"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, PINK, TEAL, INDIGO, "#FFCC00", RED, "#A2845E"]
SEQUENCE = [[0.0, TEAL], [0.25, BLUE], [0.5, INDIGO], [0.7, PURPLE],
            [0.85, PINK], [1.0, ORANGE]]

FONT = ('-apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, "Segoe UI", '
        'Roboto, "Helvetica Neue", Arial, sans-serif')
CONFIG = {"displaylogo": False, "responsive": True,
          "toImageButtonOptions": {"format": "svg", "filename": "eis-studio"},
          "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def colors_for(n: int) -> list[str]:
    if n <= len(PALETTE):
        return PALETTE[:n]
    return sample_colorscale(SEQUENCE, [i / max(n - 1, 1) for i in range(n)])


def _style(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
        margin=dict(l=10, r=10, t=48 if title else 28, b=10),
        font=dict(family=FONT, size=12.5),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(font_family=FONT),
        hovermode="closest",
    )
    fig.update_xaxes(zeroline=False, ticks="outside", ticklen=4, showline=False)
    fig.update_yaxes(zeroline=False, ticks="outside", ticklen=4, showline=False)
    return fig


def _fmt_freq(f: float) -> str:
    for scale, unit in ((1e6, "MHz"), (1e3, "kHz"), (1, "Hz"), (1e-3, "mHz"), (1e-6, "µHz")):
        if f >= scale * 0.999:
            v = f / scale
            return f"{v:.3g} {unit}"
    return f"{f:.2g} Hz"


# --------------------------------------------------------------------------
# trace specs: dict(name, f, Z, kind in {"data","fit","guess","kk"}, color)
# --------------------------------------------------------------------------
def _nyq_trace(t: dict, show_legend=True) -> go.Scatter:
    kind = t.get("kind", "data")
    color = t.get("color", BLUE)
    f, Z = np.asarray(t["f"]), np.asarray(t["Z"])
    hover = ("<b>%{meta}</b><br>f = %{customdata:.4g} Hz<br>"
             "Z′ = %{x:.4g}<br>−Z″ = %{y:.4g}<extra></extra>")
    common = dict(x=Z.real, y=-Z.imag, customdata=f, meta=t["name"], name=t["name"],
                  hovertemplate=hover, showlegend=show_legend and t.get("legend", True),
                  legendgroup=t.get("group", t["name"]))
    if kind == "data":
        return go.Scatter(mode="markers", marker=dict(size=7, color=color, opacity=0.85,
                                                      line=dict(width=0)), **common)
    dash = {"fit": "solid", "guess": "dot", "kk": "dash"}.get(kind, "solid")
    width = 2.6 if kind == "fit" else 1.8
    return go.Scatter(mode="lines", line=dict(color=color, width=width, dash=dash),
                      **common)


def nyquist(traces: list[dict], unit: str = "Ω", height: int = 440,
            decade_labels: bool = True, title: str | None = None) -> go.Figure:
    fig = go.Figure([_nyq_trace(t) for t in traces])
    if decade_labels:
        data = next((t for t in traces if t.get("kind", "data") == "data"), None)
        if data is not None and len(data["f"]) > 4:
            f, Z = np.asarray(data["f"]), np.asarray(data["Z"])
            lo, hi = math.ceil(np.log10(f.min())), math.floor(np.log10(f.max()))
            span = max(np.ptp(Z.real), np.ptp(Z.imag), 1e-30)
            placed: list[complex] = []
            for d in range(hi, lo - 1, -1):
                i = int(np.argmin(np.abs(np.log10(f) - d)))
                if abs(np.log10(f[i]) - d) >= 0.25:
                    continue
                z = complex(Z.real[i], -Z.imag[i])
                if any(abs(z - q) < 0.06 * span for q in placed):   # avoid collisions
                    continue
                placed.append(z)
                fig.add_annotation(x=z.real, y=z.imag, text=_fmt_freq(10.0 ** d),
                                   showarrow=False, yshift=13, font=dict(size=10, color=GRAY))
    fig.update_xaxes(title_text=f"Z′ ({unit})")
    fig.update_yaxes(title_text=f"−Z″ ({unit})", scaleanchor="x", scaleratio=1)
    return _style(fig, height, title)


def bode(traces: list[dict], unit: str = "Ω", height: int = 440,
         mode: str = "magphase", title: str | None = None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for t in traces:
        f, Z = np.asarray(t["f"]), np.asarray(t["Z"])
        if mode == "magphase":
            y1, y2 = np.abs(Z), -np.angle(Z, deg=True)
        else:
            y1, y2 = Z.real, -Z.imag
        for row, y in ((1, y1), (2, y2)):
            tr = _nyq_trace(t, show_legend=(row == 1))
            tr.x, tr.y = f, y
            tr.customdata = None
            tr.hovertemplate = "<b>%{meta}</b><br>f = %{x:.4g} Hz<br>%{y:.4g}<extra></extra>"
            fig.add_trace(tr, row=row, col=1)
    fig.update_xaxes(type="log")
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    if mode == "magphase":
        fig.update_yaxes(title_text=f"|Z| ({unit})", type="log", row=1, col=1)
        fig.update_yaxes(title_text="−Phase (°)", row=2, col=1)
    else:
        fig.update_yaxes(title_text=f"Z′ ({unit})", row=1, col=1)
        fig.update_yaxes(title_text=f"−Z″ ({unit})", row=2, col=1)
    return _style(fig, height, title)


def residuals(f, res_real, res_imag, height: int = 260, band: float = 1.0,
              title: str | None = None) -> go.Figure:
    f = np.asarray(f)
    fig = go.Figure()
    fig.add_hrect(y0=-band, y1=band, fillcolor=GREEN, opacity=0.08, line_width=0)
    for y, name, color, sym in ((np.asarray(res_real) * 100, "ΔZ′ / |Z|", BLUE, "circle"),
                                (np.asarray(res_imag) * 100, "ΔZ″ / |Z|", ORANGE, "diamond")):
        fig.add_trace(go.Scatter(x=f, y=y, name=name, mode="lines+markers",
                                 line=dict(color=color, width=1.4),
                                 marker=dict(size=6, color=color, symbol=sym),
                                 hovertemplate="f = %{x:.4g} Hz<br>%{y:.3f} %<extra>"
                                               + name + "</extra>"))
    fig.add_hline(y=0, line=dict(color=GRAY, width=1))
    fig.update_xaxes(type="log", title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Residual (%)")
    return _style(fig, height, title)


def kk_scan(scan, c: float, height: int = 260) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=scan.M, y=scan.mu, name="μ", mode="lines+markers",
                             line=dict(color=BLUE, width=2), marker=dict(size=5)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=scan.M, y=scan.chi2, name="χ²", mode="lines",
                             line=dict(color=ORANGE, width=2)), secondary_y=True)
    if "chosen" in scan:
        ch = scan[scan.chosen]
        fig.add_trace(go.Scatter(x=ch.M, y=ch.mu, name="selected M", mode="markers",
                                 marker=dict(size=13, color=GREEN, symbol="circle-open",
                                             line=dict(width=3))), secondary_y=False)
    fig.add_hline(y=c, line=dict(color=GRAY, dash="dot", width=1),
                  annotation_text=f"c = {c:.2f}", annotation_position="bottom right")
    fig.update_xaxes(title_text="Number of RC elements, M")
    fig.update_yaxes(title_text="μ", secondary_y=False)
    fig.update_yaxes(title_text="χ²", type="log", secondary_y=True, showgrid=False,
                     exponentformat="power")
    return _style(fig, height)


def _drt_x(tau, axis: str):
    return tau if axis == "tau" else 1 / (2 * np.pi * np.asarray(tau))


def drt(res, axis: str = "tau", show_peaks: bool = True, height: int = 420,
        title: str | None = None) -> go.Figure:
    """γ(ln τ) with optional 95 % credible band and HN peak decomposition."""
    x = _drt_x(res.tau, axis)
    fig = go.Figure()
    if res.gamma_lo is not None:
        fig.add_trace(go.Scatter(x=x, y=res.gamma_hi, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=res.gamma_lo, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(0,122,255,0.16)",
                                 name="95 % credible interval", hoverinfo="skip"))
    if show_peaks:
        for k, c in enumerate(res.peak_curves):
            col = PALETTE[(k + 1) % len(PALETTE)]
            fig.add_trace(go.Scatter(x=x, y=c, mode="lines", name=f"Peak {k + 1}",
                                     line=dict(color=col, width=1.6, dash="dot"),
                                     fill="tozeroy", fillcolor=_rgba(col, 0.07)))
    xl = "τ = %{x:.3g} s" if axis == "tau" else "f = %{x:.3g} Hz"
    fig.add_trace(go.Scatter(x=x, y=res.gamma, mode="lines", name="DRT",
                             line=dict(color=BLUE, width=2.8),
                             hovertemplate=xl + "<br>γ = %{y:.4g}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=GRAY, width=1))
    fig.update_xaxes(type="log", title_text="τ (s)" if axis == "tau" else "Frequency (Hz)",
                     autorange="reversed" if axis == "freq" else True)
    fig.update_yaxes(title_text=f"γ ({res.unit})")
    return _style(fig, height, title)


def drt_overlay(results: list, colors: list[str], axis: str = "tau",
                height: int = 440) -> go.Figure:
    fig = go.Figure()
    for r, col in zip(results, colors):
        x = _drt_x(r.tau, axis)
        fig.add_trace(go.Scatter(x=x, y=r.gamma, mode="lines", name=r.name,
                                 line=dict(color=col, width=2.2),
                                 hovertemplate="<b>%{meta}</b><br>%{x:.3g}<br>γ = %{y:.4g}"
                                               "<extra></extra>", meta=r.name))
    fig.add_hline(y=0, line=dict(color=GRAY, width=1))
    unit = results[0].unit if results else "Ω"
    fig.update_xaxes(type="log", title_text="τ (s)" if axis == "tau" else "Frequency (Hz)",
                     autorange="reversed" if axis == "freq" else True)
    fig.update_yaxes(title_text=f"γ ({unit})")
    return _style(fig, height)


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def trends(df, params: list[str], x: str, x_title: str, height_per_row: int = 230
           ) -> go.Figure:
    n = max(len(params), 1)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=params,
                        horizontal_spacing=0.1, vertical_spacing=0.16 if rows > 1 else 0.1)
    for k, p in enumerate(params):
        r, c = k // cols + 1, k % cols + 1
        err = df.get(f"{p} ±")
        fig.add_trace(go.Scatter(
            x=df[x], y=df[p], mode="lines+markers", name=p, showlegend=False,
            line=dict(color=PALETTE[k % len(PALETTE)], width=2),
            marker=dict(size=7),
            error_y=dict(type="data", array=err, visible=True, thickness=1.2, width=3)
            if err is not None else None,
            text=df["File"], hovertemplate="<b>%{text}</b><br>%{y:.4g}<extra></extra>"),
            row=r, col=c)
        fig.update_xaxes(title_text=x_title if r == rows else None, row=r, col=c)
    fig.update_annotations(font=dict(size=13))
    return _style(fig, rows * height_per_row + 40)
