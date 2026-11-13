"""Apple-style presentation helpers.

Follows the Human Interface Guidelines' three themes:
* Clarity   - system font, strong type hierarchy, plain language, one primary
              action per view.
* Deference - content (the spectra) fills the canvas; chrome is quiet and
              translucent; colour is reserved for data and state.
* Depth     - layered surfaces (tiles, popovers), soft shadows, gentle motion.

Surfaces are tinted from ``currentColor`` so the same CSS works in light and
dark appearance without knowing which one is active.
"""

from __future__ import annotations

import html
from urllib.parse import quote

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root { --eis-ease: cubic-bezier(.25,.1,.25,1); }

[data-testid="stMainBlockContainer"] { padding-top: 4.2rem; max-width: 1320px; }
[data-testid="stHeader"] {
  background: transparent;
  backdrop-filter: saturate(180%) blur(20px);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
}
h1, h2, h3, h4 { letter-spacing: -0.022em; }

/* ---------- hero ---------- */
.eis-hero { margin: 2.2rem 0 1.4rem 0; }
.eis-hero .eyebrow { font-size: .78rem; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; color: #007AFF; margin-bottom: .25rem; }
.eis-hero h1 { font-size: 2.35rem; font-weight: 700; line-height: 1.08; margin: 0;
  padding: 0; letter-spacing: -0.028em; }
.eis-hero p { font-size: 1.06rem; opacity: .62; margin: .45rem 0 0 0; max-width: 46rem; }

/* ---------- section titles ---------- */
.eis-section { margin: 1.6rem 0 .55rem 0; }
.eis-section h3 { font-size: 1.22rem; font-weight: 600; margin: 0; padding: 0; }
.eis-section p { font-size: .9rem; opacity: .58; margin: .15rem 0 0 0; }

/* ---------- stat tiles ---------- */
.eis-tiles { display: grid; gap: .75rem;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin: .2rem 0 1rem 0; }
.eis-tile { border-radius: 16px; padding: .85rem 1rem .8rem 1rem;
  background: color-mix(in srgb, currentColor 4.5%, transparent);
  transition: transform .25s var(--eis-ease), background .25s var(--eis-ease); }
.eis-tile:hover { transform: translateY(-1px);
  background: color-mix(in srgb, currentColor 7%, transparent); }
.eis-tile .label { font-size: .74rem; font-weight: 500; opacity: .55; letter-spacing: .01em; }
.eis-tile .value { font-size: 1.32rem; font-weight: 600; letter-spacing: -0.02em;
  margin-top: .1rem; font-variant-numeric: tabular-nums; }
.eis-tile .sub { font-size: .74rem; opacity: .5; margin-top: .05rem; }

/* ---------- pills ---------- */
.eis-pill { display: inline-flex; align-items: center; gap: .35rem; font-size: .78rem;
  font-weight: 600; padding: .22rem .65rem; border-radius: 999px; }
.eis-pill .dot { width: .5rem; height: .5rem; border-radius: 50%; background: currentColor; }
.eis-pill.green  { color: #248A3D; background: rgba(52,199,89,.14); }
.eis-pill.orange { color: #C93400; background: rgba(255,149,0,.16); }
.eis-pill.red    { color: #D70015; background: rgba(255,59,48,.14); }
.eis-pill.blue   { color: #0066CC; background: rgba(0,122,255,.12); }
.eis-pill.gray   { color: inherit; opacity: .7; background: color-mix(in srgb, currentColor 8%, transparent); }
@media (prefers-color-scheme: dark) {
  .eis-pill.green { color: #30D158; } .eis-pill.orange { color: #FF9F0A; }
  .eis-pill.red { color: #FF453A; } .eis-pill.blue { color: #409CFF; }
}

/* ---------- empty state ---------- */
.eis-empty { text-align: center; padding: 3.2rem 1rem 2.6rem 1rem; border-radius: 22px;
  background: color-mix(in srgb, currentColor 3.5%, transparent); margin-top: .5rem; }
.eis-empty .glyph { width: 64px; height: 64px; margin: 0 auto 1rem auto; border-radius: 18px;
  display: grid; place-items: center; color: #fff;
  background: linear-gradient(145deg, #5AC8FA 0%, #007AFF 55%, #5856D6 100%);
  box-shadow: 0 10px 30px rgba(0,122,255,.28); }
.eis-empty h3 { font-size: 1.35rem; font-weight: 600; margin: 0; padding: 0; }
.eis-empty p { opacity: .6; max-width: 34rem; margin: .45rem auto 0 auto; font-size: .98rem; }
.eis-steps { display: flex; gap: .6rem; justify-content: center; flex-wrap: wrap; margin-top: 1.3rem; }
.eis-steps span { font-size: .82rem; padding: .35rem .8rem; border-radius: 999px;
  background: color-mix(in srgb, currentColor 6%, transparent); }

/* ---------- circuit chip ---------- */
.eis-circuit { font-family: "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  font-size: .9rem; padding: .5rem .75rem; border-radius: 10px;
  background: color-mix(in srgb, currentColor 5%, transparent); overflow-x: auto; }
.eis-note { font-size: .84rem; opacity: .6; margin: .1rem 0 .4rem 0; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] h2 { font-size: 1.05rem; }
.eis-brand { display: flex; align-items: center; gap: .6rem; margin: -.4rem 0 1rem 0; }
.eis-brand .mark { width: 30px; height: 30px; border-radius: 8px; display: grid;
  place-items: center; color: #fff;
  background: linear-gradient(145deg, #5AC8FA 0%, #007AFF 55%, #5856D6 100%); }
.eis-brand .name { font-weight: 650; font-size: 1.02rem; letter-spacing: -0.01em; }
.eis-brand .ver { font-size: .72rem; opacity: .5; }
.eis-file { display: flex; justify-content: space-between; gap: .5rem; font-size: .84rem;
  padding: .32rem 0; border-bottom: 1px solid color-mix(in srgb, currentColor 8%, transparent); }
.eis-file .meta { opacity: .5; white-space: nowrap; }

/* ---------- app mark: white sine wave drawn with a CSS mask ---------- */
.eis-brand .mark, .eis-empty .glyph { position: relative; }
.eis-brand .mark::after, .eis-empty .glyph::after {
  content: ""; position: absolute; inset: 22%; background: #fff;
  -webkit-mask: __WAVE_MASK__ center / contain no-repeat;
          mask: __WAVE_MASK__ center / contain no-repeat;
}

/* ---------- motion ---------- */
@media (prefers-reduced-motion: no-preference) {
  .eis-hero, .eis-tiles, .eis-empty { animation: eis-in .45s var(--eis-ease) both; }
  @keyframes eis-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
}
</style>
"""

# st.html sanitises inline SVG (and drops a <style> whose text contains raw
# "<tag"), so the mark's wave is a percent-encoded SVG used as a CSS mask.
_WAVE_SVG = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
             "stroke='black' stroke-width='2.4' stroke-linecap='round'>"
             "<path d='M2 12c2.5-7 5-7 7.5 0s5 7 7.5 0 3.5-4 5-3'/></svg>")
_CSS = _CSS.replace("__WAVE_MASK__",
                    'url("data:image/svg+xml,' + quote(_WAVE_SVG, safe="/:=' ") + '")')
_WAVE = ""


def inject_css() -> None:
    st.html(_CSS)


def esc(s) -> str:
    return html.escape(str(s))


def brand(version: str) -> None:
    st.html(f'<div class="eis-brand"><div class="mark">{_WAVE}</div>'
            f'<div><div class="name">EIS Studio</div>'
            f'<div class="ver">impedance.py · v{esc(version)}</div></div></div>')


def hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.html(f'<div class="eis-hero"><div class="eyebrow">{esc(eyebrow)}</div>'
            f'<h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>')


def section(title: str, caption: str | None = None) -> None:
    cap = f"<p>{esc(caption)}</p>" if caption else ""
    st.html(f'<div class="eis-section"><h3>{esc(title)}</h3>{cap}</div>')


def tiles(items: list[tuple[str, str, str | None]]) -> None:
    cells = "".join(
        f'<div class="eis-tile"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div>'
        + (f'<div class="sub">{esc(sub)}</div>' if sub else "") + "</div>"
        for label, value, sub in items)
    st.html(f'<div class="eis-tiles">{cells}</div>')


def pill(text: str, color: str = "gray") -> str:
    return f'<span class="eis-pill {esc(color)}"><span class="dot"></span>{esc(text)}</span>'


def show_pill(text: str, color: str = "gray") -> None:
    st.html(pill(text, color))


def circuit_chip(circuit: str) -> None:
    st.html(f'<div class="eis-circuit">{esc(circuit)}</div>')


def note(text: str) -> None:
    st.html(f'<div class="eis-note">{esc(text)}</div>')


def empty_state(title: str, text: str, steps: list[str] | None = None) -> None:
    chips = "".join(f"<span>{esc(s)}</span>" for s in steps or [])
    st.html(f'<div class="eis-empty"><div class="glyph">{_WAVE}</div>'
            f'<h3>{esc(title)}</h3><p>{esc(text)}</p>'
            + (f'<div class="eis-steps">{chips}</div>' if chips else "") + "</div>")


def file_row(name: str, meta: str) -> None:
    st.html(f'<div class="eis-file"><span>{esc(name)}</span>'
            f'<span class="meta">{esc(meta)}</span></div>')


def fmt(v: float, digits: int = 4) -> str:
    if v is None or v != v:
        return "–"
    if v == 0:
        return "0"
    a = abs(v)
    if 1e-3 <= a < 1e5:
        return f"{v:.{digits}g}"
    return f"{v:.{max(digits - 1, 1)}e}"


def fmt_freq(f: float) -> str:
    for scale, unit in ((1e6, "MHz"), (1e3, "kHz"), (1, "Hz"), (1e-3, "mHz")):
        if f >= scale * 0.999:
            return f"{f / scale:.3g} {unit}"
    return f"{f:.2g} Hz"
