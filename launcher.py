"""EIS Studio launcher - works the same on Windows, macOS and Linux.

    python launcher.py            set up (first time only) and start the app
    python launcher.py --bayes    also install CmdStan for Bayesian DRT (one time)
    python launcher.py --check    set up and verify, but do not start
    python launcher.py --reset    delete the private environment and start over

The double-click wrappers (Start EIS Studio.bat, start_eis_studio.sh) find a
suitable Python and call this file.

Everything is installed into a private virtual environment outside this
folder (so cloud-synced folders such as iCloud/OneDrive/Dropbox are not
flooded with files):
    Windows : %LOCALAPPDATA%\\EISStudio
    macOS   : ~/Library/Application Support/EISStudio
    Linux   : ~/.local/share/eis-studio
Override with the EIS_STUDIO_HOME environment variable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import socket
import subprocess
import sys
import venv
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_PY, MAX_PY = (3, 10), (3, 13)
WINDOWS = os.name == "nt"

for _stream in (sys.stdout, sys.stderr):   # never crash on legacy console code pages
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


def say(msg: str = "") -> None:
    print(msg, flush=True)


def fail(msg: str, code: int = 1) -> None:
    say()
    say("  ✗ " + msg)
    say()
    sys.exit(code)


def home() -> Path:
    if os.environ.get("EIS_STUDIO_HOME"):
        return Path(os.environ["EIS_STUDIO_HOME"]).expanduser()
    if WINDOWS:
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "EISStudio"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "EISStudio"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "eis-studio"


def venv_python(env: Path) -> Path:
    return env / ("Scripts/python.exe" if WINDOWS else "bin/python")


def run(cmd: list, **kw) -> int:
    return subprocess.call([str(c) for c in cmd], **kw)


def check_python() -> None:
    v = sys.version_info[:2]
    if v < MIN_PY:
        fail(f"Python {v[0]}.{v[1]} is too old. EIS Studio needs Python 3.10 – 3.13.\n"
             "    Download Python 3.12 from https://www.python.org/downloads/")
    if v > MAX_PY:
        say(f"  ! Python {v[0]}.{v[1]} is newer than tested (3.10 – 3.13). "
            "If installation fails, install Python 3.12.")


def stamp_for() -> str:
    h = hashlib.sha256()
    h.update((HERE / "requirements.txt").read_bytes())
    h.update(platform.python_version().encode())
    return h.hexdigest()


def ensure_env(env: Path) -> Path:
    py = venv_python(env)
    if not py.exists():
        say(f"  • Creating a private Python environment in\n    {env}")
        try:
            venv.EnvBuilder(with_pip=True, clear=True).create(env)
        except Exception as exc:  # e.g. Debian/Ubuntu without python3-venv
            hint = ("\n    On Debian/Ubuntu run:  sudo apt install python3-venv"
                    if sys.platform.startswith("linux") else "")
            fail(f"Could not create the environment ({exc}).{hint}")
    stamp = env / ".requirements-stamp"
    if not stamp.exists() or stamp.read_text() != stamp_for():
        say("  • Installing EIS Studio's components (first run: 2–5 minutes)…")
        pip = [py, "-m", "pip", "--disable-pip-version-check"]
        if run(pip + ["install", "--upgrade", "pip", "setuptools", "wheel", "-q"]) != 0:
            fail("Could not update pip. Check your internet connection and try again.")
        env_vars = dict(os.environ)
        short = None
        if WINDOWS:   # keep build paths short (Windows 260-character limit)
            short = env.parent / "t"
            try:
                short.mkdir(parents=True, exist_ok=True)
                env_vars["TMP"] = env_vars["TEMP"] = str(short)
            except OSError:
                short = None
        try:
            code = run(pip + ["install", "-r", HERE / "requirements.txt"], env=env_vars)
        finally:
            if short is not None:
                shutil.rmtree(short, ignore_errors=True)
        if code != 0:
            fail("Installing the components failed (see messages above).\n"
                 "    Most common causes: no internet connection, or a Python version\n"
                 "    without ready-made packages - Python 3.12 is the safest choice.")
        stamp.write_text(stamp_for())
        say("  ✓ Components installed")
    return py


def install_cmdstan(py: Path) -> None:
    say("  • Installing CmdStan for Bayesian DRT (one time, 5–15 minutes)…")
    code = ("import cmdstanpy, sys; "
            f"cmdstanpy.install_cmdstan(compiler={WINDOWS}, progress=True); "
            "print('CmdStan at', cmdstanpy.cmdstan_path())")
    if run([py, "-c", code]) != 0:
        say("  ! CmdStan installation failed. Ridge DRT still works.")
        if sys.platform == "darwin":
            say("    On macOS first run:  xcode-select --install")
        elif not WINDOWS:
            say("    On Linux first install a compiler, e.g.:  sudo apt install build-essential")
    else:
        say("  ✓ Bayesian DRT ready")


def verify(py: Path) -> None:
    code = ("import streamlit, impedance, plotly, scipy, pandas;"
            "import sys; sys.path.insert(0, r'%s');"
            "from eis_studio import drt; a = drt.availability();"
            "print('    streamlit', streamlit.__version__, '| impedance', impedance.__version__);"
            "print('    DRT ridge:', 'yes' if a.ridge else 'no', "
            "'| DRT Bayesian:', 'yes' if a.bayes else 'no (optional: run with --bayes)')"
            % HERE)
    if run([py, "-c", code]) != 0:
        fail("The installation looks incomplete. Run again with --reset.")


def free_port(start: int = 8501) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start


def main() -> None:
    ap = argparse.ArgumentParser(description="Set up and start EIS Studio.")
    ap.add_argument("--bayes", action="store_true", help="install CmdStan (Bayesian DRT)")
    ap.add_argument("--check", action="store_true", help="set up and verify, do not start")
    ap.add_argument("--reset", action="store_true", help="re-create the environment")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    say()
    say("  EIS Studio")
    say("  ──────────")
    check_python()
    env = home() / "venv"
    if args.reset and env.exists():
        say("  • Removing the old environment…")
        shutil.rmtree(env, ignore_errors=True)
    py = ensure_env(env)
    if args.bayes:
        install_cmdstan(py)
    verify(py)
    if args.check:
        say("  ✓ Ready. Start without --check to open the app.")
        return

    port = args.port or free_port()
    say(f"  ✓ Starting at http://localhost:{port}  (close this window to stop)")
    say()
    os.chdir(HERE)
    sys.exit(run([py, "-m", "streamlit", "run", HERE / "app.py",
                  "--server.port", port, "--server.headless", "false",
                  "--browser.gatherUsageStats", "false"]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\n  Stopped.")
