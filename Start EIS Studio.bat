@echo off
rem  EIS Studio - double-click to start (Windows).
rem  Finds Python 3.10-3.13 (offers to install 3.12 if missing), then runs launcher.py.
rem  Extra options are passed through, e.g.:  "Start EIS Studio.bat" --bayes
setlocal EnableExtensions
title EIS Studio
cd /d "%~dp0"
set "PYEXE="

rem 1) the "py" launcher with a supported version
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PYEXE (
    for /f "delims=" %%P in ('py -%%V -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
  )
)

rem 2) "python" on PATH, if its version is supported
if not defined PYEXE (
  for /f "delims=" %%P in ('python -c "import sys;print(sys.executable if (3,10)<=sys.version_info[:2]<=(3,13) else '')" 2^>nul') do set "PYEXE=%%P"
)

rem 3) the default per-user install folders (python.org / winget)
for %%D in (Python312 Python311 Python313 Python310) do (
  if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\%%D\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
)

rem discard anything that is not a working interpreter (e.g. the Microsoft Store alias)
if defined PYEXE (
  "%PYEXE%" -c "import sys" >nul 2>&1 || set "PYEXE="
)

rem 4) offer to install Python 3.12 with winget
if not defined PYEXE (
  echo.
  echo   EIS Studio needs Python 3.10 - 3.13, which was not found on this computer.
  where winget >nul 2>&1 && (
    choice /C YN /M "  Install Python 3.12 now with Microsoft winget (about 30 MB)"
    if not errorlevel 2 winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
  )
  for %%D in (Python312) do (
    if exist "%LOCALAPPDATA%\Programs\Python\%%D\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
  )
)

if not defined PYEXE (
  echo.
  echo   Please install Python 3.12 from https://www.python.org/downloads/
  echo   ^(tick "Add python.exe to PATH" during setup^), then double-click this file again.
  start "" https://www.python.org/downloads/
  echo.
  pause
  exit /b 1
)

"%PYEXE%" "%~dp0launcher.py" %*
if errorlevel 1 (
  echo.
  pause
)
endlocal
