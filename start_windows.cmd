@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
pushd "%~dp0" || exit /b 1

py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 goto try_python
py -3 -m nursing_charting.windows_start
set "POC_EXIT=%errorlevel%"
goto finish

:try_python
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 goto missing_python
python -m nursing_charting.windows_start
set "POC_EXIT=%errorlevel%"
goto finish

:missing_python
echo Python 3.12 or later is required. See WINDOWS.md for setup.
set "POC_EXIT=1"

:finish
popd
if not "%POC_EXIT%"=="0" pause
exit /b %POC_EXIT%
