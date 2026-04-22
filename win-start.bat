@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

REM ============================================================
REM  PaperBanana - Windows Launcher
REM  Double-click to start. Auto-installs dependencies on first run.
REM ============================================================

REM --- Config ---
set "PYTHON_MIN_VER=3.10"
set "VENV_DIR=.venv"
set "RUNTIME_DIR=runtime"
set "PORT=8501"
set "APP_NAME=PaperBanana 论文图表助手"

REM --- Enter project directory ---
cd /d "%~dp0"

echo.
echo ==========================================
echo   %APP_NAME%
echo ==========================================
echo.

REM ============================================================
REM Step 1: Find / Install Python
REM ============================================================
set "PYTHON_CMD="

REM 1a. Check runtime\ portable Python
if exist "%RUNTIME_DIR%\python\python.exe" (
    call :check_python_ver "%RUNTIME_DIR%\python\python.exe"
    if !errorlevel! == 0 (
        set "PYTHON_CMD=%RUNTIME_DIR%\python\python.exe"
        echo   [OK] Portable Python found
        goto :found_python
    )
)

REM 1b. Check system Python
call :try_system_python python  && goto :found_python
call :try_system_python python3 && goto :found_python
call :try_system_python py      && goto :found_python

REM 1c. Try winget install (Windows 10 1709+)
where winget >nul 2>&1 || goto :skip_winget
echo   [..] Installing Python 3.12 via winget ...
winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements >nul 2>&1 || goto :winget_failed
REM Refresh PATH after winget install
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
call :try_system_python python && goto :found_python
:winget_failed
echo   [!!] winget install failed, trying portable download ...
:skip_winget

REM 1d. Auto-download portable Python (python-build-standalone)
echo   [..] Python 3.10+ not found, downloading portable Python ...

if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"

REM Use PowerShell to query GitHub API and download
echo   [..] Querying latest version ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ProgressPreference = 'SilentlyContinue'; " ^
    "try { " ^
    "  $releases = Invoke-RestMethod -Uri 'https://api.github.com/repos/indygreg/python-build-standalone/releases?per_page=5' -TimeoutSec 30; " ^
    "  $url = ''; " ^
    "  foreach ($r in $releases) { " ^
    "    foreach ($a in $r.assets) { " ^
    "      if ($a.name -match 'cpython-3\.12.*x86_64-pc-windows-msvc-install_only\.tar\.gz$') { " ^
    "        $url = $a.browser_download_url; break " ^
    "      } " ^
    "    }; " ^
    "    if ($url) { break } " ^
    "  }; " ^
    "  if (-not $url) { Write-Host '  [!!] Download URL not found'; exit 1 }; " ^
    "  Write-Host '  [..] Downloading (~40MB, please wait) ...'; " ^
    "  $ProgressPreference = 'Continue'; " ^
    "  Invoke-WebRequest -Uri $url -OutFile '%RUNTIME_DIR%\python.tar.gz' -UseBasicParsing; " ^
    "  Write-Host '  [..] Extracting ...'; " ^
    "  tar -xzf '%RUNTIME_DIR%\python.tar.gz' -C '%RUNTIME_DIR%\'; " ^
    "  Remove-Item '%RUNTIME_DIR%\python.tar.gz' -Force; " ^
    "  Write-Host '  [OK] Portable Python installed'; " ^
    "} catch { " ^
    "  Write-Host \"  [!!] Download failed: $_\"; exit 1 " ^
    "}"

if exist "%RUNTIME_DIR%\python\python.exe" (
    set "PYTHON_CMD=%RUNTIME_DIR%\python\python.exe"
    goto :found_python
)

REM All methods failed
echo.
echo   [!!] Cannot auto-install Python. Please install manually:
echo.
echo       Option 1: Search "Python 3.12" in Microsoft Store
echo       Option 2: Download from https://www.python.org/downloads/
echo                  Check "Add Python to PATH" during installation
echo.
pause
exit /b 1

:found_python

REM ============================================================
REM Step 2: Create / check virtual environment
REM ============================================================
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo   [..] Creating Python virtual environment ...
    "%PYTHON_CMD%" -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo   [!!] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo   [OK] Virtual environment created
) else (
    echo   [OK] Virtual environment exists
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

REM ============================================================
REM Step 3: Install / update dependencies
REM ============================================================
echo   [..] Installing Python dependencies (first run may be slow) ...
echo.
"%VENV_PYTHON%" -m pip install -r requirements.txt --disable-pip-version-check -i https://pypi.tuna.tsinghua.edu.cn/simple
if !errorlevel! neq 0 (
    echo.
    echo   [!!] Failed to install dependencies
    pause
    exit /b 1
)
echo.
echo   [OK] Dependencies ready

REM ============================================================
REM Step 4: Create data directories
REM ============================================================
if not exist "data\PaperBananaBench\diagram" mkdir "data\PaperBananaBench\diagram"
if not exist "data\PaperBananaBench\plot" mkdir "data\PaperBananaBench\plot"
if not exist "data\PaperBananaBench\diagram\ref.json" (
    >>"data\PaperBananaBench\diagram\ref.json" echo []
)
if not exist "data\PaperBananaBench\plot\ref.json" (
    >>"data\PaperBananaBench\plot\ref.json" echo []
)

REM ============================================================
REM Step 5: Clean up port and launch app
REM ============================================================
echo   [..] Checking port %PORT% ...
for /f "tokens=5" %%A in ('netstat -ano 2^>nul ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo   [!!] Port %PORT% in use ^(PID: %%A^), cleaning up ...
    taskkill /F /PID %%A >nul 2>&1
    timeout /t 1 /nobreak >nul
    echo   [OK] Port released
)

echo.
echo ==========================================
echo   Starting %APP_NAME%
echo   Browser will open http://localhost:%PORT%
echo   Close this window to stop the server
echo ==========================================
echo.

REM Open browser after delay
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:%PORT%"

REM Launch Streamlit
"%VENV_PYTHON%" -m streamlit run demo.py ^
    --server.port %PORT% ^
    --server.address 0.0.0.0 ^
    --server.headless true

pause
exit /b 0

REM ============================================================
REM Subroutine: Check Python version >= 3.10
REM ============================================================
:check_python_ver
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>nul
exit /b !errorlevel!

REM ============================================================
REM Subroutine: Try system Python
REM ============================================================
:try_system_python
where %~1 >nul 2>&1 || exit /b 1
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>nul || exit /b 1
set "PYTHON_CMD=%~1"
for /f "delims=" %%V in ('%~1 --version 2^>^&1') do echo   [OK] System %%V found
exit /b 0
