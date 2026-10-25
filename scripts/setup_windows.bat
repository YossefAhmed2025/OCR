@echo off
REM ====================================================================
REM  Phonics Prism OCR -- Windows setup
REM
REM  Run from the PROJECT ROOT:
REM      scripts\setup_windows.bat
REM      scripts\setup_windows.bat cpu     (force the CPU build)
REM      scripts\setup_windows.bat gpu     (force the GPU build)
REM
REM  With no argument the script reads the NVIDIA driver version and
REM  picks a PaddlePaddle build the driver actually supports.
REM ====================================================================
setlocal EnableDelayedExpansion

if not exist "scripts\setup_windows.bat" (
    echo ERROR: run this from the project root, not from inside scripts\.
    echo     cd ^<project root^>
    echo     scripts\setup_windows.bat
    exit /b 1
)

set "MODE=%~1"
if "%MODE%"=="" set "MODE=auto"

echo.
echo ======================================================================
echo  Phonics Prism OCR -- Windows setup  [mode: %MODE%]
echo ======================================================================

REM -- 1. Find a supported 64-bit Python ------------------------------
echo.
echo [1/6] Locating Python...
set "PY_CMD="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY_CMD (
        py -%%V -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>&1
        if !errorlevel! equ 0 (
            set "PY_CMD=py -%%V"
            echo       found 64-bit Python %%V
        )
    )
)
if not defined PY_CMD (
    python -c "import sys;sys.exit(0 if sys.version_info[:2]>=(3,9) and sys.version_info[:2]<=(3,13) and sys.maxsize>2**32 else 1)" >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY_CMD=python"
        echo       using 'python' from PATH
    )
)
if not defined PY_CMD (
    echo ERROR: no supported 64-bit Python found.
    echo        PaddlePaddle needs 64-bit Python 3.9 - 3.13.
    echo        Download: https://www.python.org/downloads/windows/
    exit /b 1
)

REM -- 2. Virtual environment -----------------------------------------
echo.
echo [2/6] Creating the virtual environment...
if exist ".venv\Scripts\python.exe" (
    echo       .venv already exists; reusing it.
) else (
    %PY_CMD% -m venv .venv
    if errorlevel 1 ( echo ERROR: venv creation failed. & exit /b 1 )
    echo       created .venv
)
set "VPY=.venv\Scripts\python.exe"

echo.
echo [3/6] Upgrading pip...
"%VPY%" -m pip install --upgrade pip setuptools wheel --quiet
if errorlevel 1 ( echo ERROR: pip upgrade failed. & exit /b 1 )

REM -- 4. Choose CPU or GPU -------------------------------------------
echo.
echo [4/6] Selecting the PaddlePaddle build...

if /i "%MODE%"=="cpu" goto :install_cpu
if /i "%MODE%"=="gpu" goto :install_gpu

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo       nvidia-smi not found -- no usable NVIDIA GPU.
    goto :install_cpu
)

for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do set "DRV_MAJOR=%%a"
if not defined DRV_MAJOR (
    echo       could not read the driver version; defaulting to CPU.
    goto :install_cpu
)
echo       NVIDIA driver major version: %DRV_MAJOR%

REM cu118 needs Windows driver >= 452.39; cu126/cu129 need >= 550.54.14.
REM Only cu118 is used here, so 452 is the gate. See requirements-gpu.txt.
if %DRV_MAJOR% GEQ 452 (
    echo       driver supports the CUDA 11.8 build.
    goto :install_gpu
)
echo       driver %DRV_MAJOR% is older than 452 -- no CUDA build is supported.
goto :install_cpu

:install_gpu
echo.
echo [5/6] Installing paddlepaddle-gpu (CUDA 11.8)...
echo       This downloads roughly 1.5 GB and needs about 4 GB on disk.
"%VPY%" -m pip uninstall -y paddlepaddle >nul 2>&1
"%VPY%" -m pip install -r requirements-gpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
if errorlevel 1 (
    echo.
    echo ERROR: the GPU build failed to install.
    echo        Retry on CPU:  scripts\setup_windows.bat cpu
    exit /b 1
)
goto :install_common

:install_cpu
echo.
echo [5/6] Installing paddlepaddle (CPU)...
"%VPY%" -m pip uninstall -y paddlepaddle-gpu >nul 2>&1
"%VPY%" -m pip install -r requirements-cpu.txt -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
if errorlevel 1 ( echo ERROR: the CPU build failed to install. & exit /b 1 )
goto :install_common

:install_common
echo.
echo [6/6] Installing PaddleOCR and the remaining dependencies...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 ( echo ERROR: dependency installation failed. & exit /b 1 )

echo.
echo ======================================================================
echo  Install finished. Verifying...
echo ======================================================================
"%VPY%" scripts\check_environment.py

echo.
echo ======================================================================
echo  Next steps
echo ======================================================================
echo   1. Activate the environment:
echo        .venv\Scripts\activate
echo   2. Download the models (needs internet, once):
echo        python scripts\download_models.py
echo   3. Confirm offline inference works:
echo        python scripts\verify_offline.py
echo   4. Put page images in data\input\ and run:
echo        python scripts\run_ocr.py --image data\input\your_page.jpg --lang ar
echo.
endlocal
