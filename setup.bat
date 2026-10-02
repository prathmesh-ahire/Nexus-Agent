@echo off
title NEXUS - Setup
cd /d "%~dp0"

echo ============================================
echo   NEXUS - Setup
echo ============================================
echo.

where py >nul 2>&1
if errorlevel 1 (
    echo Python launcher not found. Install Python 3.11+ from python.org
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment...
py -3.11 -m venv venv || python -m venv venv

echo [2/3] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip -q

echo [3/3] Installing NEXUS and dependencies...
echo       (llama-cpp-python uses a prebuilt CPU wheel - no compiler needed)
venv\Scripts\python.exe -m pip install -e ".[web,hotkey,dev]" ^
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

echo.
echo ============================================
echo   Setup complete.
echo.
echo   Next: place the GGUF model in model\
echo     qwen2.5-3b-instruct-q4_k_m.gguf
echo     https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
echo.
echo   Then run quickopen.bat
echo ============================================
pause
