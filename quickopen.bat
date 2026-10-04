@echo off
title NEXUS - Quick Menu
cd /d "%~dp0"

if not exist "venv\Scripts\pythonw.exe" (
    echo Virtual environment not found.
    echo Run setup.bat first.
    pause
    exit /b 1
)

echo Starting NEXUS...
start "" venv\Scripts\pythonw.exe -m nexus
