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
echo Press Ctrl+Alt+N to show or hide the window.
start "" venv\Scripts\pythonw.exe -m nexus
