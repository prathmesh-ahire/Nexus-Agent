@echo off
title NEXUS - Training Pipeline
cd /d "%~dp0"
echo Put your dataset .txt files in datasets\ first.
echo For a faster run: train.bat --quick
echo.
venv\Scripts\python.exe -m nexus.training.trainer %*
echo.
pause
