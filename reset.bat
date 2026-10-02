@echo off
title NEXUS - Reset Training Data
cd /d "%~dp0"
echo.
echo   WARNING: this deletes all custom training data from lora-weights\
echo   NEXUS will revert to the base model.
echo.
venv\Scripts\python.exe -c "from nexus.security import confirm; confirm.set_handler(confirm.console_handler); from nexus.llm.loader import reset_training; print(reset_training())"
echo.
pause
