@echo off
chcp 65001 >nul
title SMS Spam Detector

REM Chuyen den thu muc chua file BAT
cd /d "%~dp0"

REM Tim Python cua du an: uu tien eval\.venv (tao bang eval\setup_env.py), sau do .venv
set "PY=eval\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Khong tim thay Python trong eval\.venv hoac .venv
    pause
    exit /b 1
)

REM Mo trinh duyet va khoi dong website
start "" http://127.0.0.1:5000
"%PY%" app.py

pause
