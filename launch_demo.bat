@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
title Vietnamese Legal RAG - Instant Demo Launcher
echo ======================================================================
echo  Dang khoi dong he thong Vietnamese Legal RAG...
echo ======================================================================
.\.venv\Scripts\python.exe launch_demo.py
pause
