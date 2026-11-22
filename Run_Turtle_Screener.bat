@echo off
title Turtle Trend Following US Stock Ranking Screener
cd /d "%~dp0"
if exist "%~dp0Run_Turtle_Screener.exe" (
    "%~dp0Run_Turtle_Screener.exe" %*
) else (
    ".\.venv\Scripts\python.exe" screener.py %*
    ".\.venv\Scripts\python.exe" app.py
)
