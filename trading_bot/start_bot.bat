@echo off
chcp 65001 >NUL 2>&1
REM ============================================
REM BTC Trading Bot Auto-Start Script
REM ============================================
REM Register in Windows Task Scheduler:
REM   1. Win+R, type taskschd.msc, press Enter
REM   2. Click "Create Task"
REM   3. General tab: Name = BTC Trading Bot
REM   4. Triggers tab: New -> At log on
REM   5. Actions tab: New -> Start a program
REM      Program: full path to this .bat file
REM   6. Conditions tab: Uncheck AC power only
REM   7. Settings tab: Restart on failure (1min, 3x)
REM ============================================

cd /d "C:\Users\Administrator\Desktop\macd_cycle_analysis_project\trading_bot"

tasklist /FI "WINDOWTITLE eq BTC_Trading_Bot" 2>NUL | find /I "python" >NUL
if %ERRORLEVEL% EQU 0 (
    echo [%date% %time%] Bot already running. Exit.
    exit /b 0
)

title BTC_Trading_Bot

:loop
echo [%date% %time%] Starting BTC Trading Bot...
python main.py

echo [%date% %time%] Bot stopped. Restarting in 10s...
timeout /t 10 /nobreak >NUL
goto loop
