@echo off
REM Aegis server startup script — run at login or via Task Scheduler
cd /d "D:\Content\Animesh\bots\ai_signal_bot"

REM Resurrect PM2 processes that were saved
call "C:\Users\bkukr\AppData\Roaming\npm\pm2.cmd" resurrect

REM If nothing was running (first time), start from ecosystem config
call "C:\Users\bkukr\AppData\Roaming\npm\pm2.cmd" list | findstr "online" >nul 2>&1
if errorlevel 1 (
    call "C:\Users\bkukr\AppData\Roaming\npm\pm2.cmd" start ecosystem.config.js
)
