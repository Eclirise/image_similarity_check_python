@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 bootstrap.py install
) else (
  python bootstrap.py install
)

if errorlevel 1 (
  echo.
  echo Install failed. See the output above.
  pause
  exit /b 1
)

echo.
echo Install finished.
endlocal
