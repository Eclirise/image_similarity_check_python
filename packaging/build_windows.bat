@echo off
setlocal
cd /d "%~dp0\.."

call install_dependencies.bat
if errorlevel 1 exit /b 1

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  echo Virtual environment Python was not found.
  pause
  exit /b 1
)

"%PY%" -m pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple --disable-pip-version-check
if errorlevel 1 "%PY%" -m pip install pyinstaller -i https://mirrors.ustc.edu.cn/pypi/web/simple --disable-pip-version-check
if errorlevel 1 "%PY%" -m pip install pyinstaller -i https://mirrors.cloud.tencent.com/pypi/simple --disable-pip-version-check
if errorlevel 1 "%PY%" -m pip install pyinstaller -i https://pypi.org/simple --disable-pip-version-check
if errorlevel 1 (
  echo PyInstaller install failed.
  pause
  exit /b 1
)

"%PY%" -m PyInstaller --noconfirm --noconsole --name ImageSimilarityStudio run.py

echo.
echo Build finished: dist\ImageSimilarityStudio\ImageSimilarityStudio.exe
endlocal
