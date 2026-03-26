@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] 正在创建本地虚拟环境 .venv ...
  py -3 -m venv .venv
)

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] 未找到 Python 解释器，请先安装 Python 3.10+.
  pause
  exit /b 1
)

echo [INFO] 正在升级 pip ...
"%PYTHON_EXE%" -m pip install --upgrade pip >nul 2>nul

set "REQ=requirements.txt"
if not exist "%REQ%" (
  echo [ERROR] 未找到 requirements.txt
  pause
  exit /b 1
)

set "INSTALLED=0"
for %%I in (https://pypi.tuna.tsinghua.edu.cn/simple https://pypi.org/simple) do (
  echo [INFO] 尝试安装依赖源: %%I
  "%PYTHON_EXE%" -m pip install -r "%REQ%" -i %%I --prefer-binary --disable-pip-version-check
  if !errorlevel! equ 0 (
    set "INSTALLED=1"
    echo [INFO] 依赖安装成功: %%I
    goto run_app
  )
)

if "%INSTALLED%"=="0" (
  echo [ERROR] 依赖安装失败，请检查网络或手动执行 pip 安装。
  pause
  exit /b 1
)

:run_app
echo [INFO] 正在启动图片相似度工作台...
set "PYW_EXE=.venv\Scripts\pythonw.exe"
if exist "%PYW_EXE%" (
  start "" "%PYW_EXE%" -m app.main
) else (
  "%PYTHON_EXE%" -m app.main
)
endlocal
