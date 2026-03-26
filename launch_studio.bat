@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set "LOG_DIR=%~dp0logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "LOG_FILE=%LOG_DIR%\launch_%DATE:~0,4%%DATE:~5,2%%DATE:~8,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%.log"
set "LOG_FILE=%LOG_FILE: =0%"
echo ==== [%DATE% %TIME%] 启动开始 ==== > "%LOG_FILE%"

set "PY_LAUNCHER="
where py >nul 2>nul
if !errorlevel! equ 0 set "PY_LAUNCHER=py -3"
if not defined PY_LAUNCHER (
  where python >nul 2>nul
  if !errorlevel! equ 0 set "PY_LAUNCHER=python"
)
if not defined PY_LAUNCHER (
  echo [ERROR] 未找到 Python 启动器（py/python）。>> "%LOG_FILE%"
  echo [ERROR] 未找到 Python 3，请先安装 Python 3.10+ 后重试。
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [INFO] 正在创建本地虚拟环境 .venv ...
  %PY_LAUNCHER% -m venv .venv >> "%LOG_FILE%" 2>&1
  if !errorlevel! neq 0 goto :error
)

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] 未找到 Python 解释器，请先安装 Python 3.10+.
  pause
  exit /b 1
)

echo [INFO] 正在升级 pip ...
"%PYTHON_EXE%" -m pip install --upgrade pip >> "%LOG_FILE%" 2>&1
if !errorlevel! neq 0 goto :error

set "REQ=requirements.txt"
if not exist "%REQ%" (
  echo [ERROR] 未找到 requirements.txt
  pause
  exit /b 1
)

set "INSTALLED=0"
for %%I in (https://pypi.tuna.tsinghua.edu.cn/simple https://pypi.org/simple) do (
  echo [INFO] 尝试安装依赖源: %%I
  "%PYTHON_EXE%" -m pip install -r "%REQ%" -i %%I --prefer-binary --disable-pip-version-check >> "%LOG_FILE%" 2>&1
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
  "%PYW_EXE%" -m app.main >> "%LOG_FILE%" 2>&1
  if !errorlevel! neq 0 (
    echo [WARN] pythonw 启动失败，切换到控制台模式... >> "%LOG_FILE%"
    "%PYTHON_EXE%" -m app.main >> "%LOG_FILE%" 2>&1
    if !errorlevel! neq 0 goto :error
  )
) else (
  "%PYTHON_EXE%" -m app.main >> "%LOG_FILE%" 2>&1
  if !errorlevel! neq 0 goto :error
)
endlocal
exit /b 0

:error
echo [ERROR] 启动失败，详细日志：%LOG_FILE%
echo [ERROR] 启动失败，详细日志：%LOG_FILE%>> "%LOG_FILE%"
echo.
echo ===== 最近错误日志（末尾 40 行）=====
powershell -NoProfile -Command "Get-Content -Path '%LOG_FILE%' -Tail 40"
echo ======================================
pause
exit /b 1
