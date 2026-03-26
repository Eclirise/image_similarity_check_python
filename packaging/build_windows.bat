@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
)

set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --prefer-binary || "%PY%" -m pip install -r requirements.txt -i https://pypi.org/simple --prefer-binary
"%PY%" -m pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple || "%PY%" -m pip install pyinstaller -i https://pypi.org/simple
"%PY%" -m PyInstaller --noconfirm --noconsole --name ImageSimilarityStudio app/main.py

echo 打包完成：dist\ImageSimilarityStudio\ImageSimilarityStudio.exe
endlocal
