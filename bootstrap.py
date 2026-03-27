from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"

PIP_MIRRORS = [
    {"name": "清华", "url": "https://pypi.tuna.tsinghua.edu.cn/simple"},
    {"name": "中科大", "url": "https://mirrors.ustc.edu.cn/pypi/web/simple"},
    {"name": "腾讯", "url": "https://mirrors.cloud.tencent.com/pypi/simple"},
    {"name": "PyPI 官方", "url": "https://pypi.org/simple"},
]

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "PIL": "Pillow",
    "openpyxl": "openpyxl",
    "send2trash": "send2trash",
    "open_clip": "open-clip-torch",
    "torch": "torch",
}


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def running_in_local_venv() -> bool:
    current = Path(sys.executable).resolve()
    expected = venv_python().resolve()
    return current == expected


def stream_command(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        _write_console_line(line.rstrip())
    return process.wait()


def _write_console_line(text: str) -> None:
    payload = (text + "\n").encode(sys.stdout.encoding or "utf-8", errors="replace")
    sys.stdout.buffer.write(payload)
    sys.stdout.flush()


def ensure_venv_created() -> Path:
    python_path = venv_python()
    if python_path.exists():
        return python_path

    print("[准备] 正在创建本地虚拟环境 .venv", flush=True)
    exit_code = stream_command([sys.executable, "-m", "venv", str(VENV_DIR)])
    if exit_code != 0 or not python_path.exists():
        raise SystemExit("虚拟环境创建失败。")
    return python_path


def reexec_into_venv(extra_args: list[str]) -> None:
    target = ensure_venv_created()
    exit_code = subprocess.call([str(target), str(REPO_ROOT / "run.py"), *extra_args], cwd=str(REPO_ROOT))
    raise SystemExit(exit_code)


def missing_dependencies() -> list[str]:
    missing = []
    for module_name, package_name in REQUIRED_IMPORTS.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    return missing


def _pip_install_with_mirror(python_exe: Path, mirror: dict[str, str], *, upgrade_pip: bool) -> int:
    if upgrade_pip:
        print(f"[安装] 正在升级 pip，镜像：{mirror['name']}  {mirror['url']}", flush=True)
        upgrade_cmd = [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "-i",
            mirror["url"],
            "--disable-pip-version-check",
            "--timeout",
            "60",
        ]
        upgrade_code = stream_command(upgrade_cmd)
        if upgrade_code != 0:
            return upgrade_code

    print(f"[安装] 正在安装依赖，镜像：{mirror['name']}  {mirror['url']}", flush=True)
    install_cmd = [
        str(python_exe),
        "-m",
        "pip",
        "install",
        "-r",
        str(REQUIREMENTS_FILE),
        "-i",
        mirror["url"],
        "--prefer-binary",
        "--disable-pip-version-check",
        "--timeout",
        "60",
        "--progress-bar",
        "on",
    ]
    return stream_command(install_cmd)


def install_dependencies(*, force: bool = False) -> str:
    python_exe = ensure_venv_created()
    if not REQUIREMENTS_FILE.exists():
        raise FileNotFoundError(f"未找到依赖文件：{REQUIREMENTS_FILE}")

    if not force and not missing_dependencies():
        print("[完成] 当前环境依赖已齐全，无需重复安装。", flush=True)
        return "已安装"

    last_error = None
    for index, mirror in enumerate(PIP_MIRRORS):
        exit_code = _pip_install_with_mirror(
            python_exe,
            mirror,
            upgrade_pip=index == 0,
        )
        if exit_code == 0:
            print(f"[完成] 依赖安装成功：{mirror['name']}  {mirror['url']}", flush=True)
            return mirror["url"]
        last_error = f"{mirror['name']} 退出码 {exit_code}"
        print(f"[提示] 当前镜像失败，继续尝试下一个。", flush=True)

    raise SystemExit(
        "依赖安装失败。已尝试全部镜像："
        + " -> ".join(f"{mirror['name']}({mirror['url']})" for mirror in PIP_MIRRORS)
        + f"。最后状态：{last_error}"
    )


def cli() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "install":
        install_dependencies(force=True)
        return 0
    print("用法: python bootstrap.py install")
    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
