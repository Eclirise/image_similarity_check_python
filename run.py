from __future__ import annotations

import argparse
import sys

import bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True, description="图片相似度工作台启动器")
    parser.add_argument("--check", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--install-deps-only", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bootstrap.ensure_venv_created()
    if not bootstrap.running_in_local_venv():
        bootstrap.reexec_into_venv(sys.argv[1:])
        return

    if args.install_deps_only:
        bootstrap.install_dependencies(force=True)
        return

    if bootstrap.missing_dependencies():
        bootstrap.install_dependencies(force=False)

    if args.check:
        import desktop_ui  # noqa: F401
        import image_similarity_check_python  # noqa: F401

        print("环境检查通过", flush=True)
        return

    from desktop_ui import launch_app

    launch_app()


if __name__ == "__main__":
    main()
