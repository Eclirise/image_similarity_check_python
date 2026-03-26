from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

from image_similarity_check_python import (
    default_model_cache_dir,
    guess_device,
    hardware_summary,
    move_to_trash,
    open_file_location,
    run_ab_compare_job,
    run_reference_job,
    run_scan_job,
)

StatusCb = Optional[Callable[[str], None]]
ProgressCb = Optional[Callable[[int, int, str], None]]


def build_common_args(
    *,
    device: str = "auto",
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "laion2b_s34b_b79k",
    clip_mirror: str = "auto",
    clip_endpoint: str = "",
    model_cache_dir: str | None = None,
    no_openclip: bool = False,
    extensions: str = "jpg,jpeg,png,bmp,webp,tif,tiff,gif",
) -> argparse.Namespace:
    return argparse.Namespace(
        mode="gui",
        device=device,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        clip_mirror=clip_mirror,
        clip_endpoint=clip_endpoint,
        model_cache_dir=model_cache_dir or str(default_model_cache_dir()),
        no_openclip=no_openclip,
        workers=None,
        batch_size=None,
        extensions=extensions,
        tab="compare",
    )


def run_scan(
    base: argparse.Namespace,
    *,
    input_dir: str,
    output_xlsx: str | None,
    min_sim: float,
    max_sim: float,
    top_k: int,
    non_recursive: bool,
    no_openclip: bool,
    status_cb: StatusCb = None,
    progress_cb: ProgressCb = None,
) -> dict:
    args = argparse.Namespace(**vars(base))
    args.input_dir = input_dir
    args.output_xlsx = output_xlsx
    args.min_sim = min_sim
    args.max_sim = max_sim
    args.top_k = top_k
    args.non_recursive = non_recursive
    args.no_openclip = no_openclip
    return run_scan_job(args, status_cb=status_cb, progress_cb=progress_cb)


def run_reference(
    base: argparse.Namespace,
    *,
    reference_image: str,
    search_dir: str,
    output_txt: str | None,
    min_sim: float,
    max_sim: float,
    non_recursive: bool,
    no_openclip: bool,
    status_cb: StatusCb = None,
    progress_cb: ProgressCb = None,
) -> dict:
    args = argparse.Namespace(**vars(base))
    args.reference_image = reference_image
    args.search_dir = search_dir
    args.output_txt = output_txt
    args.min_sim = min_sim
    args.max_sim = max_sim
    args.non_recursive = non_recursive
    args.no_openclip = no_openclip
    return run_reference_job(args, status_cb=status_cb, progress_cb=progress_cb)


def run_compare(
    base: argparse.Namespace,
    *,
    dir_a: str,
    dir_b: str,
    min_sim: float,
    max_sim: float,
    top_k_per_a: int,
    non_recursive: bool,
    no_openclip: bool,
    status_cb: StatusCb = None,
    progress_cb: ProgressCb = None,
) -> dict:
    args = argparse.Namespace(**vars(base))
    args.dir_a = dir_a
    args.dir_b = dir_b
    args.min_sim = min_sim
    args.max_sim = max_sim
    args.top_k_per_a = top_k_per_a
    args.non_recursive = non_recursive
    args.no_openclip = no_openclip
    return run_ab_compare_job(args, status_cb=status_cb, progress_cb=progress_cb)


def open_location(path: str) -> None:
    open_file_location(path)


def safe_delete(path: str) -> str:
    return move_to_trash(path)


def normalize_recent_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def runtime_summary(device: str) -> str:
    selected = guess_device(device)
    return f"计算设备: {selected} | {hardware_summary(selected)}"
