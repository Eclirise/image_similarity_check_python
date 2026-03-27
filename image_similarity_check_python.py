#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

try:
    from blake3 import blake3 as _blake3
except Exception:
    _blake3 = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

try:
    from scipy.fft import dctn as _dctn
except Exception:
    _dctn = None

_OPEN_CLIP_IMPORT_ERROR = None
try:
    import torch
    import open_clip
except Exception as e:
    torch = None  # type: ignore
    open_clip = None  # type: ignore
    _OPEN_CLIP_IMPORT_ERROR = e

try:
    from send2trash import send2trash as _send2trash
except Exception:
    _send2trash = None

try:
    import bootstrap
except Exception:
    bootstrap = None  # type: ignore

APP_NAME = "图片相似度工作台"
APP_VERSION = "2026 Desktop Studio Light · Compare Pro"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
PIP_INDEX_CANDIDATES = [mirror["url"] for mirror in getattr(bootstrap, "PIP_MIRRORS", [])] or [
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.ustc.edu.cn/pypi/web/simple",
    "https://mirrors.cloud.tencent.com/pypi/simple",
    "https://pypi.org/simple",
]
REPO_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_CANDIDATES = [
    REPO_ROOT / "open_clip_model.safetensors",
    REPO_ROOT / "open_clip_pytorch_model.bin",
]
MATCH_TYPE_LABELS = {
    "exact_file": "文件完全一致",
    "exact_pixels": "像素完全一致",
    "near_duplicate": "高度相似",
    "strong_match": "明显相似",
    "possible_match": "可能相似",
}

StatusCallback = Optional[Callable[[str], None]]
ProgressCallback = Optional[Callable[[int, int, str], None]]


@dataclass
class ImageRecord:
    idx: int
    path: str
    rel_path: str
    file_size: int
    width: int
    height: int
    file_hash: str
    pixel_hash: str
    dhash: int
    phash: int
    feature: np.ndarray
    embedding_backend: str
    error: Optional[str] = None


def normalize_path(p: Path) -> str:
    return os.path.normcase(os.path.abspath(str(p.expanduser())))


def absolute_path(p: Path) -> Path:
    return Path(os.path.abspath(str(p.expanduser())))


def clean_input_path(raw: str) -> str:
    return raw.strip().strip('"').strip("'")


def chunked(seq: Sequence, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def cosine_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def file_digest(path: str, block_size: int = 1024 * 1024) -> str:
    if _blake3 is not None:
        hasher = _blake3(max_threads=os.cpu_count() or 1)
    else:
        hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def open_image_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        else:
            im = im.copy()
    return im


def pixel_digest(im: Image.Image) -> str:
    h = _blake3() if _blake3 is not None else hashlib.sha256()
    h.update(f"{im.mode}|{im.size[0]}|{im.size[1]}".encode("utf-8"))
    h.update(im.tobytes())
    return h.hexdigest()


def dhash64(im: Image.Image, hash_size: int = 8) -> int:
    small = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten().astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return int(value)


def phash64(im: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    if _dctn is None:
        return dhash64(im, hash_size=hash_size)

    size = hash_size * highfreq_factor
    small = im.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    dct = _dctn(arr, type=2, norm="ortho")
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, :].flatten()) if low.size > 1 else 0.0
    bits = (low > med).flatten().astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return int(value)


def hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def bit_similarity64(a: int, b: int) -> float:
    return 1.0 - (hamming64(a, b) / 64.0)


def lightweight_feature(im: Image.Image) -> np.ndarray:
    rgb = im.resize((24, 24), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0

    gx = arr[:, 1:, :] - arr[:, :-1, :]
    gy = arr[1:, :, :] - arr[:-1, :, :]
    gx = np.pad(gx, ((0, 0), (0, 1), (0, 0)))
    gy = np.pad(gy, ((0, 1), (0, 0), (0, 0)))

    hists = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=12, range=(0.0, 1.0), density=True)
        hists.append(hist.astype(np.float32))
    hist_arr = np.concatenate(hists, axis=0)

    feat = np.concatenate([arr.reshape(-1), gx.reshape(-1), gy.reshape(-1), hist_arr], axis=0).astype(np.float32)
    return cosine_normalize(feat)


def list_images(root: Path, recursive: bool = True, extensions: Optional[set] = None) -> List[Path]:
    extensions = extensions or VALID_EXTENSIONS
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    else:
        files = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in extensions]
    files.sort()
    return files


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(absolute_path(path).relative_to(absolute_path(root)))
    except Exception:
        return str(absolute_path(path))


def all_pairs(indices: Sequence[int]) -> Iterable[Tuple[int, int]]:
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            yield indices[i], indices[j]


def sanitize_filename(text: str, fallback: str = "report") -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\s]+", "_", text).strip("._")
    return cleaned or fallback


def guess_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def recommended_workers() -> int:
    cpu_threads = os.cpu_count() or 4
    return max(4, min(cpu_threads, 8))


def recommended_batch_size(device: str) -> int:
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0).lower()
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # Pascal-era laptop GPUs such as GTX 1070 are common in this project.
            # A slightly smaller default batch avoids long first-run stalls and OOM risk.
            if any(token in device_name for token in ("gtx 1060", "gtx 1070", "gtx 1080", "p10")):
                return 12
            if total_mem_gb >= 12:
                return 32
            if total_mem_gb >= 8:
                return 16
            if total_mem_gb >= 6:
                return 12
            return 12
        except Exception:
            return 12
    return 8


def hardware_summary(device: str) -> str:
    cpu_threads = os.cpu_count() or 1
    lines = [f"CPU 线程: {cpu_threads}"]
    if device == "cuda" and torch is not None and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            lines.append(f"GPU: {name} ({mem:.1f} GB)")
        except Exception:
            lines.append("GPU: CUDA 可用")
    else:
        lines.append("GPU: 未启用，使用 CPU")
    return " | ".join(lines)


def make_default_output_path(input_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return input_dir / f"image_similarity_report_{ts}.xlsx"


def make_default_txt_report_path(search_dir: Path, reference_image: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = sanitize_filename(reference_image.stem, fallback="reference")
    return search_dir / f"similar_to_{stem}_{ts}.txt"


def resolve_clip_pretrained(pretrained: str) -> str:
    candidate = Path(pretrained).expanduser()
    if candidate.exists() and candidate.is_file():
        return str(candidate.resolve())

    for local_candidate in LOCAL_MODEL_CANDIDATES:
        if local_candidate.exists() and local_candidate.is_file():
            return str(local_candidate.resolve())
    return pretrained


def default_model_cache_dir() -> Path:
    return Path.home() / '.image_similarity_workbench' / 'hf_cache'


def install_runtime_dependencies(
    requirements_file: Path,
    status_cb: StatusCallback = None,
) -> str:
    if not requirements_file.exists():
        raise FileNotFoundError(f"未找到依赖清单：{requirements_file}")
    if bootstrap is None:
        raise RuntimeError("安装模块不可用，无法执行依赖安装。")
    emit_status(status_cb, "正在安装依赖，过程会实时显示。")
    return bootstrap.install_dependencies(force=True)


def get_hf_endpoint_candidates(mirror_mode: str = 'auto', custom_endpoint: str = '') -> List[str]:
    if custom_endpoint.strip():
        return [custom_endpoint.strip()]
    mode = (mirror_mode or 'auto').strip().lower()
    if mode in {'cn', 'china', 'mainland', 'hf-mirror'}:
        return ['https://hf-mirror.com', 'https://huggingface.co']
    if mode in {'official', 'global', 'none'}:
        return ['https://huggingface.co']
    return ['https://hf-mirror.com', 'https://huggingface.co']


def configure_model_download_env(cache_dir: str | None, endpoint: str | None = None) -> dict:
    cache_path = Path(cache_dir).expanduser() if cache_dir else default_model_cache_dir()
    cache_path.mkdir(parents=True, exist_ok=True)
    hub_path = cache_path / 'hub'
    hub_path.mkdir(parents=True, exist_ok=True)
    updates = {
        'HF_HOME': str(cache_path),
        'HF_HUB_CACHE': str(hub_path),
        'HUGGINGFACE_HUB_CACHE': str(hub_path),
    }
    if endpoint:
        updates['HF_ENDPOINT'] = endpoint
    return updates


def emit_status(cb: StatusCallback, message: str) -> None:
    if cb is not None:
        cb(message)


def emit_progress(cb: ProgressCallback, current: int, total: int, stage: str) -> None:
    if cb is not None:
        cb(current, total, stage)


def run_with_status_heartbeat(
    action: Callable[[], Tuple[object, object, object]],
    *,
    status_cb: StatusCallback,
    waiting_message: str,
    success_message: str,
) -> Tuple[object, object, object]:
    done = threading.Event()

    def heartbeat() -> None:
        waited_seconds = 0
        while not done.wait(1.5):
            waited_seconds += 2
            emit_status(status_cb, f"{waiting_message}\n已等待约 {waited_seconds} 秒")

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        return action()
    finally:
        done.set()
        emit_status(status_cb, success_message)


def compute_openclip_embeddings(
    paths: List[str],
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: str,
    mirror_mode: str = 'auto',
    custom_endpoint: str = '',
    model_cache_dir: str | None = None,
    status_cb: StatusCallback = None,
    progress_cb: ProgressCallback = None,
) -> Tuple[np.ndarray, str]:
    if torch is None or open_clip is None:
        raise RuntimeError(f"open_clip / torch 不可用: {_OPEN_CLIP_IMPORT_ERROR!r}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    pretrained_spec = resolve_clip_pretrained(pretrained)
    model = None
    preprocess = None
    chosen_endpoint = 'local-file'
    previous_env = {}
    if Path(pretrained_spec).exists():
        emit_status(status_cb, f"正在加载本地 OpenCLIP 模型…\n{pretrained_spec}")
        model, _, preprocess = run_with_status_heartbeat(
            lambda: open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained_spec,
                device=device,
            ),
            status_cb=status_cb,
            waiting_message="正在加载本地模型，请稍候…",
            success_message="本地模型加载完成。",
        )
    else:
        endpoints = get_hf_endpoint_candidates(mirror_mode=mirror_mode, custom_endpoint=custom_endpoint)
        last_error = None
        for endpoint in endpoints:
            env_updates = configure_model_download_env(model_cache_dir, endpoint=endpoint)
            previous_env = {k: os.environ.get(k) for k in env_updates}
            os.environ.update(env_updates)
            chosen_endpoint = endpoint
            emit_status(status_cb, f"正在加载 OpenCLIP 模型…\n镜像/源：{endpoint}\n缓存目录：{env_updates['HF_HUB_CACHE']}")
            try:
                model, _, preprocess = run_with_status_heartbeat(
                    lambda: open_clip.create_model_and_transforms(
                        model_name=model_name,
                        pretrained=pretrained_spec,
                        device=device,
                    ),
                    status_cb=status_cb,
                    waiting_message=f"正在从 {endpoint} 获取模型，请稍候…",
                    success_message=f"模型已从 {endpoint} 加载完成。",
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                emit_status(status_cb, f"模型源连接失败，准备尝试下一个源…\n失败源：{endpoint}\n原因：{e}")
        if model is None or preprocess is None:
            raise RuntimeError(f"OpenCLIP 模型加载失败。已尝试源：{', '.join(endpoints)}。最后错误：{last_error!r}")
    model.eval()

    feats = []
    processed = 0
    total = len(paths)
    with torch.inference_mode():
        for batch_paths in chunked(paths, batch_size):
            images = []
            for p in batch_paths:
                im = open_image_rgb(p)
                images.append(preprocess(im))
            batch = torch.stack(images)
            if device == "cuda":
                batch = batch.pin_memory().to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    emb = model.encode_image(batch)
            else:
                batch = batch.to(device)
                emb = model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.float().cpu().numpy().astype(np.float32))
            processed += len(batch_paths)
            emit_progress(progress_cb, processed, total, "正在提取深度特征")
    return np.vstack(feats), f"open_clip:{model_name}/{pretrained_spec} @ {chosen_endpoint}"


def compute_core_metadata(
    idx: int,
    path: Path,
    root: Path,
) -> Tuple[int, str, str, int, int, int, str, str, int, int, np.ndarray, Optional[str]]:
    abs_path = normalize_path(path)
    rel_path = safe_relpath(path, root)
    try:
        file_size = path.stat().st_size
        fhash = file_digest(abs_path)
        im = open_image_rgb(abs_path)
        width, height = im.size
        pxhash = pixel_digest(im)
        dh = dhash64(im)
        ph = phash64(im)
        fallback_feature = lightweight_feature(im)
        return idx, abs_path, rel_path, file_size, width, height, fhash, pxhash, dh, ph, fallback_feature, None
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return idx, abs_path, rel_path, 0, 0, 0, "", "", 0, 0, np.zeros(1, dtype=np.float32), str(e)


def build_records(
    files: List[Path],
    root: Path,
    workers: int,
    use_openclip: bool,
    clip_model: str,
    clip_pretrained: str,
    batch_size: int,
    device: str,
    clip_mirror: str,
    clip_endpoint: str,
    model_cache_dir: str | None,
    status_cb: StatusCallback = None,
    progress_cb: ProgressCallback = None,
) -> Tuple[List[ImageRecord], List[Tuple[str, str]]]:
    records_raw = []
    errors = []

    emit_status(status_cb, "正在读取图片与计算基础特征…")
    done = 0
    total = len(files)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(compute_core_metadata, idx, p, root) for idx, p in enumerate(files)]
        for fut in cf.as_completed(futures):
            records_raw.append(fut.result())
            done += 1
            if done == total or done % max(1, total // 50 or 1) == 0:
                emit_progress(progress_cb, done, total, "正在分析图片")

    records_raw.sort(key=lambda x: x[0])

    valid_paths = []
    valid_positions = []
    for row in records_raw:
        if row[-1] is None:
            valid_positions.append(row[0])
            valid_paths.append(row[1])
        else:
            errors.append((row[1], row[-1]))

    clip_embeddings = None
    backend_name = "lightweight_fallback"

    if use_openclip and valid_paths:
        try:
            clip_embeddings, backend_name = compute_openclip_embeddings(
                paths=valid_paths,
                model_name=clip_model,
                pretrained=clip_pretrained,
                batch_size=batch_size,
                device=device,
                mirror_mode=clip_mirror,
                custom_endpoint=clip_endpoint,
                model_cache_dir=model_cache_dir,
                status_cb=status_cb,
                progress_cb=progress_cb,
            )
        except Exception as e:
            errors.append(("__openclip__", f"OpenCLIP 初始化/推理失败，已自动降级到轻量特征: {e}"))
            emit_status(status_cb, "OpenCLIP 不可用，已自动切换到轻量特征模式。")
            clip_embeddings = None

    pos2clip: Dict[int, np.ndarray] = {}
    if clip_embeddings is not None:
        for pos, emb in zip(valid_positions, clip_embeddings):
            pos2clip[pos] = cosine_normalize(emb)

    records: List[ImageRecord] = []
    for row in records_raw:
        idx, abs_path, rel_path, file_size, width, height, fhash, pxhash, dh, ph, fallback_feature, err = row
        if err is not None:
            records.append(
                ImageRecord(
                    idx=idx,
                    path=abs_path,
                    rel_path=rel_path,
                    file_size=file_size,
                    width=width,
                    height=height,
                    file_hash=fhash,
                    pixel_hash=pxhash,
                    dhash=dh,
                    phash=ph,
                    feature=fallback_feature,
                    embedding_backend="error",
                    error=err,
                )
            )
        else:
            feat = pos2clip.get(idx, fallback_feature)
            backend = backend_name if idx in pos2clip else "lightweight_fallback"
            records.append(
                ImageRecord(
                    idx=idx,
                    path=abs_path,
                    rel_path=rel_path,
                    file_size=file_size,
                    width=width,
                    height=height,
                    file_hash=fhash,
                    pixel_hash=pxhash,
                    dhash=dh,
                    phash=ph,
                    feature=feat,
                    embedding_backend=backend,
                    error=None,
                )
            )
    return records, errors


def build_knn_pairs(
    embeddings: np.ndarray,
    top_k: int,
    status_cb: StatusCallback = None,
    progress_cb: ProgressCallback = None,
) -> Iterable[Tuple[int, int, float]]:
    n, d = embeddings.shape
    top_k = min(top_k, max(1, n - 1))
    if n <= 1:
        return []

    emit_status(status_cb, "正在建立近邻检索索引…")
    if faiss is not None:
        index = faiss.IndexFlatIP(d)
        index.add(embeddings.astype(np.float32))
        sims, ids = index.search(embeddings.astype(np.float32), top_k + 1)
        emit_progress(progress_cb, n, n, "正在计算相似近邻")
        pairs = []
        for i in range(n):
            for sim, j in zip(sims[i], ids[i]):
                if j < 0 or j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                pairs.append((a, b, float(sim)))
        return pairs

    pairs = []
    batch = 256
    for start in range(0, n, batch):
        end = min(n, start + batch)
        sims = embeddings[start:end] @ embeddings.T
        for local_i in range(end - start):
            i = start + local_i
            row = sims[local_i]
            if top_k >= len(row) - 1:
                cand_ids = np.argsort(-row)
            else:
                cand_ids = np.argpartition(-row, top_k + 1)[: top_k + 1]
                cand_ids = cand_ids[np.argsort(-row[cand_ids])]
            for j in cand_ids:
                if j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)
                pairs.append((a, b, float(row[j])))
        emit_progress(progress_cb, end, n, "正在计算相似近邻")
    return pairs


def combined_similarity(emb_sim: float, phash_sim: float, dhash_sim: float, backend: str) -> float:
    if backend.startswith("open_clip:"):
        return clamp01(0.72 * emb_sim + 0.18 * phash_sim + 0.10 * dhash_sim)
    return clamp01(0.20 * emb_sim + 0.45 * phash_sim + 0.35 * dhash_sim)


def match_type_from_flags(
    same_file_hash: bool,
    same_pixel_hash: bool,
    score: float,
    phash_sim: float,
    dhash_sim: float,
    emb_sim: float,
) -> str:
    if same_file_hash:
        return "exact_file"
    if same_pixel_hash:
        return "exact_pixels"
    if score >= 0.97 or (phash_sim >= 0.98 and dhash_sim >= 0.95):
        return "near_duplicate"
    if score >= 0.93:
        return "strong_match"
    return "possible_match"


def make_similarity_row(a: ImageRecord, b: ImageRecord, reason: str = "direct_compare") -> dict:
    same_file_hash = bool(a.file_hash and a.file_hash == b.file_hash)
    same_pixel_hash = bool(a.pixel_hash and a.pixel_hash == b.pixel_hash)
    backend = a.embedding_backend if a.embedding_backend == b.embedding_backend else a.embedding_backend
    emb_sim = float(np.dot(a.feature, b.feature))
    phash_sim = bit_similarity64(a.phash, b.phash)
    dhash_sim = bit_similarity64(a.dhash, b.dhash)
    score = 1.0 if (same_file_hash or same_pixel_hash) else combined_similarity(emb_sim, phash_sim, dhash_sim, backend)

    return {
        "path_1": a.path,
        "path_2": b.path,
        "rel_path_1": a.rel_path,
        "rel_path_2": b.rel_path,
        "match_type": match_type_from_flags(same_file_hash, same_pixel_hash, score, phash_sim, dhash_sim, emb_sim),
        "same_file_hash": same_file_hash,
        "same_pixel_hash": same_pixel_hash,
        "file_size_1": a.file_size,
        "file_size_2": b.file_size,
        "resolution_1": f"{a.width}x{a.height}",
        "resolution_2": f"{b.width}x{b.height}",
        "embedding_backend": backend,
        "embedding_similarity": round(float(emb_sim), 6),
        "phash_similarity": round(float(phash_sim), 6),
        "dhash_similarity": round(float(dhash_sim), 6),
        "similarity_score": round(float(score), 6),
        "reason": reason,
    }


def build_results(
    records: List[ImageRecord],
    min_sim: float,
    max_sim: float,
    top_k: int,
    status_cb: StatusCallback = None,
    progress_cb: ProgressCallback = None,
) -> List[dict]:
    valid = [r for r in records if r.error is None]
    if not valid:
        return []

    emit_status(status_cb, "正在汇总相似图片候选…")
    idx_map = {r.idx: r for r in valid}
    pairs: Dict[Tuple[int, int], dict] = {}

    def upsert_pair(a: int, b: int, emb_sim: float, phash_sim: float, dhash_sim: float, reason: str):
        if a == b:
            return
        if a > b:
            a, b = b, a
        ra = idx_map[a]
        rb = idx_map[b]
        same_file_hash = ra.file_hash and (ra.file_hash == rb.file_hash)
        same_pixel_hash = ra.pixel_hash and (ra.pixel_hash == rb.pixel_hash)
        backend = ra.embedding_backend if ra.embedding_backend == rb.embedding_backend else "mixed"
        score = 1.0 if (same_file_hash or same_pixel_hash) else combined_similarity(emb_sim, phash_sim, dhash_sim, backend)
        if not (min_sim <= score <= max_sim):
            return

        row = {
            "path_1": ra.path,
            "path_2": rb.path,
            "rel_path_1": ra.rel_path,
            "rel_path_2": rb.rel_path,
            "match_type": match_type_from_flags(same_file_hash, same_pixel_hash, score, phash_sim, dhash_sim, emb_sim),
            "same_file_hash": same_file_hash,
            "same_pixel_hash": same_pixel_hash,
            "file_size_1": ra.file_size,
            "file_size_2": rb.file_size,
            "resolution_1": f"{ra.width}x{ra.height}",
            "resolution_2": f"{rb.width}x{rb.height}",
            "embedding_backend": backend,
            "embedding_similarity": round(float(emb_sim), 6),
            "phash_similarity": round(float(phash_sim), 6),
            "dhash_similarity": round(float(dhash_sim), 6),
            "similarity_score": round(float(score), 6),
            "reason": reason,
        }

        old = pairs.get((a, b))
        if old is None or row["similarity_score"] > old["similarity_score"]:
            pairs[(a, b)] = row

    file_groups: Dict[str, List[int]] = {}
    pixel_groups: Dict[str, List[int]] = {}
    for r in valid:
        file_groups.setdefault(r.file_hash, []).append(r.idx)
        pixel_groups.setdefault(r.pixel_hash, []).append(r.idx)

    for h, group in file_groups.items():
        if h and len(group) > 1:
            for a, b in all_pairs(group):
                upsert_pair(a, b, emb_sim=1.0, phash_sim=1.0, dhash_sim=1.0, reason="same_file_hash")

    for h, group in pixel_groups.items():
        if h and len(group) > 1:
            for a, b in all_pairs(group):
                ra = idx_map[a]
                rb = idx_map[b]
                emb = float(np.dot(ra.feature, rb.feature))
                upsert_pair(a, b, emb_sim=emb, phash_sim=1.0, dhash_sim=1.0, reason="same_pixel_hash")

    embeddings = np.vstack([r.feature for r in valid]).astype(np.float32)
    knn_pairs = build_knn_pairs(embeddings, top_k=top_k, status_cb=status_cb, progress_cb=progress_cb)

    pos_to_idx = [r.idx for r in valid]
    count = 0
    total = max(1, len(pos_to_idx))
    for pos_a, pos_b, emb_sim in knn_pairs:
        idx_a = pos_to_idx[pos_a]
        idx_b = pos_to_idx[pos_b]
        ra = idx_map[idx_a]
        rb = idx_map[idx_b]
        phash_sim = bit_similarity64(ra.phash, rb.phash)
        dhash_sim = bit_similarity64(ra.dhash, rb.dhash)

        if max(emb_sim, phash_sim, dhash_sim) < max(0.0, min_sim - 0.08):
            continue
        if not ra.embedding_backend.startswith("open_clip:") and phash_sim < max(0.0, min_sim - 0.12):
            continue
        upsert_pair(idx_a, idx_b, emb_sim=emb_sim, phash_sim=phash_sim, dhash_sim=dhash_sim, reason="knn_candidate")
        count += 1
        if count % max(1, total // 10) == 0:
            emit_progress(progress_cb, min(count, total), total, "正在筛选相似图片")

    rows = list(pairs.values())
    rows.sort(key=lambda x: (-x["similarity_score"], x["path_1"], x["path_2"]))
    emit_progress(progress_cb, total, total, "正在筛选相似图片")
    return rows


def build_reference_matches(
    reference_record: ImageRecord,
    candidate_records: List[ImageRecord],
    min_sim: float,
    max_sim: float,
    progress_cb: ProgressCallback = None,
) -> List[dict]:
    rows = []
    total = len(candidate_records)
    for idx, candidate in enumerate(candidate_records, start=1):
        if candidate.error is not None:
            continue
        row = make_similarity_row(reference_record, candidate)
        if min_sim <= row["similarity_score"] <= max_sim:
            rows.append(row)
        if idx == total or idx % max(1, total // 50 or 1) == 0:
            emit_progress(progress_cb, idx, total, "正在匹配参考图")
    rows.sort(key=lambda x: (-x["similarity_score"], x["path_2"]))
    return rows


def autosize_columns(ws, extra: int = 2, max_width: int = 60) -> None:
    widths = {}
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            val = str(cell.value)
            widths[cell.column] = max(widths.get(cell.column, 0), min(max_width, len(val) + extra))
    for col_idx, width in widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width


def match_type_label(match_type: str) -> str:
    return MATCH_TYPE_LABELS.get(match_type, match_type)


def explain_match_reason(row: dict) -> str:
    if row.get("same_file_hash"):
        return "文件内容完全一致。"
    if row.get("same_pixel_hash"):
        return "像素内容完全一致。"
    match_type = row.get("match_type", "")
    if match_type == "near_duplicate":
        return "综合特征、感知哈希和差异哈希都非常接近。"
    if match_type == "strong_match":
        return "综合特征接近，视觉内容大体相同。"
    return "综合特征接近，建议人工复核。"


def display_name_for_path(path: str) -> str:
    return Path(path).name or path


def display_size(num_bytes: int) -> str:
    return human_size(int(num_bytes or 0))


def describe_image_side(row: dict, prefix: str) -> str:
    resolution = row.get(f"resolution_{prefix}", "") or "-"
    size = display_size(row.get(f"file_size_{prefix}", 0))
    return f"{resolution} | {size}"


def export_match_rows(rows: List[dict]) -> List[dict]:
    exported = []
    for index, row in enumerate(rows, start=1):
        deleted_side = row.get("deleted_side", "") or ""
        exported.append(
            {
                "序号": index,
                "相似度": row["similarity_score"],
                "匹配判断": match_type_label(row.get("match_type", "")),
                "匹配原因": explain_match_reason(row),
                "图片 A": row.get("name_1", display_name_for_path(row["path_1"])),
                "图片 B": row.get("name_2", display_name_for_path(row["path_2"])),
                "A 信息": describe_image_side(row, "1"),
                "B 信息": describe_image_side(row, "2"),
                "A 路径": row["path_1"],
                "B 路径": row["path_2"],
                "A 相对路径": row.get("rel_path_1", ""),
                "B 相对路径": row.get("rel_path_2", ""),
                "状态": f"已删除 {deleted_side} 侧" if deleted_side else "",
            }
        )
    return exported


def write_unified_txt_report(
    *,
    output_txt: Path,
    title: str,
    summary_rows: List[Tuple[str, str]],
    rows: List[dict],
    left_label: str,
    right_label: str,
    errors_sections: List[Tuple[str, List[Tuple[str, str]]]],
) -> None:
    lines: List[str] = [title, "=" * 72]
    lines.extend(f"{label}: {value}" for label, value in summary_rows)
    lines.append("")

    if rows:
        lines.append("匹配结果")
        lines.append("-" * 72)
        for exported in export_match_rows(rows):
            lines.append(f"[{exported['序号']:03d}] {exported['匹配判断']} | 相似度 {exported['相似度']:.4f}")
            lines.append(f"  原因: {exported['匹配原因']}")
            lines.append(f"  {left_label}: {exported['图片 A']} | {exported['A 信息']}")
            lines.append(f"    路径: {exported['A 路径']}")
            if exported["A 相对路径"]:
                lines.append(f"    相对路径: {exported['A 相对路径']}")
            lines.append(f"  {right_label}: {exported['图片 B']} | {exported['B 信息']}")
            lines.append(f"    路径: {exported['B 路径']}")
            if exported["B 相对路径"]:
                lines.append(f"    相对路径: {exported['B 相对路径']}")
            if exported["状态"]:
                lines.append(f"  状态: {exported['状态']}")
            lines.append("")
    else:
        lines.append("没有找到符合当前阈值的结果。")
        lines.append("")

    has_errors = any(items for _, items in errors_sections)
    if has_errors:
        lines.append("异常文件")
        lines.append("-" * 72)
        for section_title, items in errors_sections:
            for path, err in items:
                lines.append(f"{section_title}: {path}")
                lines.append(f"  错误: {err}")
        lines.append("")

    output_txt.write_text("\n".join(lines), encoding="utf-8")


def write_excel(
    output_xlsx: str,
    root_dir: str,
    rows: List[dict],
    records: List[ImageRecord],
    errors: List[Tuple[str, str]],
    min_sim: float,
    max_sim: float,
    top_k: int,
) -> None:
    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = "摘要"
    ws_matches = wb.create_sheet("匹配结果")
    ws_errors = wb.create_sheet("异常文件")

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin_gray = Side(style="thin", color="B7B7B7")
    border = Border(left=thin_gray, right=thin_gray, top=thin_gray, bottom=thin_gray)

    ws_summary["A1"] = "图片相似结果"
    ws_summary["A1"].font = Font(size=15, bold=True)

    summary_rows = [
        ("扫描目录", root_dir),
        ("扫描图片数", len(records)),
        ("有效图片数", sum(1 for r in records if r.error is None)),
        ("异常文件数", sum(1 for r in records if r.error is not None)),
        ("最低相似度", f"{min_sim:.2f}"),
        ("最高相似度", f"{max_sim:.2f}"),
        ("每张图片保留", top_k),
        ("匹配结果数", len(rows)),
    ]
    ws_summary["A3"] = "项目"
    ws_summary["B3"] = "值"
    for c in ("A3", "B3"):
        ws_summary[c].fill = header_fill
        ws_summary[c].font = header_font
        ws_summary[c].alignment = Alignment(horizontal="center")
    for i, (k, v) in enumerate(summary_rows, start=4):
        ws_summary[f"A{i}"] = k
        ws_summary[f"B{i}"] = v
    ws_summary.freeze_panes = "A4"
    autosize_columns(ws_summary)

    exported_rows = export_match_rows(rows)
    headers = [
        "序号", "相似度", "匹配判断", "匹配原因",
        "图片 A", "A 信息", "A 路径",
        "图片 B", "B 信息", "B 路径",
        "状态",
    ]
    for col, h in enumerate(headers, start=1):
        cell = ws_matches.cell(row=1, column=col, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row_idx, row in enumerate(exported_rows, start=2):
        for col_idx, h in enumerate(headers, start=1):
            val = row[h]
            cell = ws_matches.cell(row=row_idx, column=col_idx, value=val)
            cell.border = border
            if h in {"相似度"}:
                cell.number_format = "0.0000"
            if h in {"A 路径", "B 路径"}:
                p = Path(str(val))
                if p.exists():
                    cell.hyperlink = p.as_uri()
                    cell.style = "Hyperlink"

    ws_matches.freeze_panes = "A2"
    ws_matches.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(2, len(rows) + 1)}"
    autosize_columns(ws_matches, extra=3, max_width=90)

    ws_errors["A1"] = "路径"
    ws_errors["B1"] = "错误"
    for c in ("A1", "B1"):
        ws_errors[c].fill = header_fill
        ws_errors[c].font = header_font
        ws_errors[c].alignment = Alignment(horizontal="center")
    for i, (p, err) in enumerate(errors, start=2):
        ws_errors[f"A{i}"] = p
        ws_errors[f"B{i}"] = err
    ws_errors.freeze_panes = "A2"
    ws_errors.auto_filter.ref = f"A1:B{max(2, len(errors) + 1)}"
    autosize_columns(ws_errors, extra=3, max_width=100)

    wb.save(output_xlsx)


def write_reference_txt_report(
    output_txt: Path,
    reference_path: Path,
    search_dir: Path,
    rows: List[dict],
    scanned_count: int,
    valid_count: int,
    errors: List[Tuple[str, str]],
    min_sim: float,
    max_sim: float,
    recursive: bool,
) -> None:
    write_unified_txt_report(
        output_txt=output_txt,
        title="参考图检索报告",
        summary_rows=[
            ("生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("参考图片", str(reference_path)),
            ("搜索目录", str(search_dir)),
            ("递归扫描", "是" if recursive else "否"),
            ("扫描图片数", str(scanned_count)),
            ("有效图片数", str(valid_count)),
            ("异常文件数", str(len(errors))),
            ("相似度区间", f"{min_sim:.2f} ~ {max_sim:.2f}"),
            ("匹配结果数", str(len(rows))),
        ],
        rows=rows,
        left_label="参考图",
        right_label="命中图",
        errors_sections=[("异常文件", errors)],
    )


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(0, num_bytes))
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{num_bytes} B"


def open_file_location(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer", "/select,", str(p)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p.parent)])


def move_to_trash(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if _send2trash is not None:
        _send2trash(str(p))
        return "已移到回收站"
    p.unlink()
    return "已直接删除"


def build_cross_directory_matches(
    records_a: List[ImageRecord],
    records_b: List[ImageRecord],
    min_sim: float,
    max_sim: float,
    top_k_per_a: int,
    status_cb: StatusCallback = None,
    progress_cb: ProgressCallback = None,
) -> List[dict]:
    valid_a = [r for r in records_a if r.error is None]
    valid_b = [r for r in records_b if r.error is None]
    if not valid_a or not valid_b:
        return []

    emit_status(status_cb, "正在进行 A/B 交叉相似检索…")
    pairs: Dict[Tuple[str, str], dict] = {}

    def upsert(ra: ImageRecord, rb: ImageRecord, reason: str, force_exact: bool = False) -> None:
        row = make_similarity_row(ra, rb, reason=reason)
        if force_exact:
            row["similarity_score"] = 1.0
        if not (min_sim <= row["similarity_score"] <= max_sim):
            return
        row["folder_1"] = str(Path(ra.path).parent)
        row["folder_2"] = str(Path(rb.path).parent)
        row["name_1"] = Path(ra.path).name
        row["name_2"] = Path(rb.path).name
        row["deleted_side"] = row.get("deleted_side", "")
        key = (row["path_1"], row["path_2"])
        old = pairs.get(key)
        if old is None or row["similarity_score"] > old["similarity_score"]:
            pairs[key] = row

    b_by_file: Dict[str, List[ImageRecord]] = {}
    b_by_pixel: Dict[str, List[ImageRecord]] = {}
    for rb in valid_b:
        if rb.file_hash:
            b_by_file.setdefault(rb.file_hash, []).append(rb)
        if rb.pixel_hash:
            b_by_pixel.setdefault(rb.pixel_hash, []).append(rb)

    for idx, ra in enumerate(valid_a, start=1):
        for rb in b_by_file.get(ra.file_hash, []):
            upsert(ra, rb, reason="cross_same_file_hash", force_exact=True)
        for rb in b_by_pixel.get(ra.pixel_hash, []):
            upsert(ra, rb, reason="cross_same_pixel_hash")
        if idx == len(valid_a) or idx % max(1, len(valid_a) // 20 or 1) == 0:
            emit_progress(progress_cb, idx, len(valid_a), "正在建立精确候选")

    emb_a = np.vstack([r.feature for r in valid_a]).astype(np.float32)
    emb_b = np.vstack([r.feature for r in valid_b]).astype(np.float32)
    top_k_per_a = max(1, min(top_k_per_a, len(valid_b)))

    if faiss is not None:
        index = faiss.IndexFlatIP(emb_b.shape[1])
        index.add(emb_b)
        sims, ids = index.search(emb_a, top_k_per_a)
        for i, ra in enumerate(valid_a, start=1):
            for sim, j in zip(sims[i - 1], ids[i - 1]):
                if j < 0:
                    continue
                rb = valid_b[int(j)]
                phash_sim = bit_similarity64(ra.phash, rb.phash)
                dhash_sim = bit_similarity64(ra.dhash, rb.dhash)
                if max(float(sim), phash_sim, dhash_sim) < max(0.0, min_sim - 0.08):
                    continue
                upsert(ra, rb, reason="cross_knn_candidate")
            if i == len(valid_a) or i % max(1, len(valid_a) // 20 or 1) == 0:
                emit_progress(progress_cb, i, len(valid_a), "正在搜索 A/B 相似对")
    else:
        batch = 128
        done = 0
        for start in range(0, len(valid_a), batch):
            end = min(len(valid_a), start + batch)
            sims = emb_a[start:end] @ emb_b.T
            for local_i in range(end - start):
                row = sims[local_i]
                if top_k_per_a >= len(valid_b):
                    cand_ids = np.argsort(-row)
                else:
                    cand_ids = np.argpartition(-row, top_k_per_a - 1)[:top_k_per_a]
                    cand_ids = cand_ids[np.argsort(-row[cand_ids])]
                ra = valid_a[start + local_i]
                for j in cand_ids:
                    rb = valid_b[int(j)]
                    phash_sim = bit_similarity64(ra.phash, rb.phash)
                    dhash_sim = bit_similarity64(ra.dhash, rb.dhash)
                    if max(float(row[j]), phash_sim, dhash_sim) < max(0.0, min_sim - 0.08):
                        continue
                    upsert(ra, rb, reason="cross_dense_candidate")
                done += 1
                emit_progress(progress_cb, done, len(valid_a), "正在搜索 A/B 相似对")

    rows = list(pairs.values())
    rows.sort(key=lambda x: (-x["similarity_score"], x["path_1"], x["path_2"]))
    return rows


def write_cross_compare_txt_report(
    output_txt: Path,
    dir_a: Path,
    dir_b: Path,
    rows: List[dict],
    errors_a: List[Tuple[str, str]],
    errors_b: List[Tuple[str, str]],
    min_sim: float,
    max_sim: float,
) -> None:
    write_unified_txt_report(
        output_txt=output_txt,
        title="双目录复核报告",
        summary_rows=[
            ("生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("目录 A", str(dir_a)),
            ("目录 B", str(dir_b)),
            ("相似度区间", f"{min_sim:.2f} ~ {max_sim:.2f}"),
            ("匹配结果数", str(len(rows))),
        ],
        rows=rows,
        left_label="目录 A",
        right_label="目录 B",
        errors_sections=[("目录 A 异常", errors_a), ("目录 B 异常", errors_b)],
    )


def normalize_common_args(args: argparse.Namespace) -> argparse.Namespace:
    device = guess_device(getattr(args, "device", "auto"))
    args.min_sim = 0.90 if getattr(args, "min_sim", None) is None else args.min_sim
    args.max_sim = 1.00 if getattr(args, "max_sim", None) is None else args.max_sim
    args.workers = getattr(args, "workers", None) or recommended_workers()
    args.batch_size = getattr(args, "batch_size", None) or recommended_batch_size(device)
    args.device = getattr(args, "device", "auto")
    args.clip_model = getattr(args, "clip_model", "ViT-B-32")
    args.clip_pretrained = getattr(args, "clip_pretrained", "laion2b_s34b_b79k")
    args.no_openclip = getattr(args, "no_openclip", False)
    args.non_recursive = getattr(args, "non_recursive", False)
    args.extensions = getattr(args, "extensions", "jpg,jpeg,png,bmp,webp,tif,tiff,gif")
    return args


def validate_similarity_range(min_sim: float, max_sim: float) -> None:
    if not (0.0 <= min_sim <= 1.0 and 0.0 <= max_sim <= 1.0 and min_sim <= max_sim):
        raise ValueError("相似度范围必须满足 0 ≤ 最低相似度 ≤ 最高相似度 ≤ 1")


def run_scan_job(args: argparse.Namespace, status_cb: StatusCallback = None, progress_cb: ProgressCallback = None) -> dict:
    total_start = time.perf_counter()
    args = normalize_common_args(args)

    input_dir = absolute_path(Path(clean_input_path(args.input_dir)))
    output_xlsx = absolute_path(Path(clean_input_path(args.output_xlsx))) if args.output_xlsx else make_default_output_path(input_dir)
    if output_xlsx.suffix.lower() != ".xlsx":
        output_xlsx = output_xlsx.with_suffix(".xlsx")
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"输入目录不存在或不是目录: {input_dir}")

    min_sim = float(args.min_sim)
    max_sim = float(args.max_sim)
    validate_similarity_range(min_sim, max_sim)

    device = guess_device(args.device)
    workers = args.workers
    batch_size = args.batch_size
    top_k = getattr(args, "top_k", None) or 20
    use_openclip = not args.no_openclip
    extensions = {("." + x.strip().lower().lstrip(".")) for x in args.extensions.split(",") if x.strip()}

    emit_status(status_cb, hardware_summary(device))
    files = list_images(input_dir, recursive=not args.non_recursive, extensions=extensions)
    if not files:
        raise ValueError("没有找到图片文件。")
    emit_status(status_cb, f"已找到 {len(files)} 张图片，开始分析。")

    records, errors = build_records(
        files=files,
        root=input_dir,
        workers=workers,
        use_openclip=use_openclip,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        batch_size=batch_size,
        device=device,
        clip_mirror=getattr(args, 'clip_mirror', 'auto'),
        clip_endpoint=getattr(args, 'clip_endpoint', ''),
        model_cache_dir=getattr(args, 'model_cache_dir', None),
        status_cb=status_cb,
        progress_cb=progress_cb,
    )
    rows = build_results(records=records, min_sim=min_sim, max_sim=max_sim, top_k=top_k, status_cb=status_cb, progress_cb=progress_cb)

    emit_status(status_cb, "正在写入 Excel 报告…")
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    write_excel(
        output_xlsx=str(output_xlsx),
        root_dir=str(input_dir),
        rows=rows,
        records=records,
        errors=errors,
        min_sim=min_sim,
        max_sim=max_sim,
        top_k=top_k,
    )
    total_elapsed = time.perf_counter() - total_start
    valid_count = sum(1 for r in records if r.error is None)
    emit_progress(progress_cb, 1, 1, "已完成")
    return {
        "output_path": str(output_xlsx),
        "valid_count": valid_count,
        "row_count": len(rows),
        "error_count": len(errors),
        "elapsed": total_elapsed,
    }


def run_reference_job(args: argparse.Namespace, status_cb: StatusCallback = None, progress_cb: ProgressCallback = None) -> dict:
    total_start = time.perf_counter()
    args = normalize_common_args(args)

    reference_image = absolute_path(Path(clean_input_path(args.reference_image)))
    search_dir = absolute_path(Path(clean_input_path(args.search_dir)))
    output_txt = absolute_path(Path(clean_input_path(args.output_txt))) if args.output_txt else make_default_txt_report_path(search_dir, reference_image)
    if output_txt.suffix.lower() != ".txt":
        output_txt = output_txt.with_suffix(".txt")
    if not reference_image.exists() or not reference_image.is_file():
        raise ValueError(f"参考图片不存在或不是文件: {reference_image}")
    if reference_image.suffix.lower() not in VALID_EXTENSIONS:
        raise ValueError(f"参考图片格式不支持: {reference_image.suffix}")
    if not search_dir.exists() or not search_dir.is_dir():
        raise ValueError(f"搜索目录不存在或不是目录: {search_dir}")

    min_sim = float(args.min_sim)
    max_sim = float(args.max_sim)
    validate_similarity_range(min_sim, max_sim)

    device = guess_device(args.device)
    workers = args.workers
    batch_size = args.batch_size
    use_openclip = not args.no_openclip
    extensions = {("." + x.strip().lower().lstrip(".")) for x in args.extensions.split(",") if x.strip()}

    emit_status(status_cb, hardware_summary(device))
    emit_status(status_cb, f"参考图: {reference_image.name}")
    search_files = list_images(search_dir, recursive=not args.non_recursive, extensions=extensions)
    reference_norm = normalize_path(reference_image)
    search_files = [p for p in search_files if normalize_path(p) != reference_norm]
    if not search_files:
        raise ValueError("目标文件夹中没有找到图片文件。")
    emit_status(status_cb, f"已找到 {len(search_files)} 张候选图片，开始分析。")

    all_files = [reference_image] + search_files
    records, errors = build_records(
        files=all_files,
        root=search_dir,
        workers=workers,
        use_openclip=use_openclip,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        batch_size=batch_size,
        device=device,
        clip_mirror=getattr(args, 'clip_mirror', 'auto'),
        clip_endpoint=getattr(args, 'clip_endpoint', ''),
        model_cache_dir=getattr(args, 'model_cache_dir', None),
        status_cb=status_cb,
        progress_cb=progress_cb,
    )
    reference_record = records[0]
    if reference_record.error is not None:
        raise ValueError(f"参考图片无法读取：{reference_record.error}")

    candidate_records = records[1:]
    valid_count = sum(1 for r in candidate_records if r.error is None)
    emit_status(status_cb, "正在与参考图进行相似匹配…")
    rows = build_reference_matches(reference_record, candidate_records, min_sim=min_sim, max_sim=max_sim, progress_cb=progress_cb)

    emit_status(status_cb, "正在写入 TXT 报告…")
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    write_reference_txt_report(
        output_txt=output_txt,
        reference_path=reference_image,
        search_dir=search_dir,
        rows=rows,
        scanned_count=len(search_files),
        valid_count=valid_count,
        errors=errors,
        min_sim=min_sim,
        max_sim=max_sim,
        recursive=not args.non_recursive,
    )
    total_elapsed = time.perf_counter() - total_start
    emit_progress(progress_cb, 1, 1, "已完成")
    return {
        "output_path": str(output_txt),
        "valid_count": valid_count,
        "row_count": len(rows),
        "error_count": len(errors),
        "elapsed": total_elapsed,
    }


def run_ab_compare_job(args: argparse.Namespace, status_cb: StatusCallback = None, progress_cb: ProgressCallback = None) -> dict:
    total_start = time.perf_counter()
    args = normalize_common_args(args)

    dir_a = absolute_path(Path(clean_input_path(args.dir_a)))
    dir_b = absolute_path(Path(clean_input_path(args.dir_b)))
    if not dir_a.exists() or not dir_a.is_dir():
        raise ValueError(f"目录 A 不存在或不是目录: {dir_a}")
    if not dir_b.exists() or not dir_b.is_dir():
        raise ValueError(f"目录 B 不存在或不是目录: {dir_b}")
    if dir_a == dir_b:
        raise ValueError("目录 A 和目录 B 不能是同一个目录。")

    min_sim = float(args.min_sim)
    max_sim = float(args.max_sim)
    validate_similarity_range(min_sim, max_sim)

    device = guess_device(args.device)
    workers = args.workers
    batch_size = args.batch_size
    use_openclip = not args.no_openclip
    top_k_per_a = getattr(args, "top_k_per_a", None) or 5
    extensions = {("." + x.strip().lower().lstrip(".")) for x in args.extensions.split(",") if x.strip()}

    emit_status(status_cb, hardware_summary(device))
    files_a = list_images(dir_a, recursive=not args.non_recursive, extensions=extensions)
    files_b = list_images(dir_b, recursive=not args.non_recursive, extensions=extensions)
    if not files_a:
        raise ValueError("目录 A 中没有找到图片文件。")
    if not files_b:
        raise ValueError("目录 B 中没有找到图片文件。")
    emit_status(status_cb, f"已找到 A={len(files_a)} 张，B={len(files_b)} 张。")

    emit_status(status_cb, "正在分析目录 A…")
    records_a, errors_a = build_records(
        files=files_a,
        root=dir_a,
        workers=workers,
        use_openclip=use_openclip,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        batch_size=batch_size,
        device=device,
        clip_mirror=getattr(args, 'clip_mirror', 'auto'),
        clip_endpoint=getattr(args, 'clip_endpoint', ''),
        model_cache_dir=getattr(args, 'model_cache_dir', None),
        status_cb=status_cb,
        progress_cb=progress_cb,
    )
    emit_status(status_cb, "正在分析目录 B…")
    records_b, errors_b = build_records(
        files=files_b,
        root=dir_b,
        workers=workers,
        use_openclip=use_openclip,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        batch_size=batch_size,
        device=device,
        clip_mirror=getattr(args, 'clip_mirror', 'auto'),
        clip_endpoint=getattr(args, 'clip_endpoint', ''),
        model_cache_dir=getattr(args, 'model_cache_dir', None),
        status_cb=status_cb,
        progress_cb=progress_cb,
    )
    rows = build_cross_directory_matches(
        records_a=records_a,
        records_b=records_b,
        min_sim=min_sim,
        max_sim=max_sim,
        top_k_per_a=top_k_per_a,
        status_cb=status_cb,
        progress_cb=progress_cb,
    )
    total_elapsed = time.perf_counter() - total_start
    emit_progress(progress_cb, 1, 1, "已完成")
    return {
        "rows": rows,
        "records_a": records_a,
        "records_b": records_b,
        "errors_a": errors_a,
        "errors_b": errors_b,
        "dir_a": str(dir_a),
        "dir_b": str(dir_b),
        "elapsed": total_elapsed,
        "min_sim": min_sim,
        "max_sim": max_sim,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图片相似度桌面工作台")
    parser.add_argument("--mode", default="gui", help="兼容旧参数；此版本固定启动图形界面")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--clip-mirror", default="auto", help="模型下载源：auto / cn / official")
    parser.add_argument("--clip-endpoint", default="", help="自定义 Hugging Face 端点，例如 https://hf-mirror.com")
    parser.add_argument("--model-cache-dir", default=str(default_model_cache_dir()), help="模型缓存目录；删除此目录即可清理自动下载的模型")
    parser.add_argument("--no-openclip", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--extensions", default="jpg,jpeg,png,bmp,webp,tif,tiff,gif")
    parser.add_argument("--tab", choices=["compare", "reference", "scan"], default="compare")
    return parser.parse_args()


def launch_gui_mode(base_args: argparse.Namespace) -> None:
    from desktop_ui import launch_app

    launch_app()
    return

    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import ImageTk
    except Exception as e:
        raise SystemExit(f"图形界面依赖不可用：{e}")
    try:
        import ttkbootstrap as ttkb  # type: ignore
    except Exception:
        ttkb = None  # type: ignore

    class ZoomPreviewPane:
        def __init__(self, parent, side: str, bg: str, border: str, text_color: str, muted: str, accent: str, view_change_callback) -> None:
            self.side = side
            self.bg = bg
            self.border = border
            self.text_color = text_color
            self.muted = muted
            self.accent = accent
            self.view_change_callback = view_change_callback
            self.frame = tk.Frame(parent, bg=bg, bd=0, highlightthickness=1, highlightbackground=border)
            self.canvas = tk.Canvas(
                self.frame,
                bg=bg,
                bd=0,
                highlightthickness=0,
                relief='flat',
                cursor='fleur',
            )
            self.canvas.pack(fill='both', expand=True)
            self.placeholder = self.canvas.create_text(
                0, 0, text=f'{side} 侧图片预览', fill=muted, font=('Microsoft YaHei UI', 18, 'bold')
            )
            self._overlay_zoom = tk.Label(
                self.frame,
                text='适应窗口',
                bg='#ffffff',
                fg=text_color,
                font=('Microsoft YaHei UI', 11, 'bold'),
                padx=10,
                pady=4,
                bd=1,
                relief='solid',
                highlightthickness=1,
                highlightbackground=border,
            )
            self._overlay_zoom.place(relx=1.0, x=-12, y=12, anchor='ne')
            self._image_id = None
            self._photo = None
            self.source_image = None
            self.zoom = 1.0
            self.fit_zoom = 1.0
            self.view_left = 0.0
            self.view_top = 0.0
            self._drag_start = None
            self._after_id = None
            self.canvas.bind('<Configure>', self._on_configure)
            self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)
            self.canvas.bind('<Button-4>', self._on_mouse_wheel)
            self.canvas.bind('<Button-5>', self._on_mouse_wheel)
            self.canvas.bind('<ButtonPress-1>', self._on_drag_start)
            self.canvas.bind('<B1-Motion>', self._on_drag_motion)
            self.canvas.bind('<Double-Button-1>', self._on_double_click)

        def destroy(self) -> None:
            if self._after_id is not None:
                try:
                    self.canvas.after_cancel(self._after_id)
                except Exception:
                    pass
                self._after_id = None

        def _on_configure(self, _event=None) -> None:
            self.schedule_render()

        def schedule_render(self, delay: int = 50) -> None:
            if self._after_id is not None:
                try:
                    self.canvas.after_cancel(self._after_id)
                except Exception:
                    pass
            self._after_id = self.canvas.after(delay, self.render)

        def clear(self, message: str) -> None:
            self.source_image = None
            self._photo = None
            self.zoom = 1.0
            self.fit_zoom = 1.0
            self.view_left = 0.0
            self.view_top = 0.0
            self.canvas.delete('all')
            cw = max(10, self.canvas.winfo_width())
            ch = max(10, self.canvas.winfo_height())
            self.placeholder = self.canvas.create_text(cw // 2, ch // 2, text=message, fill=self.muted, font=('Microsoft YaHei UI', 18, 'bold'))
            self._overlay_zoom.configure(text='等待结果')

        def set_image(self, image, reset_view: bool = True) -> None:
            self.source_image = image
            if reset_view:
                self.fit_to_view(render=False)
            self.schedule_render()

        def _canvas_size(self) -> tuple[int, int]:
            return max(120, self.canvas.winfo_width()), max(120, self.canvas.winfo_height())

        def _calc_fit_zoom(self) -> float:
            if self.source_image is None:
                return 1.0
            cw, ch = self._canvas_size()
            iw, ih = self.source_image.size
            if iw <= 0 or ih <= 0:
                return 1.0
            return max(0.05, min(cw / iw, ch / ih, 8.0))

        def fit_to_view(self, render: bool = True) -> None:
            if self.source_image is None:
                return
            self.fit_zoom = self._calc_fit_zoom()
            self.zoom = self.fit_zoom
            iw, ih = self.source_image.size
            vw, vh = self._visible_src_size(self.zoom)
            self.view_left = max(0.0, (iw - vw) / 2.0)
            self.view_top = max(0.0, (ih - vh) / 2.0)
            if render:
                self.render()

        def actual_size(self) -> None:
            if self.source_image is None:
                return
            self.set_zoom_and_center(1.0, *self.get_center_ratios())

        def _visible_src_size(self, zoom: float | None = None) -> tuple[float, float]:
            if self.source_image is None:
                return 1.0, 1.0
            cw, ch = self._canvas_size()
            zoom = zoom or self.zoom or 1.0
            return cw / zoom, ch / zoom

        def _clamp_view(self) -> None:
            if self.source_image is None:
                return
            iw, ih = self.source_image.size
            vw, vh = self._visible_src_size(self.zoom)
            self.view_left = min(max(0.0, self.view_left), max(0.0, iw - vw))
            self.view_top = min(max(0.0, self.view_top), max(0.0, ih - vh))

        def get_center_ratios(self) -> tuple[float, float]:
            if self.source_image is None:
                return 0.5, 0.5
            iw, ih = self.source_image.size
            vw, vh = self._visible_src_size(self.zoom)
            cx = min(max((self.view_left + vw / 2.0) / max(iw, 1), 0.0), 1.0)
            cy = min(max((self.view_top + vh / 2.0) / max(ih, 1), 0.0), 1.0)
            return cx, cy

        def set_zoom_and_center(self, zoom: float, cx_ratio: float, cy_ratio: float, render: bool = True) -> None:
            if self.source_image is None:
                return
            iw, ih = self.source_image.size
            self.fit_zoom = self._calc_fit_zoom()
            min_zoom = max(min(self.fit_zoom * 0.6, self.fit_zoom), 0.03)
            max_zoom = max(1.0, self.fit_zoom * 16.0)
            self.zoom = min(max(zoom, min_zoom), max_zoom)
            vw, vh = self._visible_src_size(self.zoom)
            self.view_left = cx_ratio * iw - vw / 2.0
            self.view_top = cy_ratio * ih - vh / 2.0
            self._clamp_view()
            if render:
                self.render()

        def get_view_state(self) -> tuple[float, float, float]:
            cx, cy = self.get_center_ratios()
            return self.zoom, cx, cy

        def sync_from_other(self, zoom: float, cx_ratio: float, cy_ratio: float) -> None:
            if self.source_image is None:
                return
            self.set_zoom_and_center(zoom, cx_ratio, cy_ratio)

        def render(self) -> None:
            self._after_id = None
            self.canvas.delete('all')
            cw, ch = self._canvas_size()
            if self.source_image is None:
                self.placeholder = self.canvas.create_text(cw // 2, ch // 2, text=f'{self.side} 侧图片预览', fill=self.muted, font=('Microsoft YaHei UI', 18, 'bold'))
                self._overlay_zoom.configure(text='等待结果')
                return
            iw, ih = self.source_image.size
            self.fit_zoom = self._calc_fit_zoom()
            self._clamp_view()
            img_zoom_w = iw * self.zoom
            img_zoom_h = ih * self.zoom
            if self.zoom <= self.fit_zoom + 1e-4:
                disp_w = max(1, int(round(img_zoom_w)))
                disp_h = max(1, int(round(img_zoom_h)))
                resized = self.source_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
                self._photo = ImageTk.PhotoImage(resized)
                x = int(round((cw - disp_w) / 2.0))
                y = int(round((ch - disp_h) / 2.0))
                self._image_id = self.canvas.create_image(x, y, image=self._photo, anchor='nw')
            else:
                x_pad = max((cw - img_zoom_w) / 2.0, 0.0)
                y_pad = max((ch - img_zoom_h) / 2.0, 0.0)
                view_src_w, view_src_h = self._visible_src_size(self.zoom)
                crop_left = max(0.0, self.view_left)
                crop_top = max(0.0, self.view_top)
                crop_right = min(iw, crop_left + min(view_src_w, iw))
                crop_bottom = min(ih, crop_top + min(view_src_h, ih))
                crop_box = (int(crop_left), int(crop_top), max(int(crop_left) + 1, int(np.ceil(crop_right))), max(int(crop_top) + 1, int(np.ceil(crop_bottom))))
                crop = self.source_image.crop(crop_box)
                disp_w = max(1, int(round((crop_box[2] - crop_box[0]) * self.zoom)))
                disp_h = max(1, int(round((crop_box[3] - crop_box[1]) * self.zoom)))
                resized = crop.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
                self._photo = ImageTk.PhotoImage(resized)
                x = x_pad
                y = y_pad
                if img_zoom_w > cw:
                    x = -(crop_left * self.zoom)
                if img_zoom_h > ch:
                    y = -(crop_top * self.zoom)
                x = int(round(x))
                y = int(round(y))
                self._image_id = self.canvas.create_image(x, y, image=self._photo, anchor='nw')
            self.canvas.create_rectangle(1, 1, cw - 2, ch - 2, outline=self.border, width=1)
            zoom_text = f'{self.zoom * 100:.0f}%'
            if abs(self.zoom - self.fit_zoom) < 1e-4:
                zoom_text += ' · 适应'
            elif abs(self.zoom - 1.0) < 1e-4:
                zoom_text += ' · 1:1'
            self._overlay_zoom.configure(text=zoom_text)

        def _canvas_to_image_coords(self, x: float, y: float) -> tuple[float, float]:
            if self.source_image is None:
                return 0.0, 0.0
            iw, ih = self.source_image.size
            cw, ch = self._canvas_size()
            img_zoom_w = iw * self.zoom
            img_zoom_h = ih * self.zoom
            x_pad = max((cw - img_zoom_w) / 2.0, 0.0)
            y_pad = max((ch - img_zoom_h) / 2.0, 0.0)
            src_x = self.view_left + (x - x_pad) / self.zoom
            src_y = self.view_top + (y - y_pad) / self.zoom
            return min(max(src_x, 0.0), iw), min(max(src_y, 0.0), ih)

        def user_zoom(self, factor: float, x: float | None = None, y: float | None = None, notify: bool = True) -> None:
            if self.source_image is None:
                return
            cw, ch = self._canvas_size()
            x = cw / 2 if x is None else x
            y = ch / 2 if y is None else y
            old_src_x, old_src_y = self._canvas_to_image_coords(x, y)
            new_zoom = self.zoom * factor
            self.fit_zoom = self._calc_fit_zoom()
            min_zoom = max(min(self.fit_zoom * 0.6, self.fit_zoom), 0.03)
            max_zoom = max(1.0, self.fit_zoom * 16.0)
            new_zoom = min(max(new_zoom, min_zoom), max_zoom)
            if abs(new_zoom - self.zoom) < 1e-6:
                return
            self.zoom = new_zoom
            iw, ih = self.source_image.size
            cw, ch = self._canvas_size()
            img_zoom_w = iw * self.zoom
            img_zoom_h = ih * self.zoom
            x_pad = max((cw - img_zoom_w) / 2.0, 0.0)
            y_pad = max((ch - img_zoom_h) / 2.0, 0.0)
            self.view_left = old_src_x - (x - x_pad) / self.zoom
            self.view_top = old_src_y - (y - y_pad) / self.zoom
            self._clamp_view()
            self.render()
            if notify and self.view_change_callback is not None:
                self.view_change_callback(self.side, 'view_changed', *self.get_view_state())

        def user_set_actual(self, notify: bool = True) -> None:
            if self.source_image is None:
                return
            self.actual_size()
            if notify and self.view_change_callback is not None:
                self.view_change_callback(self.side, 'view_changed', *self.get_view_state())

        def user_fit(self, notify: bool = True) -> None:
            if self.source_image is None:
                return
            self.fit_to_view()
            if notify and self.view_change_callback is not None:
                self.view_change_callback(self.side, 'view_changed', *self.get_view_state())

        def _on_mouse_wheel(self, event) -> str:
            if self.source_image is None:
                return 'break'
            delta = 0
            if getattr(event, 'num', None) == 4:
                delta = 1
            elif getattr(event, 'num', None) == 5:
                delta = -1
            else:
                delta = 1 if event.delta > 0 else -1
            factor = 1.12 if delta > 0 else (1 / 1.12)
            self.user_zoom(factor, getattr(event, 'x', None), getattr(event, 'y', None), notify=True)
            return 'break'

        def _on_drag_start(self, event) -> None:
            self._drag_start = (event.x, event.y)

        def _on_drag_motion(self, event) -> str:
            if self.source_image is None or self._drag_start is None:
                return 'break'
            last_x, last_y = self._drag_start
            dx = event.x - last_x
            dy = event.y - last_y
            self._drag_start = (event.x, event.y)
            self.view_left -= dx / max(self.zoom, 1e-6)
            self.view_top -= dy / max(self.zoom, 1e-6)
            self._clamp_view()
            self.render()
            if self.view_change_callback is not None:
                self.view_change_callback(self.side, 'view_changed', *self.get_view_state())
            return 'break'

        def _on_double_click(self, _event=None) -> str:
            if self.source_image is None:
                return 'break'
            if abs(self.zoom - 1.0) < 1e-3:
                self.user_fit(notify=True)
            else:
                self.user_set_actual(notify=True)
            return 'break'

    class CompareReviewWindow:
        def __init__(self, app) -> None:
            self.app = app
            self._syncing_view = False
            self.current_paths = {"A": None, "B": None}
            self.win = tk.Toplevel(app.root)
            self.win.title(f"{APP_NAME} · 全屏双图对比")
            self.win.configure(bg=app.BG)
            self.win.minsize(1500, 900)
            sw = max(1600, int(self.win.winfo_screenwidth()))
            sh = max(900, int(self.win.winfo_screenheight()))
            width = min(sw, 1920)
            height = min(sh, 1080)
            self.win.geometry(f"{width}x{height}+0+0")
            try:
                if sys.platform.startswith("win"):
                    self.win.state("zoomed")
            except Exception:
                pass
            self.win.protocol("WM_DELETE_WINDOW", self.close)
            self.win.bind("<Escape>", lambda _e: self.close())
            self.win.bind("<F11>", lambda _e: self._toggle_zoomed())
            self._build_layout()

        def _toggle_zoomed(self) -> str:
            try:
                if self.win.state() == 'zoomed':
                    self.win.state('normal')
                else:
                    self.win.state('zoomed')
            except Exception:
                pass
            return 'break'

        def _build_layout(self) -> None:
            outer = ttk.Frame(self.win)
            outer.pack(fill='both', expand=True, padx=12, pady=10)
            outer.columnconfigure(0, weight=1)
            outer.rowconfigure(2, weight=1)

            header = ttk.Frame(outer, style='Surface.TFrame')
            header.grid(row=0, column=0, sticky='ew')
            header.columnconfigure(0, weight=1)
            ttk.Label(header, text='全屏双图对比', style='Hero.TLabel').grid(row=0, column=0, sticky='w', padx=14, pady=(10, 2))
            ttk.Label(header, text='默认按 16:9 全屏工作。Esc 关闭全屏页，F11 切换最大化。滚轮缩放，左键拖动平移。', style='MutedSurface.TLabel').grid(row=1, column=0, sticky='w', padx=14, pady=(0, 8))
            ttk.Button(header, text='关闭全屏页', style='Tool.TButton', command=self.close).grid(row=0, column=1, rowspan=2, sticky='e', padx=14, pady=8)

            topbar = ttk.Frame(outer, style='Surface.TFrame')
            topbar.grid(row=1, column=0, sticky='ew', pady=(12, 12))
            topbar.columnconfigure(0, weight=1)
            topbar.columnconfigure(1, weight=1)
            self.pair_title_var = tk.StringVar(value='等待结果')
            self.score_var = tk.StringVar(value='相似度 --')
            self.type_var = tk.StringVar(value='类型 --')
            self.recommend_var = tk.StringVar(value='自动建议：等待结果。')
            left_meta = ttk.Frame(topbar, style='Surface.TFrame')
            left_meta.grid(row=0, column=0, sticky='ew', padx=(14, 8), pady=8)
            ttk.Label(left_meta, textvariable=self.pair_title_var, style='Hero.TLabel').pack(anchor='w')
            ttk.Label(left_meta, textvariable=self.score_var, style='Chip.TLabel').pack(anchor='w', pady=(4, 2))
            ttk.Label(left_meta, textvariable=self.type_var, style='MutedSurface.TLabel').pack(anchor='w')
            self.recommend_label = tk.Label(topbar, textvariable=self.recommend_var, bg=self.app.PANEL, fg=self.app.ACCENT, justify='left', anchor='w', wraplength=560, font=('Microsoft YaHei UI', 11, 'bold'), padx=10, pady=8, relief='solid', bd=1, highlightthickness=1, highlightbackground=self.app.BORDER)
            self.recommend_label.grid(row=0, column=1, sticky='ew', padx=(8, 14), pady=8)

            toolbar = ttk.Frame(outer, style='Surface.TFrame')
            toolbar.grid(row=2, column=0, sticky='ew')
            self.prev_button = ttk.Button(toolbar, text='上一组', style='Big.TButton', command=lambda: self.app._show_pair(self.app.current_index - 1, reset_view=True))
            self.prev_button.pack(side='left', padx=(14, 8), pady=8)
            self.next_button = ttk.Button(toolbar, text='下一组', style='Big.TButton', command=lambda: self.app._show_pair(self.app.current_index + 1, reset_view=True))
            self.next_button.pack(side='left', padx=8, pady=8)
            self.fit_button = ttk.Button(toolbar, text='适应窗口', style='Tool.TButton', command=self.apply_fit)
            self.fit_button.pack(side='left', padx=(16, 8), pady=8)
            self.actual_button = ttk.Button(toolbar, text='1:1 查看', style='Tool.TButton', command=self.apply_actual)
            self.actual_button.pack(side='left', padx=8, pady=8)
            tk.Checkbutton(toolbar, text='同步缩放 / 平移', variable=self.app.compare_sync_zoom_var, bg=self.app.PANEL, fg=self.app.TEXT, activebackground=self.app.PANEL, activeforeground=self.app.TEXT, selectcolor=self.app.BG, font=('Microsoft YaHei UI', 11, 'bold')).pack(side='left', padx=(16, 8), pady=8)
            self.recommend_button = ttk.Button(toolbar, text='按推荐删除', style='Danger.TButton', command=self.app._apply_recommended_delete)
            self.recommend_button.pack(side='right', padx=(8, 14), pady=8)
            self.keep_button = ttk.Button(toolbar, text='保留两张并继续', style='Accent.TButton', command=self.app._keep_and_next)
            self.keep_button.pack(side='right', padx=8, pady=8)

            main = ttk.Frame(outer)
            main.grid(row=3, column=0, sticky='nsew')
            main.columnconfigure(0, weight=1)
            main.columnconfigure(1, weight=1)
            main.rowconfigure(0, weight=1)
            outer.rowconfigure(3, weight=1)

            self.left_card = ttk.Frame(main, style='Soft.TFrame')
            self.left_card.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
            self.left_card.rowconfigure(1, weight=1)
            self.left_card.columnconfigure(0, weight=1)
            ttk.Label(self.left_card, text='目录 A', style='CardTitle.TLabel').grid(row=0, column=0, sticky='w', padx=12, pady=(10, 4))
            self.left_pane = ZoomPreviewPane(self.left_card, 'A', '#fbfdff', self.app.BORDER, self.app.TEXT, self.app.MUTED, self.app.ACCENT, self._on_preview_view_change)
            self.left_pane.frame.grid(row=1, column=0, sticky='nsew', padx=12, pady=(0, 6))
            self.left_info_var = tk.StringVar(value='等待结果…')
            self.left_info_label = tk.Label(self.left_card, textvariable=self.left_info_var, bg=self.app.PANEL_SOFT, fg=self.app.TEXT, justify='left', anchor='nw', font=('Microsoft YaHei UI', 11), wraplength=720)
            self.left_info_label.grid(row=2, column=0, sticky='ew', padx=12, pady=(0, 6))
            left_actions = ttk.Frame(self.left_card, style='Soft.TFrame')
            left_actions.grid(row=3, column=0, sticky='ew', padx=12, pady=(0, 10))
            self.delete_a_button = ttk.Button(left_actions, text='删除左图 A', style='Danger.TButton', command=lambda: self.app._delete_current('A'))
            self.delete_a_button.pack(side='left')
            self.open_a_button = ttk.Button(left_actions, text='打开 A 所在位置', style='Tool.TButton', command=lambda: self.app._open_current_side('A'))
            self.open_a_button.pack(side='left', padx=10)

            self.right_card = ttk.Frame(main, style='Soft.TFrame')
            self.right_card.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
            self.right_card.rowconfigure(1, weight=1)
            self.right_card.columnconfigure(0, weight=1)
            ttk.Label(self.right_card, text='目录 B', style='CardTitle.TLabel').grid(row=0, column=0, sticky='w', padx=12, pady=(10, 4))
            self.right_pane = ZoomPreviewPane(self.right_card, 'B', '#fbfdff', self.app.BORDER, self.app.TEXT, self.app.MUTED, self.app.ACCENT, self._on_preview_view_change)
            self.right_pane.frame.grid(row=1, column=0, sticky='nsew', padx=12, pady=(0, 6))
            self.right_info_var = tk.StringVar(value='等待结果…')
            self.right_info_label = tk.Label(self.right_card, textvariable=self.right_info_var, bg=self.app.PANEL_SOFT, fg=self.app.TEXT, justify='left', anchor='nw', font=('Microsoft YaHei UI', 11), wraplength=720)
            self.right_info_label.grid(row=2, column=0, sticky='ew', padx=12, pady=(0, 6))
            right_actions = ttk.Frame(self.right_card, style='Soft.TFrame')
            right_actions.grid(row=3, column=0, sticky='ew', padx=12, pady=(0, 10))
            self.delete_b_button = ttk.Button(right_actions, text='删除右图 B', style='Danger.TButton', command=lambda: self.app._delete_current('B'))
            self.delete_b_button.pack(side='left')
            self.open_b_button = ttk.Button(right_actions, text='打开 B 所在位置', style='Tool.TButton', command=lambda: self.app._open_current_side('B'))
            self.open_b_button.pack(side='left', padx=10)

        def close(self) -> None:
            for pane in (self.left_pane, self.right_pane):
                if pane is not None:
                    try:
                        pane.destroy()
                    except Exception:
                        pass
            try:
                self.win.destroy()
            finally:
                self.app.compare_window = None

        def clear_state(self, message: str = '暂无结果。') -> None:
            self.pair_title_var.set(message)
            self.score_var.set('相似度 --')
            self.type_var.set('类型 --')
            self.recommend_var.set('自动建议：等待结果。')
            self.left_info_var.set('等待结果…')
            self.right_info_var.set('等待结果…')
            self.current_paths = {'A': None, 'B': None}
            self.left_pane.clear('A 侧图片预览')
            self.right_pane.clear('B 侧图片预览')
            for btn in (self.prev_button, self.next_button, self.fit_button, self.actual_button, self.keep_button, self.recommend_button, self.delete_a_button, self.delete_b_button, self.open_a_button, self.open_b_button):
                btn.configure(state='disabled')

        def _load_side(self, side: str, path: str, reset_view: bool = True) -> None:
            pane = self.left_pane if side == 'A' else self.right_pane
            if pane is None:
                return
            if self.current_paths.get(side) == path and pane.source_image is not None and not reset_view:
                return
            try:
                im = open_image_rgb(path)
                pane.set_image(im, reset_view=reset_view)
                self.current_paths[side] = path
            except Exception as e:
                pane.clear(f'{side} 侧图片无法预览\n{e}')
                self.current_paths[side] = None

        def refresh_from_app(self, reset_view: bool = True) -> None:
            if not self.app.current_results or self.app.current_index < 0 or self.app.current_index >= len(self.app.current_results):
                self.clear_state('暂无结果。')
                return
            row = self.app.current_results[self.app.current_index]
            idx = self.app.current_index
            deleted_side = row.get('deleted_side', '')
            state_suffix = f' · 已删除 {deleted_side}' if deleted_side else ''
            self.pair_title_var.set(f'第 {idx + 1} / {len(self.app.current_results)} 组{state_suffix}')
            self.score_var.set(f"相似度 {row['similarity_score']:.4f}")
            self.type_var.set(f"类型 {row['match_type']}")
            recommendation = self.app._recommend_deletion(row)
            reason_text = '\n'.join('• ' + x for x in recommendation.get('reason_lines', [])[:4])
            self.recommend_var.set(recommendation['text'] + (f'\n{reason_text}' if reason_text else ''))
            self.left_info_var.set(self.app._format_side_info(row, 'A'))
            self.right_info_var.set(self.app._format_side_info(row, 'B'))
            self._load_side('A', row['path_1'], reset_view=reset_view)
            self._load_side('B', row['path_2'], reset_view=reset_view)
            if reset_view:
                self.apply_fit()
            self._update_button_states(row, recommendation)
            try:
                self.win.lift()
            except Exception:
                pass

        def _update_button_states(self, row: dict, recommendation: dict) -> None:
            deleted_side = row.get('deleted_side', '')
            a_exists = Path(row['path_1']).exists()
            b_exists = Path(row['path_2']).exists()
            self.prev_button.configure(state='normal' if self.app.current_index > 0 else 'disabled')
            self.next_button.configure(state='normal' if self.app.current_index < len(self.app.current_results) - 1 else 'disabled')
            self.keep_button.configure(state='normal')
            self.fit_button.configure(state='normal' if a_exists or b_exists else 'disabled')
            self.actual_button.configure(state='normal' if a_exists or b_exists else 'disabled')
            self.delete_a_button.configure(state='normal' if (not deleted_side and a_exists) else 'disabled')
            self.delete_b_button.configure(state='normal' if (not deleted_side and b_exists) else 'disabled')
            self.open_a_button.configure(state='normal' if a_exists else 'disabled')
            self.open_b_button.configure(state='normal' if b_exists else 'disabled')
            rec_side = recommendation.get('side')
            rec_enabled = bool(rec_side) and ((rec_side == 'A' and a_exists) or (rec_side == 'B' and b_exists)) and not deleted_side
            self.recommend_button.configure(state='normal' if rec_enabled else 'disabled')
            self.fullscreen_compare_button.configure(state='normal')
            if self.compare_window is not None:
                self.compare_window.refresh_from_app(reset_view=reset_view)

        def apply_fit(self) -> None:
            panes = [p for p in (self.left_pane, self.right_pane) if p is not None and p.source_image is not None]
            if not panes:
                return
            self._syncing_view = True
            try:
                for pane in panes:
                    pane.user_fit(notify=False)
            finally:
                self._syncing_view = False

        def apply_actual(self) -> None:
            panes = [p for p in (self.left_pane, self.right_pane) if p is not None and p.source_image is not None]
            if not panes:
                return
            self._syncing_view = True
            try:
                for pane in panes:
                    pane.user_set_actual(notify=False)
            finally:
                self._syncing_view = False

        def _on_preview_view_change(self, side: str, _kind: str, zoom: float, cx_ratio: float, cy_ratio: float) -> None:
            if self._syncing_view or not self.app.compare_sync_zoom_var.get():
                return
            other = self.right_pane if side == 'A' else self.left_pane
            if other is None or other.source_image is None:
                return
            self._syncing_view = True
            try:
                other.sync_from_other(zoom, cx_ratio, cy_ratio)
            finally:
                self._syncing_view = False

    class StudioApp:
        BG = "#f4f7fb"
        PANEL = "#ffffff"
        PANEL_SOFT = "#f8fbff"
        BORDER = "#d9e2ec"
        TEXT = "#16202f"
        MUTED = "#62748a"
        ACCENT = "#2563eb"
        ACCENT_2 = "#60a5fa"
        SUCCESS = "#16a34a"
        WARNING = "#d97706"
        DANGER = "#dc2626"
        INPUT = "#ffffff"
        SELECT = "#dbeafe"

        def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
            self.root = root
            self.using_ttkbootstrap = ttkb is not None
            self.base_args = args
            self.queue: queue.Queue = queue.Queue()
            self.job_running = False
            self.job_controls: List = []
            self.job_done_handler = None
            self.progress_mode = "indeterminate"
            self.current_results: List[dict] = []
            self.compare_meta: Optional[dict] = None
            self.current_index = -1
            self.left_pane = None
            self.right_pane = None
            self.activity_widgets: Dict[str, tk.Text] = {}
            self._suspend_tree_event = False
            self._closing = False
            self._syncing_view = False
            self._last_recommendation: dict = {}
            self.compare_window: Optional[CompareReviewWindow] = None
            self._configure_window()
            self._configure_style()
            self._build_menubar()
            self._build_layout()
            if not self._closing:
                self.root.after(80, self._poll_queue)

        def _configure_window(self) -> None:
            self.root.title(f"{APP_NAME} · {APP_VERSION}")
            self.root.configure(bg=self.BG)
            self.root.minsize(1480, 940)
            try:
                self.root.geometry("1760x1080")
                if sys.platform.startswith("win"):
                    self.root.state("zoomed")
            except Exception:
                self.root.geometry("1760x1080")
            try:
                self.root.tk.call("tk", "scaling", 1.25)
            except Exception:
                pass
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
            self.root.bind("<F11>", lambda _e: self._open_compare_window())

        def _configure_style(self) -> None:
            style = ttk.Style(self.root)
            if not self.using_ttkbootstrap:
                try:
                    style.theme_use("clam")
                except Exception:
                    pass

            font_main = ("Microsoft YaHei UI", 13)
            font_bold = ("Microsoft YaHei UI", 13, "bold")
            font_title = ("Microsoft YaHei UI", 24, "bold")
            font_hero = ("Microsoft YaHei UI", 18, "bold")
            font_chip = ("Microsoft YaHei UI", 11, "bold")

            style.configure("TFrame", background=self.BG)
            style.configure("Surface.TFrame", background=self.PANEL, relief="flat", borderwidth=1)
            style.configure("Soft.TFrame", background=self.PANEL_SOFT, relief="flat", borderwidth=1)
            style.configure("TLabel", background=self.BG, foreground=self.TEXT, font=font_main)
            style.configure("Surface.TLabel", background=self.PANEL, foreground=self.TEXT, font=font_main)
            style.configure("Muted.TLabel", background=self.BG, foreground=self.MUTED, font=("Microsoft YaHei UI", 11))
            style.configure("MutedSurface.TLabel", background=self.PANEL, foreground=self.MUTED, font=("Microsoft YaHei UI", 11))
            style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=font_title)
            style.configure("Hero.TLabel", background=self.PANEL, foreground=self.TEXT, font=font_hero)
            style.configure("CardTitle.TLabel", background=self.PANEL_SOFT, foreground=self.TEXT, font=font_hero)
            style.configure("Chip.TLabel", background=self.PANEL_SOFT, foreground=self.ACCENT, font=font_chip)

            style.configure(
                "Big.TButton",
                font=("Microsoft YaHei UI", 15, "bold"),
                padding=(22, 16),
                background=self.PANEL_SOFT,
                foreground=self.TEXT,
                borderwidth=1,
                relief="flat",
            )
            style.map(
                "Big.TButton",
                background=[("active", "#eef5ff"), ("pressed", "#e2ebf8"), ("disabled", "#eef2f7")],
                foreground=[("disabled", "#97a3b6")],
            )

            style.configure(
                "Accent.TButton",
                font=("Microsoft YaHei UI", 15, "bold"),
                padding=(22, 16),
                background=self.ACCENT,
                foreground="#ffffff",
                borderwidth=0,
                relief="flat",
            )
            style.map(
                "Accent.TButton",
                background=[("active", "#1d4ed8"), ("pressed", "#1e40af"), ("disabled", "#9db9f5")],
                foreground=[("disabled", "#ffffff")],
            )

            style.configure(
                "Danger.TButton",
                font=("Microsoft YaHei UI", 15, "bold"),
                padding=(22, 16),
                background=self.DANGER,
                foreground="#ffffff",
                borderwidth=0,
                relief="flat",
            )
            style.map(
                "Danger.TButton",
                background=[("active", "#b91c1c"), ("pressed", "#991b1b"), ("disabled", "#f0b7b7")],
                foreground=[("disabled", "#ffffff")],
            )

            style.configure("Tool.TButton", font=font_bold, padding=(16, 12), background=self.PANEL_SOFT, foreground=self.TEXT, borderwidth=1, relief="flat")
            style.map("Tool.TButton", background=[("active", "#eef5ff"), ("pressed", "#e2ebf8"), ("disabled", "#eef2f7")], foreground=[("disabled", "#97a3b6")])

            style.configure("Studio.TNotebook", background=self.BG, borderwidth=0)
            style.configure(
                "Studio.TNotebook.Tab",
                font=("Microsoft YaHei UI", 14, "bold"),
                padding=(30, 16),
                background="#eaf0f7",
                foreground=self.MUTED,
                borderwidth=0,
            )
            style.map(
                "Studio.TNotebook.Tab",
                background=[("selected", self.PANEL), ("active", "#eef5ff")],
                foreground=[("selected", self.TEXT)],
            )

            style.configure(
                "Studio.Horizontal.TProgressbar",
                troughcolor="#e8eef6",
                background=self.ACCENT,
                bordercolor="#e8eef6",
                lightcolor=self.ACCENT,
                darkcolor=self.ACCENT,
            )
            style.configure(
                "Treeview",
                rowheight=42,
                fieldbackground=self.PANEL,
                background=self.PANEL,
                foreground=self.TEXT,
                font=("Consolas", 12),
                borderwidth=0,
            )
            style.configure("Treeview.Heading", background="#eef4fb", foreground=self.TEXT, font=("Microsoft YaHei UI", 13, "bold"))
            style.map("Treeview", background=[("selected", self.SELECT)], foreground=[("selected", self.TEXT)])
            self.root.option_add("*Font", font_main)

        def _build_menubar(self) -> None:
            menubar = tk.Menu(self.root, tearoff=0)
            file_menu = tk.Menu(menubar, tearoff=0)
            file_menu.add_command(label="一键检查并安装依赖", command=self._start_dependency_install)
            file_menu.add_separator()
            file_menu.add_command(label="退出", command=self._on_close)
            menubar.add_cascade(label="文件", menu=file_menu)

            view_menu = tk.Menu(menubar, tearoff=0)
            view_menu.add_command(label="打开全屏双图对比 (F11)", command=self._open_compare_window)
            menubar.add_cascade(label="视图", menu=view_menu)

            tool_menu = tk.Menu(menubar, tearoff=0)
            tool_menu.add_command(label="切换到双目录对比", command=lambda: self.notebook.select(0) if hasattr(self, "notebook") else None)
            tool_menu.add_command(label="切换到参考图检索", command=lambda: self.notebook.select(1) if hasattr(self, "notebook") else None)
            tool_menu.add_command(label="切换到目录扫描", command=lambda: self.notebook.select(2) if hasattr(self, "notebook") else None)
            menubar.add_cascade(label="工具", menu=tool_menu)

            help_menu = tk.Menu(menubar, tearoff=0)
            help_menu.add_command(label="关于", command=lambda: messagebox.showinfo("关于", f"{APP_NAME}\n{APP_VERSION}\n\n专注图片相似检索、复核与清理。"))
            menubar.add_cascade(label="帮助", menu=help_menu)
            self.root.configure(menu=menubar)

        def _build_layout(self) -> None:
            outer = ttk.Frame(self.root)
            outer.pack(fill="both", expand=True, padx=20, pady=20)

            header = ttk.Frame(outer)
            header.pack(fill="x")
            ttk.Label(header, text=APP_NAME, style="Title.TLabel").pack(side="left")
            ttk.Label(header, text="全中文 · 本地桌面工作台 · 高流畅双图复核", style="Muted.TLabel").pack(side="left", padx=(14, 0), pady=(8, 0))
            ttk.Button(header, text="一键安装依赖", style="Tool.TButton", command=self._start_dependency_install).pack(side="right")

            status_bar = ttk.Frame(outer, style="Surface.TFrame")
            status_bar.pack(fill="x", pady=(16, 14))
            self.status_var = tk.StringVar(value="准备就绪。请选择功能并开始。")
            self.progress_var = tk.StringVar(value="等待任务")
            self.summary_var = tk.StringVar(value="")
            ttk.Label(status_bar, textvariable=self.status_var, style="Surface.TLabel").pack(side="left", padx=18, pady=14)
            ttk.Label(status_bar, textvariable=self.summary_var, style="MutedSurface.TLabel").pack(side="right", padx=18, pady=14)

            progress_row = ttk.Frame(outer)
            progress_row.pack(fill="x", pady=(0, 14))
            ttk.Label(progress_row, textvariable=self.progress_var, style="Muted.TLabel").pack(side="left")
            self.progressbar = ttk.Progressbar(progress_row, style="Studio.Horizontal.TProgressbar", mode="indeterminate")
            self.progressbar.pack(side="left", fill="x", expand=True, padx=(14, 0), ipady=4)

            notebook = ttk.Notebook(outer, style="Studio.TNotebook")
            notebook.pack(fill="both", expand=True)
            self.notebook = notebook

            self.compare_tab = ttk.Frame(notebook)
            self.reference_tab = ttk.Frame(notebook)
            self.scan_tab = ttk.Frame(notebook)
            notebook.add(self.compare_tab, text="双目录对比")
            notebook.add(self.reference_tab, text="参考图 → TXT")
            notebook.add(self.scan_tab, text="目录扫描 → Excel")

            self._build_compare_tab()
            self._build_reference_tab()
            self._build_scan_tab()

            default_tab_index = {"compare": 0, "reference": 1, "scan": 2}.get(getattr(self.base_args, "tab", "compare"), 0)
            notebook.select(default_tab_index)

        def _make_entry_row(self, parent, label: str, var: tk.StringVar, browse_cmd, browse_text: str = "浏览…"):
            row = ttk.Frame(parent, style="Surface.TFrame")
            row.pack(fill="x", pady=8)
            ttk.Label(row, text=label, style="Surface.TLabel", width=14).pack(side="left", padx=(0, 12))
            entry = tk.Entry(
                row,
                textvariable=var,
                bg=self.INPUT,
                fg=self.TEXT,
                insertbackground=self.TEXT,
                relief="solid",
                bd=1,
                highlightthickness=1,
                highlightbackground=self.BORDER,
                highlightcolor=self.ACCENT,
                font=("Microsoft YaHei UI", 14),
            )
            entry.pack(side="left", fill="x", expand=True, ipady=12)
            button = ttk.Button(row, text=browse_text, command=browse_cmd, style="Tool.TButton")
            button.pack(side="left", padx=(12, 0))
            return entry, button

        def _make_activity_panel(self, parent, key: str):
            text = tk.Text(
                parent,
                height=8,
                bg=self.INPUT,
                fg=self.TEXT,
                relief="solid",
                bd=1,
                highlightthickness=1,
                highlightbackground=self.BORDER,
                highlightcolor=self.ACCENT,
                wrap="word",
                font=("Consolas", 12),
            )
            text.pack(fill="both", expand=True, padx=14, pady=14)
            text.configure(state="disabled")
            self.activity_widgets[key] = text
            return text

        def _append_activity(self, key: str, message: str) -> None:
            widget = self.activity_widgets.get(key)
            if widget is None:
                return
            widget.configure(state="normal")
            widget.insert("end", f"{datetime.now().strftime('%H:%M:%S')}  {message}\n")
            line_count = int(widget.index("end-1c").split(".")[0])
            if line_count > 400:
                widget.delete("1.0", "51.0")
            widget.see("end")
            widget.configure(state="disabled")

        def _read_float(self, raw: str, field_name: str, default: float, min_value: float = 0.0, max_value: float = 1.0) -> Optional[float]:
            text = raw.strip()
            if not text:
                return default
            try:
                value = float(text)
            except ValueError:
                messagebox.showwarning("输入有误", f"{field_name} 必须是数字。")
                return None
            if value < min_value or value > max_value:
                messagebox.showwarning("输入有误", f"{field_name} 必须在 {min_value} 到 {max_value} 之间。")
                return None
            return value

        def _read_int(self, raw: str, field_name: str, default: int, min_value: int = 1, max_value: int = 10_000) -> Optional[int]:
            text = raw.strip()
            if not text:
                return default
            try:
                value = int(text)
            except ValueError:
                messagebox.showwarning("输入有误", f"{field_name} 必须是整数。")
                return None
            if value < min_value or value > max_value:
                messagebox.showwarning("输入有误", f"{field_name} 必须在 {min_value} 到 {max_value} 之间。")
                return None
            return value

        def _validate_similarity_inputs(self, min_sim: float, max_sim: float) -> bool:
            if min_sim > max_sim:
                messagebox.showwarning("输入有误", "最低相似度不能高于最高相似度。")
                return False
            return True

        def _on_close(self) -> None:
            if self.job_running:
                if not messagebox.askyesno("任务仍在运行", "当前任务仍在运行。现在关闭窗口，会中止界面显示。是否继续关闭？", icon="warning"):
                    return
            self._closing = True
            if self.compare_window is not None:
                try:
                    self.compare_window.close()
                except Exception:
                    self.compare_window = None
            for pane in (self.left_pane, self.right_pane):
                if pane is not None:
                    try:
                        pane.destroy()
                    except Exception:
                        pass
            self.root.destroy()

        def _browse_dir(self, var: tk.StringVar) -> None:
            path = filedialog.askdirectory()
            if path:
                var.set(path)

        def _browse_file(self, var: tk.StringVar) -> None:
            path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.gif"), ("全部文件", "*.*")])
            if path:
                var.set(path)

        def _start_job(self, worker_fn, done_handler, controls: List, activity_key: str) -> None:
            if self.job_running:
                messagebox.showinfo("任务进行中", "当前已有任务在运行，请等待完成后再开始新的任务。")
                return
            self.job_running = True
            self.job_done_handler = done_handler
            self.job_controls = controls
            for control in controls:
                try:
                    control.configure(state="disabled")
                except Exception:
                    pass
            self.progress_mode = "indeterminate"
            self.progressbar.configure(mode="indeterminate", maximum=100, value=0)
            self.progressbar.start(10)
            self.status_var.set("任务已启动。")
            self.progress_var.set("正在准备…")
            self.summary_var.set("")
            self._append_activity(activity_key, "任务已启动。")

            def emit(kind: str, **payload):
                self.queue.put((kind, payload))

            def runner():
                try:
                    result = worker_fn(emit)
                except Exception as exc:
                    emit("error", error=exc, trace=traceback.format_exc(), activity_key=activity_key)
                else:
                    emit("done", result=result, activity_key=activity_key)

            threading.Thread(target=runner, daemon=True).start()

        def _start_dependency_install(self) -> None:
            requirements_file = Path(__file__).with_name("requirements.txt")
            if self.job_running:
                messagebox.showinfo("任务进行中", "当前已有任务在运行，请等待完成后再执行依赖检查。")
                return

            def worker(emit):
                status = lambda msg: emit("status", message=msg, activity_key="scan")
                used_index = install_runtime_dependencies(requirements_file=requirements_file, status_cb=status)
                return {"used_index": used_index, "requirements": str(requirements_file)}

            def done(result):
                self.status_var.set("依赖安装完成，可直接开始工作。")
                self.summary_var.set("依赖已就绪")
                self._append_activity("scan", f"依赖安装完成，使用源：{result['used_index']}")
                messagebox.showinfo(
                    "依赖安装完成",
                    f"依赖已安装完成。\n\n依赖文件：{result['requirements']}\n安装源：{result['used_index']}",
                )

            self._start_job(worker, done, [], "scan")

        def _finish_job(self) -> None:
            self.job_running = False
            self.job_done_handler = None
            self.progressbar.stop()
            self.progressbar.configure(value=0)
            self.progress_var.set("等待任务")
            for control in self.job_controls:
                try:
                    control.configure(state="normal")
                except Exception:
                    pass
            self.job_controls = []

        def _poll_queue(self) -> None:
            try:
                while True:
                    kind, payload = self.queue.get_nowait()
                    if kind == "status":
                        message = payload["message"]
                        self.status_var.set(message)
                        if payload.get("activity_key"):
                            self._append_activity(payload["activity_key"], message)
                    elif kind == "progress":
                        current = payload["current"]
                        total = max(1, payload["total"])
                        stage = payload["stage"]
                        if total > 1:
                            self.progress_mode = "determinate"
                            self.progressbar.stop()
                            self.progressbar.configure(mode="determinate", maximum=total, value=current)
                            self.progress_var.set(f"{stage} · {current}/{total}")
                        else:
                            self.progress_mode = "indeterminate"
                            self.progressbar.configure(mode="indeterminate")
                            self.progressbar.start(10)
                            self.progress_var.set(stage)
                    elif kind == "activity":
                        self._append_activity(payload["activity_key"], payload["message"])
                    elif kind == "done":
                        handler = self.job_done_handler
                        activity_key = payload.get("activity_key")
                        if activity_key:
                            self._append_activity(activity_key, "任务已完成。")
                        self._finish_job()
                        if handler is not None:
                            handler(payload["result"])
                    elif kind == "error":
                        activity_key = payload.get("activity_key")
                        if activity_key:
                            self._append_activity(activity_key, f"任务失败：{payload['error']}")
                        self._finish_job()
                        messagebox.showerror("执行失败", f"{payload['error']}\n\n{payload['trace']}")
            except queue.Empty:
                pass
            if not self._closing:
                self.root.after(80, self._poll_queue)

        def _build_scan_tab(self) -> None:
            body = ttk.Frame(self.scan_tab)
            body.pack(fill="both", expand=True, padx=18, pady=18)
            body.rowconfigure(0, weight=0)
            body.rowconfigure(1, weight=1)
            body.columnconfigure(0, weight=1)

            top = ttk.Frame(body, style="Surface.TFrame")
            top.grid(row=0, column=0, sticky="ew", pady=(0, 16))
            ttk.Label(top, text="扫描整个目录，导出 Excel 报表", style="Hero.TLabel").pack(anchor="w", padx=18, pady=(16, 6))
            ttk.Label(top, text="适合一次性排查重复图、像素一致图和高相似图片。", style="MutedSurface.TLabel").pack(anchor="w", padx=18)

            form = ttk.Frame(top, style="Surface.TFrame")
            form.pack(fill="x", padx=18, pady=18)
            self.scan_dir_var = tk.StringVar()
            self.scan_out_var = tk.StringVar()
            self.scan_min_var = tk.StringVar(value="0.90")
            self.scan_max_var = tk.StringVar(value="1.00")
            self.scan_topk_var = tk.StringVar(value="20")
            self.scan_recursive_var = tk.BooleanVar(value=True)
            self.scan_depth_var = tk.BooleanVar(value=not self.base_args.no_openclip)

            _, btn_dir = self._make_entry_row(form, "图片目录", self.scan_dir_var, lambda: self._browse_dir(self.scan_dir_var))
            _, btn_out = self._make_entry_row(form, "输出 Excel", self.scan_out_var, self._browse_scan_output, browse_text="另存为…")

            options = ttk.Frame(form, style="Surface.TFrame")
            options.pack(fill="x", pady=(8, 0))
            for label_text, var, width in [("最低相似度", self.scan_min_var, 8), ("最高相似度", self.scan_max_var, 8), ("Top-K", self.scan_topk_var, 8)]:
                block = ttk.Frame(options, style="Surface.TFrame")
                block.pack(side="left", padx=(0, 16))
                ttk.Label(block, text=label_text, style="Surface.TLabel").pack(anchor="w")
                tk.Entry(block, textvariable=var, width=width, bg=self.INPUT, fg=self.TEXT, insertbackground=self.TEXT, relief="solid", bd=1, highlightthickness=1, highlightbackground=self.BORDER, highlightcolor=self.ACCENT, font=("Microsoft YaHei UI", 14)).pack(ipady=10, pady=(6, 0))
            tk.Checkbutton(options, text="递归扫描子文件夹", variable=self.scan_recursive_var, bg=self.PANEL, fg=self.TEXT, activebackground=self.PANEL, activeforeground=self.TEXT, selectcolor=self.BG).pack(side="left", padx=(18, 0), pady=(24, 0))
            tk.Checkbutton(options, text="启用深度特征（更准）", variable=self.scan_depth_var, bg=self.PANEL, fg=self.TEXT, activebackground=self.PANEL, activeforeground=self.TEXT, selectcolor=self.BG).pack(side="left", padx=(18, 0), pady=(24, 0))

            action_row = ttk.Frame(top, style="Surface.TFrame")
            action_row.pack(fill="x", padx=18, pady=(0, 18))
            self.scan_button = ttk.Button(action_row, text="开始扫描", style="Accent.TButton", command=self._start_scan_job)
            self.scan_button.pack(side="left")
            self.scan_summary_var = tk.StringVar(value="")
            ttk.Label(action_row, textvariable=self.scan_summary_var, style="MutedSurface.TLabel").pack(side="left", padx=18, pady=(10, 0))

            log_card = ttk.Frame(body, style="Soft.TFrame")
            log_card.grid(row=1, column=0, sticky="nsew")
            ttk.Label(log_card, text="任务活动", style="CardTitle.TLabel").pack(anchor="w", padx=14, pady=(12, 0))
            self._make_activity_panel(log_card, "scan")
            self.scan_controls = [self.scan_button, btn_dir, btn_out]

        def _browse_scan_output(self) -> None:
            path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
            if path:
                self.scan_out_var.set(path)

        def _start_scan_job(self) -> None:
            args = argparse.Namespace(**vars(self.base_args))
            args.input_dir = self.scan_dir_var.get().strip()
            args.output_xlsx = self.scan_out_var.get().strip() or None
            min_sim = self._read_float(self.scan_min_var.get(), "最低相似度", 0.90)
            max_sim = self._read_float(self.scan_max_var.get(), "最高相似度", 1.00)
            top_k = self._read_int(self.scan_topk_var.get(), "Top-K", 20, min_value=1, max_value=500)
            if min_sim is None or max_sim is None or top_k is None:
                return
            if not self._validate_similarity_inputs(min_sim, max_sim):
                return
            args.min_sim = min_sim
            args.max_sim = max_sim
            args.top_k = top_k
            args.non_recursive = not self.scan_recursive_var.get()
            args.no_openclip = not self.scan_depth_var.get()
            args.clip_mirror = getattr(self, 'clip_mirror_var', tk.StringVar(value='auto')).get() if hasattr(self, 'clip_mirror_var') else getattr(self.base_args, 'clip_mirror', 'auto')
            args.clip_endpoint = getattr(self.base_args, 'clip_endpoint', '')
            args.model_cache_dir = getattr(self, 'model_cache_dir_var', tk.StringVar(value=str(default_model_cache_dir()))).get() if hasattr(self, 'model_cache_dir_var') else getattr(self.base_args, 'model_cache_dir', str(default_model_cache_dir()))

            if not args.input_dir:
                messagebox.showwarning("缺少目录", "请先选择图片目录。")
                return

            def worker(emit):
                status = lambda msg: emit("status", message=msg, activity_key="scan")
                progress = lambda cur, total, stage: emit("progress", current=cur, total=total, stage=stage)
                return run_scan_job(args, status_cb=status, progress_cb=progress)

            def done(result):
                self.scan_summary_var.set(f"有效图片 {result['valid_count']} · 匹配 {result['row_count']} · 异常 {result['error_count']} · {result['elapsed']:.1f}s")
                self.summary_var.set(f"Excel: {Path(result['output_path']).name}")
                self.status_var.set("Excel 报表已生成。")
                messagebox.showinfo("完成", f"Excel 已生成：\n\n{result['output_path']}")

            self._start_job(worker, done, self.scan_controls, "scan")

        def _build_reference_tab(self) -> None:
            body = ttk.Frame(self.reference_tab)
            body.pack(fill="both", expand=True, padx=18, pady=18)
            body.rowconfigure(0, weight=0)
            body.rowconfigure(1, weight=1)
            body.columnconfigure(0, weight=1)

            top = ttk.Frame(body, style="Surface.TFrame")
            top.grid(row=0, column=0, sticky="ew", pady=(0, 16))
            ttk.Label(top, text="选择参考图，在目录中查找相似图片", style="Hero.TLabel").pack(anchor="w", padx=18, pady=(16, 6))
            ttk.Label(top, text="会在目标目录生成 TXT 报告，适合先审阅再处理。", style="MutedSurface.TLabel").pack(anchor="w", padx=18)

            form = ttk.Frame(top, style="Surface.TFrame")
            form.pack(fill="x", padx=18, pady=18)
            self.ref_image_var = tk.StringVar()
            self.ref_dir_var = tk.StringVar()
            self.ref_out_var = tk.StringVar()
            self.ref_min_var = tk.StringVar(value="0.90")
            self.ref_max_var = tk.StringVar(value="1.00")
            self.ref_recursive_var = tk.BooleanVar(value=True)
            self.ref_depth_var = tk.BooleanVar(value=not self.base_args.no_openclip)

            _, btn_img = self._make_entry_row(form, "参考图片", self.ref_image_var, lambda: self._browse_file(self.ref_image_var))
            _, btn_dir = self._make_entry_row(form, "搜索目录", self.ref_dir_var, lambda: self._browse_dir(self.ref_dir_var))
            _, btn_out = self._make_entry_row(form, "输出 TXT", self.ref_out_var, self._browse_reference_output, browse_text="另存为…")

            options = ttk.Frame(form, style="Surface.TFrame")
            options.pack(fill="x", pady=(8, 0))
            for label_text, var, width in [("最低相似度", self.ref_min_var, 8), ("最高相似度", self.ref_max_var, 8)]:
                block = ttk.Frame(options, style="Surface.TFrame")
                block.pack(side="left", padx=(0, 16))
                ttk.Label(block, text=label_text, style="Surface.TLabel").pack(anchor="w")
                tk.Entry(block, textvariable=var, width=width, bg=self.INPUT, fg=self.TEXT, insertbackground=self.TEXT, relief="solid", bd=1, highlightthickness=1, highlightbackground=self.BORDER, highlightcolor=self.ACCENT, font=("Microsoft YaHei UI", 14)).pack(ipady=10, pady=(6, 0))
            tk.Checkbutton(options, text="递归扫描子文件夹", variable=self.ref_recursive_var, bg=self.PANEL, fg=self.TEXT, activebackground=self.PANEL, activeforeground=self.TEXT, selectcolor=self.BG).pack(side="left", padx=(18, 0), pady=(24, 0))
            tk.Checkbutton(options, text="启用深度特征（更准）", variable=self.ref_depth_var, bg=self.PANEL, fg=self.TEXT, activebackground=self.PANEL, activeforeground=self.TEXT, selectcolor=self.BG).pack(side="left", padx=(18, 0), pady=(24, 0))

            action_row = ttk.Frame(top, style="Surface.TFrame")
            action_row.pack(fill="x", padx=18, pady=(0, 18))
            self.ref_button = ttk.Button(action_row, text="开始检索", style="Accent.TButton", command=self._start_reference_job)
            self.ref_button.pack(side="left")
            self.ref_summary_var = tk.StringVar(value="")
            ttk.Label(action_row, textvariable=self.ref_summary_var, style="MutedSurface.TLabel").pack(side="left", padx=18, pady=(10, 0))

            log_card = ttk.Frame(body, style="Soft.TFrame")
            log_card.grid(row=1, column=0, sticky="nsew")
            ttk.Label(log_card, text="任务活动", style="CardTitle.TLabel").pack(anchor="w", padx=14, pady=(12, 0))
            self._make_activity_panel(log_card, "reference")
            self.ref_controls = [self.ref_button, btn_img, btn_dir, btn_out]

        def _browse_reference_output(self) -> None:
            path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("TXT", "*.txt")])
            if path:
                self.ref_out_var.set(path)

        def _start_reference_job(self) -> None:
            args = argparse.Namespace(**vars(self.base_args))
            args.reference_image = self.ref_image_var.get().strip()
            args.search_dir = self.ref_dir_var.get().strip()
            args.output_txt = self.ref_out_var.get().strip() or None
            min_sim = self._read_float(self.ref_min_var.get(), "最低相似度", 0.90)
            max_sim = self._read_float(self.ref_max_var.get(), "最高相似度", 1.00)
            if min_sim is None or max_sim is None:
                return
            if not self._validate_similarity_inputs(min_sim, max_sim):
                return
            args.min_sim = min_sim
            args.max_sim = max_sim
            args.non_recursive = not self.ref_recursive_var.get()
            args.no_openclip = not self.ref_depth_var.get()
            args.clip_mirror = getattr(self, 'clip_mirror_var', tk.StringVar(value='auto')).get() if hasattr(self, 'clip_mirror_var') else getattr(self.base_args, 'clip_mirror', 'auto')
            args.clip_endpoint = getattr(self.base_args, 'clip_endpoint', '')
            args.model_cache_dir = getattr(self, 'model_cache_dir_var', tk.StringVar(value=str(default_model_cache_dir()))).get() if hasattr(self, 'model_cache_dir_var') else getattr(self.base_args, 'model_cache_dir', str(default_model_cache_dir()))

            if not args.reference_image or not args.search_dir:
                messagebox.showwarning("信息不完整", "请先选择参考图片和搜索目录。")
                return

            def worker(emit):
                status = lambda msg: emit("status", message=msg, activity_key="reference")
                progress = lambda cur, total, stage: emit("progress", current=cur, total=total, stage=stage)
                return run_reference_job(args, status_cb=status, progress_cb=progress)

            def done(result):
                self.ref_summary_var.set(f"有效候选 {result['valid_count']} · 匹配 {result['row_count']} · 异常 {result['error_count']} · {result['elapsed']:.1f}s")
                self.summary_var.set(f"TXT: {Path(result['output_path']).name}")
                self.status_var.set("TXT 报告已生成。")
                messagebox.showinfo("完成", f"TXT 已生成：\n\n{result['output_path']}")

            self._start_job(worker, done, self.ref_controls, "reference")

        def _build_compare_tab(self) -> None:
            body = ttk.Frame(self.compare_tab)
            body.pack(fill="both", expand=True, padx=18, pady=18)
            body.rowconfigure(1, weight=14)
            body.rowconfigure(2, weight=2)
            body.columnconfigure(0, weight=1)

            top = ttk.Frame(body, style="Surface.TFrame")
            top.grid(row=0, column=0, sticky="ew", pady=(0, 12))
            ttk.Label(top, text="A / B 双目录相似图片复核", style="Hero.TLabel").pack(anchor="w", padx=20, pady=(16, 2))
            ttk.Label(top, text="Lightroom 风格双图对照：更大的预览、同步缩放、1:1 查看、滚轮放大、左右独立删除。", style="MutedSurface.TLabel").pack(anchor="w", padx=20)

            form = ttk.Frame(top, style="Surface.TFrame")
            form.pack(fill="x", padx=20, pady=(12, 8))
            self.dir_a_var = tk.StringVar()
            self.dir_b_var = tk.StringVar()
            self.compare_min_var = tk.StringVar(value="0.92")
            self.compare_max_var = tk.StringVar(value="1.00")
            self.compare_topk_var = tk.StringVar(value="3")
            self.compare_recursive_var = tk.BooleanVar(value=True)
            self.compare_depth_var = tk.BooleanVar(value=not self.base_args.no_openclip)
            self.compare_sync_zoom_var = tk.BooleanVar(value=True)

            _, btn_a = self._make_entry_row(form, "目录 A", self.dir_a_var, lambda: self._browse_dir(self.dir_a_var))
            _, btn_b = self._make_entry_row(form, "目录 B", self.dir_b_var, lambda: self._browse_dir(self.dir_b_var))

            options = ttk.Frame(form, style="Surface.TFrame")
            options.pack(fill="x", pady=(4, 0))
            for label_text, var, width in [("最低相似度", self.compare_min_var, 8), ("最高相似度", self.compare_max_var, 8), ("每张 A 保留候选数", self.compare_topk_var, 8)]:
                block = ttk.Frame(options, style="Surface.TFrame")
                block.pack(side="left", padx=(0, 18))
                ttk.Label(block, text=label_text, style="Surface.TLabel").pack(anchor="w")
                tk.Entry(block, textvariable=var, width=width, bg=self.INPUT, fg=self.TEXT, insertbackground=self.TEXT, relief="solid", bd=1, highlightthickness=1, highlightbackground=self.BORDER, highlightcolor=self.ACCENT, font=("Microsoft YaHei UI", 14)).pack(ipady=10, pady=(6, 0))
            tk.Checkbutton(options, text="递归扫描子文件夹", variable=self.compare_recursive_var, bg=self.PANEL, fg=self.TEXT, activebackground=self.PANEL, activeforeground=self.TEXT, selectcolor=self.BG, font=("Microsoft YaHei UI", 12)).pack(side="left", padx=(18, 0), pady=(24, 0))
            tk.Checkbutton(options, text="启用深度特征（更准）", variable=self.compare_depth_var, bg=self.PANEL, fg=self.TEXT, activebackground=self.PANEL, activeforeground=self.TEXT, selectcolor=self.BG, font=("Microsoft YaHei UI", 12)).pack(side="left", padx=(18, 0), pady=(24, 0))
            tk.Checkbutton(options, text="同步缩放 / 平移", variable=self.compare_sync_zoom_var, bg=self.PANEL, fg=self.TEXT, activebackground=self.PANEL, activeforeground=self.TEXT, selectcolor=self.BG, font=("Microsoft YaHei UI", 12, "bold")).pack(side="left", padx=(18, 0), pady=(24, 0))

            mirror_row = ttk.Frame(form, style="Surface.TFrame")
            mirror_row.pack(fill="x", pady=(10, 0))
            self.clip_mirror_var = tk.StringVar(value=getattr(self.base_args, 'clip_mirror', 'auto'))
            self.model_cache_dir_var = tk.StringVar(value=getattr(self.base_args, 'model_cache_dir', str(default_model_cache_dir())))
            ttk.Label(mirror_row, text="模型源", style="Surface.TLabel", width=14).pack(side="left", padx=(0, 12))
            mirror_box = ttk.Combobox(mirror_row, textvariable=self.clip_mirror_var, values=["auto", "cn", "official"], state="readonly", width=10, font=("Microsoft YaHei UI", 12))
            mirror_box.pack(side="left", ipady=6)
            ttk.Label(mirror_row, text="模型缓存目录", style="Surface.TLabel").pack(side="left", padx=(18, 10))
            cache_entry = tk.Entry(mirror_row, textvariable=self.model_cache_dir_var, bg=self.INPUT, fg=self.TEXT, insertbackground=self.TEXT, relief="solid", bd=1, highlightthickness=1, highlightbackground=self.BORDER, highlightcolor=self.ACCENT, font=("Microsoft YaHei UI", 12))
            cache_entry.pack(side="left", fill="x", expand=True, ipady=8)
            ttk.Label(mirror_row, text="提示：默认先尝试中国大陆镜像，再回退官方。删除这个缓存目录即可清理自动下载模型。", style="MutedSurface.TLabel").pack(side="left", padx=(12, 0))

            action_row = ttk.Frame(top, style="Surface.TFrame")
            action_row.pack(fill="x", padx=20, pady=(6, 16))
            self.compare_button = ttk.Button(action_row, text="开始 A / B 对比", style="Accent.TButton", command=self._start_compare_job)
            self.compare_button.pack(side="left")
            self.export_compare_button = ttk.Button(action_row, text="导出复核报告", style="Big.TButton", command=self._export_compare_report, state="disabled")
            self.export_compare_button.pack(side="left", padx=12)
            self.fit_view_button = ttk.Button(action_row, text="适应窗口", style="Tool.TButton", command=self._apply_fit_view, state="disabled")
            self.fit_view_button.pack(side="left", padx=(18, 8))
            self.actual_view_button = ttk.Button(action_row, text="1:1 查看", style="Tool.TButton", command=self._apply_actual_view, state="disabled")
            self.actual_view_button.pack(side="left")
            self.fullscreen_compare_button = ttk.Button(action_row, text="打开全屏对比", style="Big.TButton", command=self._open_compare_window, state="disabled")
            self.fullscreen_compare_button.pack(side="left", padx=(12, 0))
            self.compare_summary_var = tk.StringVar(value="")
            ttk.Label(action_row, textvariable=self.compare_summary_var, style="MutedSurface.TLabel").pack(side="left", padx=18, pady=(10, 0))
            self.compare_controls = [self.compare_button, self.export_compare_button, btn_a, btn_b, self.fit_view_button, self.actual_view_button, self.fullscreen_compare_button]

            workspace = ttk.Frame(body)
            workspace.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
            workspace.columnconfigure(0, weight=6)
            workspace.columnconfigure(1, weight=2)
            workspace.columnconfigure(2, weight=6)
            workspace.rowconfigure(0, weight=1)

            self.left_card = ttk.Frame(workspace, style="Soft.TFrame")
            self.left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
            ttk.Label(self.left_card, text="目录 A", style="CardTitle.TLabel").pack(anchor="w", padx=16, pady=(14, 6))
            self.left_pane = ZoomPreviewPane(self.left_card, "A", "#fbfdff", self.BORDER, self.TEXT, self.MUTED, self.ACCENT, self._on_preview_view_change)
            self.left_pane.frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))
            self.left_info_var = tk.StringVar(value="等待结果…")
            self.left_info_label = tk.Label(self.left_card, textvariable=self.left_info_var, bg=self.PANEL_SOFT, fg=self.TEXT, justify="left", anchor="nw", font=("Microsoft YaHei UI", 12), wraplength=760)
            self.left_info_label.pack(fill="x", padx=16, pady=(0, 10))
            left_buttons = ttk.Frame(self.left_card, style="Soft.TFrame")
            left_buttons.pack(fill="x", padx=16, pady=(0, 16))
            self.delete_a_button = ttk.Button(left_buttons, text="删除左图 A", style="Danger.TButton", command=lambda: self._delete_current("A"), state="disabled")
            self.delete_a_button.pack(side="left")
            self.open_a_button = ttk.Button(left_buttons, text="打开 A 所在位置", style="Tool.TButton", command=lambda: self._open_current_side("A"), state="disabled")
            self.open_a_button.pack(side="left", padx=10)

            center_card = ttk.Frame(workspace, style="Surface.TFrame")
            center_card.grid(row=0, column=1, sticky="nsew", padx=10)
            self.pair_title_var = tk.StringVar(value="尚未开始")
            ttk.Label(center_card, textvariable=self.pair_title_var, style="Hero.TLabel", justify="center").pack(anchor="center", pady=(24, 8), padx=14)
            self.score_var = tk.StringVar(value="相似度 --")
            ttk.Label(center_card, textvariable=self.score_var, style="Chip.TLabel").pack(anchor="center", pady=(0, 8))
            self.type_var = tk.StringVar(value="类型 --")
            ttk.Label(center_card, textvariable=self.type_var, style="MutedSurface.TLabel", justify="center").pack(anchor="center", pady=(0, 8))
            self.center_hint_var = tk.StringVar(value="滚轮缩放，按住左键拖动平移；双击可切换 1:1 / 适应窗口。")
            ttk.Label(center_card, textvariable=self.center_hint_var, style="MutedSurface.TLabel", justify="center", wraplength=280).pack(anchor="center", pady=(0, 14), padx=16)
            self.recommend_var = tk.StringVar(value="自动建议：等待结果。")
            self.recommend_label = tk.Label(center_card, textvariable=self.recommend_var, bg=self.PANEL, fg=self.ACCENT, justify="left", wraplength=300, font=("Microsoft YaHei UI", 12, "bold"), padx=12, pady=12, relief="solid", bd=1, highlightthickness=1, highlightbackground=self.BORDER)
            self.recommend_label.pack(fill="x", padx=16, pady=(0, 14))

            nav_row = ttk.Frame(center_card, style="Surface.TFrame")
            nav_row.pack(fill="x", padx=18, pady=(0, 4))
            self.prev_button = ttk.Button(nav_row, text="上一组", style="Big.TButton", command=lambda: self._show_pair(self.current_index - 1), state="disabled")
            self.prev_button.pack(fill="x", pady=4)
            self.next_button = ttk.Button(nav_row, text="下一组", style="Big.TButton", command=lambda: self._show_pair(self.current_index + 1), state="disabled")
            self.next_button.pack(fill="x", pady=4)
            self.recommend_button = ttk.Button(nav_row, text="按推荐删除", style="Danger.TButton", command=self._apply_recommended_delete, state="disabled")
            self.recommend_button.pack(fill="x", pady=(12, 4))
            self.keep_button = ttk.Button(nav_row, text="保留两张并继续", style="Accent.TButton", command=self._keep_and_next, state="disabled")
            self.keep_button.pack(fill="x", pady=(10, 4))
            quick_note = tk.Label(center_card, text="推荐删除只给出默认建议：优先保留更高分辨率、更大文件、更新的文件；完全相同时默认保留 A。", bg=self.PANEL, fg=self.MUTED, justify="left", wraplength=300, font=("Microsoft YaHei UI", 11))
            quick_note.pack(fill="x", padx=18, pady=(14, 16))

            self.right_card = ttk.Frame(workspace, style="Soft.TFrame")
            self.right_card.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
            ttk.Label(self.right_card, text="目录 B", style="CardTitle.TLabel").pack(anchor="w", padx=16, pady=(14, 6))
            self.right_pane = ZoomPreviewPane(self.right_card, "B", "#fbfdff", self.BORDER, self.TEXT, self.MUTED, self.ACCENT, self._on_preview_view_change)
            self.right_pane.frame.pack(fill="both", expand=True, padx=16, pady=(0, 10))
            self.right_info_var = tk.StringVar(value="等待结果…")
            self.right_info_label = tk.Label(self.right_card, textvariable=self.right_info_var, bg=self.PANEL_SOFT, fg=self.TEXT, justify="left", anchor="nw", font=("Microsoft YaHei UI", 12), wraplength=760)
            self.right_info_label.pack(fill="x", padx=16, pady=(0, 10))
            right_buttons = ttk.Frame(self.right_card, style="Soft.TFrame")
            right_buttons.pack(fill="x", padx=16, pady=(0, 16))
            self.delete_b_button = ttk.Button(right_buttons, text="删除右图 B", style="Danger.TButton", command=lambda: self._delete_current("B"), state="disabled")
            self.delete_b_button.pack(side="left")
            self.open_b_button = ttk.Button(right_buttons, text="打开 B 所在位置", style="Tool.TButton", command=lambda: self._open_current_side("B"), state="disabled")
            self.open_b_button.pack(side="left", padx=10)

            bottom = ttk.Frame(body)
            bottom.grid(row=2, column=0, sticky="nsew")
            bottom.columnconfigure(0, weight=4)
            bottom.columnconfigure(1, weight=2)
            bottom.rowconfigure(0, weight=1)

            table_card = ttk.Frame(bottom, style="Soft.TFrame")
            table_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
            ttk.Label(table_card, text="匹配列表", style="CardTitle.TLabel").pack(anchor="w", padx=16, pady=(12, 8))
            tree_wrap = ttk.Frame(table_card, style="Soft.TFrame")
            tree_wrap.pack(fill="both", expand=True, padx=16, pady=(0, 16))
            tree_wrap.rowconfigure(0, weight=1)
            tree_wrap.columnconfigure(0, weight=1)
            self.result_tree = ttk.Treeview(tree_wrap, columns=("idx", "score", "type", "a_name", "a_size", "b_name", "b_size", "status"), show="headings", selectmode="browse")
            self.result_tree.grid(row=0, column=0, sticky="nsew")
            for col, text, width in [("idx", "#", 54), ("score", "相似度", 100), ("type", "类型", 132), ("a_name", "A 文件", 250), ("a_size", "A 大小", 96), ("b_name", "B 文件", 250), ("b_size", "B 大小", 96), ("status", "状态", 110)]:
                self.result_tree.heading(col, text=text)
                self.result_tree.column(col, width=width, anchor="w")
            self.result_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
            tree_scroll = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.result_tree.yview)
            tree_scroll.grid(row=0, column=1, sticky="ns")
            self.result_tree.configure(yscrollcommand=tree_scroll.set)

            activity_card = ttk.Frame(bottom, style="Soft.TFrame")
            activity_card.grid(row=0, column=1, sticky="nsew")
            ttk.Label(activity_card, text="任务活动", style="CardTitle.TLabel").pack(anchor="w", padx=16, pady=(12, 8))
            self._make_activity_panel(activity_card, "compare")
        def _open_compare_window(self) -> str | None:
            if not self.current_results:
                messagebox.showinfo('没有结果', '请先完成 A / B 对比，再打开全屏双图对比页。')
                return 'break'
            self._ensure_compare_window(focus=True, reset_view=False)
            return 'break'

        def _ensure_compare_window(self, focus: bool = True, reset_view: bool = False) -> None:
            try:
                if self.compare_window is None or not self.compare_window.win.winfo_exists():
                    self.compare_window = CompareReviewWindow(self)
            except Exception:
                self.compare_window = CompareReviewWindow(self)
            self.compare_window.refresh_from_app(reset_view=reset_view)
            if focus and self.compare_window is not None:
                try:
                    self.compare_window.win.deiconify()
                    self.compare_window.win.lift()
                    self.compare_window.win.focus_force()
                except Exception:
                    pass

        def _start_compare_job(self) -> None:
            args = argparse.Namespace(**vars(self.base_args))
            args.dir_a = self.dir_a_var.get().strip()
            args.dir_b = self.dir_b_var.get().strip()
            min_sim = self._read_float(self.compare_min_var.get(), "最低相似度", 0.92)
            max_sim = self._read_float(self.compare_max_var.get(), "最高相似度", 1.00)
            top_k_per_a = self._read_int(self.compare_topk_var.get(), "每张 A 保留候选数", 3, min_value=1, max_value=100)
            if min_sim is None or max_sim is None or top_k_per_a is None:
                return
            if not self._validate_similarity_inputs(min_sim, max_sim):
                return
            args.min_sim = min_sim
            args.max_sim = max_sim
            args.top_k_per_a = top_k_per_a
            args.non_recursive = not self.compare_recursive_var.get()
            args.no_openclip = not self.compare_depth_var.get()
            args.clip_mirror = self.clip_mirror_var.get()
            args.clip_endpoint = getattr(self.base_args, 'clip_endpoint', '')
            args.model_cache_dir = self.model_cache_dir_var.get().strip() or str(default_model_cache_dir())

            if not args.dir_a or not args.dir_b:
                messagebox.showwarning("信息不完整", "请先选择目录 A 和目录 B。")
                return
            if Path(clean_input_path(args.dir_a)).resolve() == Path(clean_input_path(args.dir_b)).resolve():
                messagebox.showwarning("目录重复", "目录 A 和目录 B 不能是同一个目录。")
                return

            def worker(emit):
                status = lambda msg: emit("status", message=msg, activity_key="compare")
                progress = lambda cur, total, stage: emit("progress", current=cur, total=total, stage=stage)
                return run_ab_compare_job(args, status_cb=status, progress_cb=progress)

            def done(result):
                self._load_compare_results(result)
                rows = result["rows"]
                self.compare_summary_var.set(f"匹配 {len(rows)} 组 · A 异常 {len(result['errors_a'])} · B 异常 {len(result['errors_b'])} · {result['elapsed']:.1f}s")
                self.summary_var.set(f"A/B 结果: {len(rows)} 组")
                self.status_var.set("A / B 对比完成。")
                if rows:
                    self.root.after(120, lambda: self._ensure_compare_window(focus=True, reset_view=True))
                else:
                    messagebox.showinfo("完成", "没有找到符合阈值的相似图片。")

            self._start_job(worker, done, self.compare_controls, "compare")

        def _load_compare_results(self, result: dict) -> None:
            self.compare_meta = result
            self.current_results = result["rows"]
            self.current_index = -1
            self.result_tree.delete(*self.result_tree.get_children())
            self.export_compare_button.configure(state="normal" if self.current_results else "disabled")
            self.fullscreen_compare_button.configure(state="normal" if self.current_results else "disabled")
            self._append_activity("compare", f"准备加载 {len(self.current_results)} 组结果。")
            self._populate_tree_chunk(0)

        def _populate_tree_chunk(self, start: int, chunk_size: int = 250) -> None:
            end = min(start + chunk_size, len(self.current_results))
            for idx in range(start, end):
                row = self.current_results[idx]
                self.result_tree.insert(
                    "",
                    "end",
                    iid=str(idx),
                    values=(
                        idx + 1,
                        f"{row['similarity_score']:.4f}",
                        row['match_type'],
                        row.get('name_1', Path(row['path_1']).name),
                        human_size(row['file_size_1']),
                        row.get('name_2', Path(row['path_2']).name),
                        human_size(row['file_size_2']),
                        row.get('deleted_side', '') or "待处理",
                    ),
                )
            if end < len(self.current_results):
                self.progress_var.set(f"正在载入结果列表 · {end}/{len(self.current_results)}")
                self.root.after(1, lambda: self._populate_tree_chunk(end, chunk_size))
                return
            if self.current_results:
                self.progress_var.set("结果列表已载入")
                self._show_pair(0)
            else:
                self._show_empty_state("没有符合阈值的匹配结果。")

        def _get_file_mtime(self, path: str) -> float:
            try:
                return Path(path).stat().st_mtime
            except Exception:
                return 0.0

        def _parse_resolution(self, text: str) -> tuple[int, int]:
            m = re.match(r"\s*(\d+)\s*x\s*(\d+)\s*", text or "")
            if not m:
                return 0, 0
            return int(m.group(1)), int(m.group(2))

        def _format_mtime(self, path: str) -> str:
            ts = self._get_file_mtime(path)
            if not ts:
                return '未知'
            return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')

        def _format_side_info(self, row: dict, side: str) -> str:
            prefix = '1' if side == 'A' else '2'
            deleted_side = row.get('deleted_side', '')
            state = '已删除' if deleted_side == side else '保留中'
            return '\n'.join([
                f"文件名：{row.get(f'name_{prefix}', Path(row[f'path_{prefix}']).name)}",
                f"文件大小：{human_size(row[f'file_size_{prefix}'])}",
                f"分辨率：{row[f'resolution_{prefix}']}",
                f"修改时间：{self._format_mtime(row[f'path_{prefix}'])}",
                f"所在目录：{row.get(f'folder_{prefix}', str(Path(row[f'path_{prefix}']).parent))}",
                f"相对路径：{row[f'rel_path_{prefix}']}",
                f"状态：{state}",
            ])

        def _recommend_deletion(self, row: dict) -> dict:
            deleted_side = row.get('deleted_side', '')
            if deleted_side:
                return {'side': None, 'text': f'当前已删除 {deleted_side} 侧，不再给出建议。', 'reason_lines': []}
            w1, h1 = self._parse_resolution(row.get('resolution_1', '0x0'))
            w2, h2 = self._parse_resolution(row.get('resolution_2', '0x0'))
            px1 = w1 * h1
            px2 = w2 * h2
            size1 = int(row.get('file_size_1', 0) or 0)
            size2 = int(row.get('file_size_2', 0) or 0)
            mt1 = self._get_file_mtime(row['path_1'])
            mt2 = self._get_file_mtime(row['path_2'])
            reasons: list[str] = []
            keep_side = None
            confidence = '中'
            match_type = row.get('match_type', '')

            if match_type in {'exact_file', 'cross_same_file_hash'}:
                keep_side = 'A'
                reasons.append('两张文件内容完全一致，默认保留目录 A，建议删右图 B。')
                confidence = '高'
            elif match_type in {'exact_pixel', 'cross_same_pixel_hash'} and px1 == px2 and size1 == size2:
                keep_side = 'A'
                reasons.append('两张像素完全一致，尺寸和大小也一致，默认保留目录 A。')
                confidence = '高'
            else:
                score_a = 0.0
                score_b = 0.0
                if px1 and px2 and px1 != px2:
                    ratio = max(px1, px2) / max(min(px1, px2), 1)
                    if ratio >= 1.08:
                        if px1 > px2:
                            score_a += 4.0
                            reasons.append(f'A 分辨率更高：{row["resolution_1"]} > {row["resolution_2"]}')
                        else:
                            score_b += 4.0
                            reasons.append(f'B 分辨率更高：{row["resolution_2"]} > {row["resolution_1"]}')
                if size1 and size2 and size1 != size2:
                    ratio = max(size1, size2) / max(min(size1, size2), 1)
                    if ratio >= 1.12:
                        if size1 > size2:
                            score_a += 2.0
                            reasons.append(f'A 文件更大：{human_size(size1)} > {human_size(size2)}')
                        else:
                            score_b += 2.0
                            reasons.append(f'B 文件更大：{human_size(size2)} > {human_size(size1)}')
                if mt1 and mt2 and mt1 != mt2:
                    delta_days = abs(mt1 - mt2) / 86400.0
                    if delta_days >= 1.0:
                        if mt1 > mt2:
                            score_a += 0.8
                            reasons.append(f'A 修改时间更新：{self._format_mtime(row["path_1"])}')
                        else:
                            score_b += 0.8
                            reasons.append(f'B 修改时间更新：{self._format_mtime(row["path_2"])}')
                if score_a == score_b:
                    keep_side = 'A'
                    reasons.append('两边质量指标接近，默认保留目录 A，建议删右图 B。')
                    confidence = '低'
                elif score_a > score_b:
                    keep_side = 'A'
                    confidence = '高' if abs(score_a - score_b) >= 3.0 else '中'
                else:
                    keep_side = 'B'
                    confidence = '高' if abs(score_b - score_a) >= 3.0 else '中'

            delete_side = 'B' if keep_side == 'A' else 'A'
            headline = f'自动建议：删{delete_side}图 {delete_side}，保留{keep_side}图 {keep_side}。置信度：{confidence}'
            return {'side': delete_side, 'keep_side': keep_side, 'text': headline, 'reason_lines': reasons}

        def _show_empty_state(self, message: str) -> None:
            self.pair_title_var.set(message)
            self.score_var.set('相似度 --')
            self.type_var.set('类型 --')
            self.center_hint_var.set('滚轮缩放，按住左键拖动平移；双击可切换 1:1 / 适应窗口。')
            self.recommend_var.set('自动建议：等待结果。')
            if self.left_pane is not None:
                self.left_pane.clear('A 侧图片预览')
            if self.right_pane is not None:
                self.right_pane.clear('B 侧图片预览')
            self.left_info_var.set('等待结果…')
            self.right_info_var.set('等待结果…')
            for btn in [self.prev_button, self.next_button, self.keep_button, self.delete_a_button, self.delete_b_button, self.open_a_button, self.open_b_button, self.fit_view_button, self.actual_view_button, self.recommend_button, self.fullscreen_compare_button]:
                btn.configure(state='disabled')
            if self.compare_window is not None:
                self.compare_window.clear_state(message)

        def _load_preview_source(self, side: str, path: str, reset_view: bool = True) -> None:
            pane = self.left_pane if side == 'A' else self.right_pane
            if pane is None:
                return
            try:
                im = open_image_rgb(path)
                pane.set_image(im, reset_view=reset_view)
            except Exception as e:
                pane.clear(f'{side} 侧图片无法预览\n{e}')

        def _apply_pair_view(self, zoom: float | None = None, center: tuple[float, float] | None = None, mode: str | None = None, source_side: str | None = None) -> None:
            panes = [p for p in (self.left_pane, self.right_pane) if p is not None and p.source_image is not None]
            if not panes:
                return
            self._syncing_view = True
            try:
                if mode == 'fit':
                    for pane in panes:
                        pane.user_fit(notify=False)
                elif mode == 'actual':
                    if source_side == 'B' and self.right_pane is not None and self.left_pane is not None and self.right_pane.source_image is not None:
                        zoom, cx, cy = self.right_pane.get_view_state()
                        self.right_pane.user_set_actual(notify=False)
                        zoom, cx, cy = self.right_pane.get_view_state()
                        if self.compare_sync_zoom_var.get():
                            self.left_pane.sync_from_other(zoom, cx, cy)
                    else:
                        for pane in panes:
                            pane.user_set_actual(notify=False)
                elif zoom is not None and center is not None:
                    cx, cy = center
                    if source_side == 'A':
                        if self.left_pane is not None:
                            self.left_pane.set_zoom_and_center(zoom, cx, cy)
                        if self.compare_sync_zoom_var.get() and self.right_pane is not None:
                            self.right_pane.sync_from_other(zoom, cx, cy)
                    elif source_side == 'B':
                        if self.right_pane is not None:
                            self.right_pane.set_zoom_and_center(zoom, cx, cy)
                        if self.compare_sync_zoom_var.get() and self.left_pane is not None:
                            self.left_pane.sync_from_other(zoom, cx, cy)
                    else:
                        for pane in panes:
                            pane.set_zoom_and_center(zoom, cx, cy)
            finally:
                self._syncing_view = False

        def _apply_fit_view(self) -> None:
            self._apply_pair_view(mode='fit')
            if self.compare_window is not None:
                self.compare_window.apply_fit()

        def _apply_actual_view(self) -> None:
            self._apply_pair_view(mode='actual')
            if self.compare_window is not None:
                self.compare_window.apply_actual()

        def _on_preview_view_change(self, side: str, _kind: str, zoom: float, cx_ratio: float, cy_ratio: float) -> None:
            if self._syncing_view or not self.compare_sync_zoom_var.get():
                return
            other = self.right_pane if side == 'A' else self.left_pane
            if other is None or other.source_image is None:
                return
            self._syncing_view = True
            try:
                other.sync_from_other(zoom, cx_ratio, cy_ratio)
            finally:
                self._syncing_view = False

        def _show_pair(self, index: int, reset_view: bool = True) -> None:
            if not self.current_results:
                self._show_empty_state('暂无结果。')
                return
            index = max(0, min(index, len(self.current_results) - 1))
            self.current_index = index
            row = self.current_results[index]
            if self.result_tree.exists(str(index)):
                current_sel = self.result_tree.selection()
                if current_sel != (str(index),):
                    self._suspend_tree_event = True
                    try:
                        self.result_tree.selection_set(str(index))
                        self.result_tree.focus(str(index))
                        self.result_tree.see(str(index))
                    finally:
                        self._suspend_tree_event = False
            deleted_side = row.get('deleted_side', '')
            state_suffix = f' · 已删除 {deleted_side}' if deleted_side else ''
            self.pair_title_var.set(f'第 {index + 1} / {len(self.current_results)} 组{state_suffix}')
            self.score_var.set(f"相似度 {row['similarity_score']:.4f}")
            self.type_var.set(f"类型 {row['match_type']}")
            self.center_hint_var.set(
                f"A：{human_size(row['file_size_1'])} · {row['resolution_1']}\n"
                f"B：{human_size(row['file_size_2'])} · {row['resolution_2']}"
            )
            recommendation = self._recommend_deletion(row)
            self._last_recommendation = recommendation
            reason_text = '\n'.join('• ' + x for x in recommendation.get('reason_lines', [])[:3])
            self.recommend_var.set(recommendation['text'] + (f'\n{reason_text}' if reason_text else ''))
            self.left_info_var.set(self._format_side_info(row, 'A'))
            self.right_info_var.set(self._format_side_info(row, 'B'))
            self._load_preview_source('A', row['path_1'], reset_view=reset_view)
            self._load_preview_source('B', row['path_2'], reset_view=reset_view)
            if reset_view:
                self._apply_fit_view()
            self.prev_button.configure(state='normal' if index > 0 else 'disabled')
            self.next_button.configure(state='normal' if index < len(self.current_results) - 1 else 'disabled')
            self.keep_button.configure(state='normal')
            a_exists = Path(row['path_1']).exists()
            b_exists = Path(row['path_2']).exists()
            self.delete_a_button.configure(state='normal' if (not deleted_side and a_exists) else 'disabled')
            self.delete_b_button.configure(state='normal' if (not deleted_side and b_exists) else 'disabled')
            self.open_a_button.configure(state='normal' if a_exists else 'disabled')
            self.open_b_button.configure(state='normal' if b_exists else 'disabled')
            self.fit_view_button.configure(state='normal' if a_exists or b_exists else 'disabled')
            self.actual_view_button.configure(state='normal' if a_exists or b_exists else 'disabled')
            rec_side = recommendation.get('side')
            rec_enabled = bool(rec_side) and ((rec_side == 'A' and a_exists) or (rec_side == 'B' and b_exists)) and not deleted_side
            self.recommend_button.configure(state='normal' if rec_enabled else 'disabled')

        def _on_tree_select(self, _event=None) -> None:
            if self._suspend_tree_event:
                return
            sel = self.result_tree.selection()
            if sel:
                self._show_pair(int(sel[0]), reset_view=True)

        def _keep_and_next(self) -> None:
            if self.current_index < 0:
                return
            if self.current_index < len(self.current_results) - 1:
                self._show_pair(self.current_index + 1, reset_view=True)

        def _apply_recommended_delete(self) -> None:
            side = self._last_recommendation.get('side')
            if side in {'A', 'B'}:
                self._delete_current(side)

        def _open_current_side(self, side: str) -> None:
            if self.current_index < 0 or self.current_index >= len(self.current_results):
                return
            row = self.current_results[self.current_index]
            path = row["path_1"] if side == "A" else row["path_2"]
            try:
                open_file_location(path)
            except Exception as e:
                messagebox.showerror("打开失败", str(e))

        def _delete_current(self, side: str) -> None:
            if self.current_index < 0 or self.current_index >= len(self.current_results):
                return
            row = self.current_results[self.current_index]
            if row.get("deleted_side"):
                return
            target_path = row["path_1"] if side == "A" else row["path_2"]
            target_name = row["name_1"] if side == "A" else row["name_2"]
            prompt = (
                f"请确认删除 {side} 侧文件。\n\n"
                f"文件名：{target_name}\n"
                f"完整路径：{target_path}\n\n"
                "系统会优先移入回收站。"
            )
            if not messagebox.askyesno("确认删除", prompt, icon="warning"):
                return
            try:
                action = move_to_trash(target_path)
            except Exception as e:
                messagebox.showerror("删除失败", str(e))
                return
            row["deleted_side"] = side
            self.result_tree.set(str(self.current_index), "status", f"已删 {side}")
            self._append_activity("compare", f"{action}：{target_path}")
            self.status_var.set(f"{action}：{target_name}")
            self._show_pair(self.current_index, reset_view=False)
            if self.current_index < len(self.current_results) - 1:
                self.root.after(220, lambda: self._show_pair(self.current_index + 1, reset_view=True))

        def _export_compare_report(self) -> None:
            if not self.current_results or self.compare_meta is None:
                messagebox.showinfo("没有结果", "当前没有可导出的匹配结果。")
                return
            default_name = f"ab_similarity_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=default_name, filetypes=[("TXT", "*.txt")])
            if not path:
                return
            try:
                write_cross_compare_txt_report(
                    output_txt=Path(path),
                    dir_a=Path(self.compare_meta["dir_a"]),
                    dir_b=Path(self.compare_meta["dir_b"]),
                    rows=self.current_results,
                    errors_a=self.compare_meta["errors_a"],
                    errors_b=self.compare_meta["errors_b"],
                    min_sim=self.compare_meta["min_sim"],
                    max_sim=self.compare_meta["max_sim"],
                )
            except Exception as e:
                messagebox.showerror("导出失败", str(e))
                return
            self._append_activity("compare", f"已导出复核报告：{path}")
            messagebox.showinfo("导出完成", f"复核报告已导出：\n\n{path}")

    if ttkb is not None:
        root = ttkb.Window(themename="litera")
    else:
        root = tk.Tk()
    StudioApp(root, base_args)
    root.mainloop()


def main() -> None:
    from desktop_ui import launch_app

    launch_app()


if __name__ == "__main__":
    main()
