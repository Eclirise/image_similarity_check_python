from __future__ import annotations

import argparse
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from image_similarity_check_python import (
    APP_NAME,
    default_model_cache_dir,
    guess_device,
    hardware_summary,
    human_size,
    match_type_label,
    move_to_trash,
    open_file_location,
    resolve_clip_pretrained,
    run_ab_compare_job,
    run_reference_job,
    run_scan_job,
    write_cross_compare_txt_report,
)

BG = "#F3F0EA"
SURFACE = "#FFFFFF"
SURFACE_SOFT = "#EEE8DF"
BORDER = "#DED7CD"
TEXT = "#191714"
MUTED = "#6F665C"
ACCENT = "#23584D"
ACCENT_HOVER = "#1D4C42"
ACCENT_SOFT = "#E4F0EC"
FONT_UI = "Microsoft YaHei UI"
MODEL_FILE_NAMES = ["open_clip_model.safetensors", "open_clip_pytorch_model.bin"]


def _font(size: int, weight: str = "normal") -> tuple[str, int, str]:
    return (FONT_UI, size, weight)


class FlatButton(tk.Button):
    def __init__(self, master, *, primary: bool = False, quiet: bool = False, **kwargs):
        bg = ACCENT if primary else SURFACE
        fg = "#FFFFFF" if primary else TEXT
        active_bg = ACCENT_HOVER if primary else SURFACE_SOFT
        border = ACCENT if primary else BORDER
        if quiet:
            bg = BG
            fg = MUTED
            active_bg = SURFACE_SOFT
            border = BG
        super().__init__(
            master,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            cursor="hand2",
            font=_font(12, "bold" if primary else "normal"),
            highlightthickness=1,
            highlightbackground=border,
            highlightcolor=border,
            disabledforeground="#A8A095",
            **kwargs,
        )
        self.default_bg = bg
        self.hover_bg = active_bg
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, _event):
        if self["state"] != "disabled":
            self.configure(bg=self.hover_bg)

    def _on_leave(self, _event):
        self.configure(bg=self.default_bg)


class InputField(tk.Frame):
    def __init__(self, master, label: str, *, browse_kind: str | None = None):
        super().__init__(master, bg=SURFACE)
        self.browse_kind = browse_kind
        self.value = tk.StringVar()
        tk.Label(self, text=label, bg=SURFACE, fg=MUTED, font=_font(11)).pack(anchor="w")
        row = tk.Frame(self, bg=SURFACE)
        row.pack(fill="x", pady=(6, 0))
        self.entry = tk.Entry(
            row,
            textvariable=self.value,
            font=_font(13),
            relief="flat",
            bd=0,
            bg="#FBFAF7",
            fg=TEXT,
            insertbackground=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        self.entry.pack(side="left", fill="x", expand=True, ipady=10)
        if browse_kind is not None:
            FlatButton(row, text="选择", command=self._browse).pack(side="left", padx=(10, 0))

    def _browse(self) -> None:
        if self.browse_kind == "dir":
            path = filedialog.askdirectory()
        elif self.browse_kind == "image":
            path = filedialog.askopenfilename(filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.gif")])
        elif self.browse_kind == "xlsx":
            path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        elif self.browse_kind == "txt":
            path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("文本", "*.txt")])
        else:
            path = ""
        if path:
            self.value.set(path)

    def get(self) -> str:
        return self.value.get().strip()


class ZoomPreviewPane(tk.Frame):
    def __init__(self, master, title: str, *, open_callback=None):
        super().__init__(master, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        self.open_callback = open_callback
        self.source_path = ""
        self.source_image: Image.Image | None = None
        self.tk_image = None
        self.zoom = 1.0
        self.fit_zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.drag_start: tuple[int, int] | None = None
        self.auto_fit = True
        self.title_var = tk.StringVar(value=title)
        self.caption_var = tk.StringVar(value="等待结果")

        header = tk.Frame(self, bg=SURFACE)
        header.pack(fill="x", padx=18, pady=(14, 8))
        tk.Label(header, textvariable=self.title_var, bg=SURFACE, fg=TEXT, font=_font(12, "bold")).pack(side="left")
        tk.Label(header, textvariable=self.caption_var, bg=SURFACE, fg=MUTED, font=_font(10)).pack(side="right")

        self.canvas = tk.Canvas(self, bg="#FBFAF7", bd=0, highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.canvas.bind("<Configure>", lambda _event: self._render())
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.clear()

    def clear(self, text: str = "等待结果") -> None:
        self.source_path = ""
        self.source_image = None
        self.zoom = 1.0
        self.fit_zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.auto_fit = True
        self.caption_var.set(text)
        self._render()

    def set_image(self, path: str) -> None:
        self.source_path = path
        try:
            with Image.open(path) as image:
                image.load()
                self.source_image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception:
            self.source_image = None
            self.caption_var.set("无法预览")
            self._render()
            return
        self.fit_view()

    def fit_view(self) -> None:
        if self.source_image is None:
            return
        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        zoom_x = max((canvas_w - 24) / max(self.source_image.width, 1), 0.05)
        zoom_y = max((canvas_h - 24) / max(self.source_image.height, 1), 0.05)
        self.fit_zoom = min(zoom_x, zoom_y, 1.0)
        self.zoom = self.fit_zoom
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.auto_fit = True
        self.caption_var.set(f"{self.source_image.width}×{self.source_image.height}")
        self._render()

    def actual_size(self) -> None:
        if self.source_image is None:
            return
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.auto_fit = False
        self._render()

    def _on_wheel(self, event) -> None:
        if self.source_image is None:
            return
        factor = 1.12 if event.delta > 0 else 1 / 1.12
        self.zoom = min(max(self.zoom * factor, self.fit_zoom * 0.7), 8.0)
        self.auto_fit = False
        self._render()

    def _on_press(self, event) -> None:
        self.drag_start = (event.x, event.y)

    def _on_drag(self, event) -> None:
        if self.source_image is None or self.drag_start is None:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.offset_x += dx
        self.offset_y += dy
        self.auto_fit = False
        self.drag_start = (event.x, event.y)
        self._render()

    def _on_double_click(self, _event) -> None:
        if self.open_callback is not None:
            self.open_callback()

    def _render(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        if self.source_image is None:
            self.canvas.create_text(width // 2, height // 2, text="等待结果", fill=MUTED, font=_font(12))
            return
        if self.auto_fit:
            zoom_x = max((width - 24) / max(self.source_image.width, 1), 0.05)
            zoom_y = max((height - 24) / max(self.source_image.height, 1), 0.05)
            self.fit_zoom = min(zoom_x, zoom_y, 1.0)
            self.zoom = self.fit_zoom
            self.offset_x = 0.0
            self.offset_y = 0.0
        render_w = max(int(self.source_image.width * self.zoom), 1)
        render_h = max(int(self.source_image.height * self.zoom), 1)
        max_offset_x = max((render_w - width) / 2, 0)
        max_offset_y = max((render_h - height) / 2, 0)
        self.offset_x = min(max(self.offset_x, -max_offset_x), max_offset_x)
        self.offset_y = min(max(self.offset_y, -max_offset_y), max_offset_y)
        image = self.source_image.resize((render_w, render_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(width / 2 + self.offset_x, height / 2 + self.offset_y, image=self.tk_image)


class LargeCompareWindow:
    def __init__(self, master: tk.Tk, app: "ModernStudioApp") -> None:
        self.app = app
        self.win = tk.Toplevel(master)
        self.win.title("双图复核")
        self.win.configure(bg=BG)
        self.win.geometry("1740x1020")
        self.win.minsize(1360, 840)
        self.win.protocol("WM_DELETE_WINDOW", self._close)

        self.group_var = tk.StringVar(value="暂无结果")
        self.score_var = tk.StringVar(value="相似度 --")
        self.recommend_var = tk.StringVar(value="等待结果")
        self.left_info_var = tk.StringVar(value="等待结果")
        self.right_info_var = tk.StringVar(value="等待结果")

        wrap = tk.Frame(self.win, bg=BG)
        wrap.pack(fill="both", expand=True, padx=22, pady=20)

        top = tk.Frame(wrap, bg=BG)
        top.pack(fill="x", pady=(0, 14))
        title_box = tk.Frame(top, bg=BG)
        title_box.pack(side="left", fill="x", expand=True)
        tk.Label(title_box, text="双图复核", bg=BG, fg=TEXT, font=_font(24, "bold")).pack(anchor="w")
        tk.Label(title_box, textvariable=self.group_var, bg=BG, fg=MUTED, font=_font(13)).pack(anchor="w", pady=(6, 0))
        tk.Label(title_box, textvariable=self.score_var, bg=BG, fg=TEXT, font=_font(14, "bold")).pack(anchor="w", pady=(8, 0))
        tk.Label(title_box, textvariable=self.recommend_var, bg=BG, fg=ACCENT, font=_font(12, "bold"), justify="left", wraplength=760).pack(anchor="w", pady=(8, 0))

        toolbar = tk.Frame(wrap, bg=BG)
        toolbar.pack(fill="x", pady=(0, 14))
        FlatButton(toolbar, text="上一组", quiet=True, command=self._prev).pack(side="left")
        FlatButton(toolbar, text="下一组", quiet=True, command=self._next).pack(side="left", padx=(10, 0))
        FlatButton(toolbar, text="保留并下一组", primary=True, command=self._keep_and_next).pack(side="left", padx=(18, 0))
        FlatButton(toolbar, text="删除 A", command=lambda: self._delete("A")).pack(side="left", padx=(18, 0))
        FlatButton(toolbar, text="删除 B", command=lambda: self._delete("B")).pack(side="left", padx=(10, 0))
        FlatButton(toolbar, text="适应窗口", quiet=True, command=self._fit_all).pack(side="left", padx=(18, 0))
        FlatButton(toolbar, text="原始大小", quiet=True, command=self._actual_all).pack(side="left", padx=(10, 0))
        FlatButton(toolbar, text="打开 A 位置", quiet=True, command=lambda: self.app._open_compare_side("A")).pack(side="left", padx=(18, 0))
        FlatButton(toolbar, text="打开 B 位置", quiet=True, command=lambda: self.app._open_compare_side("B")).pack(side="left", padx=(10, 0))

        body = tk.Frame(wrap, bg=BG)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=5)
        body.grid_rowconfigure(1, weight=2)

        self.left_pane = ZoomPreviewPane(body, "A 图")
        self.left_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 12))
        self.right_pane = ZoomPreviewPane(body, "B 图")
        self.right_pane.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=(0, 12))

        left_meta = tk.Frame(body, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        left_meta.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        tk.Label(left_meta, text="A 图信息", bg=SURFACE, fg=TEXT, font=_font(12, "bold")).pack(anchor="w", padx=16, pady=(14, 6))
        tk.Label(left_meta, textvariable=self.left_info_var, bg=SURFACE, fg=MUTED, font=_font(11), justify="left", anchor="nw", wraplength=760).pack(fill="both", expand=True, padx=16, pady=(0, 16))

        right_meta = tk.Frame(body, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        right_meta.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        tk.Label(right_meta, text="B 图信息", bg=SURFACE, fg=TEXT, font=_font(12, "bold")).pack(anchor="w", padx=16, pady=(14, 6))
        tk.Label(right_meta, textvariable=self.right_info_var, bg=SURFACE, fg=MUTED, font=_font(11), justify="left", anchor="nw", wraplength=760).pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.refresh()

    def _close(self) -> None:
        self.app.large_compare_window = None
        self.win.destroy()

    def _current_row(self) -> dict | None:
        return self.app._current_compare_row()

    def refresh(self) -> None:
        row = self._current_row()
        total = len(self.app.compare_rows)
        if row is None or total == 0:
            self.win.title("双图复核")
            self.group_var.set("暂无结果")
            self.score_var.set("相似度 --")
            self.recommend_var.set("等待结果")
            self.left_info_var.set("等待结果")
            self.right_info_var.set("等待结果")
            self.left_pane.clear("等待结果")
            self.right_pane.clear("等待结果")
            return
        index = max(0, self.app.compare_selected_index)
        self.win.title(f"双图复核 - 第 {index + 1} / {total} 组")
        self.group_var.set(f"第 {index + 1} / {total} 组")
        deleted_side = row.get("deleted_side", "")
        suffix = f" · 已删除 {deleted_side}" if deleted_side else ""
        self.score_var.set(f"相似度 {row['similarity_score']:.4f} · {match_type_label(row.get('match_type', ''))}{suffix}")
        self.recommend_var.set(self.app._recommend_delete(row)["text"])
        self.left_info_var.set(self.app._format_side_info(row, "A"))
        self.right_info_var.set(self.app._format_side_info(row, "B"))
        if Path(row["path_1"]).exists():
            self.left_pane.set_image(row["path_1"])
        else:
            self.left_pane.clear("A 图已删除")
        if Path(row["path_2"]).exists():
            self.right_pane.set_image(row["path_2"])
        else:
            self.right_pane.clear("B 图已删除")

    def _prev(self) -> None:
        self.app._prev_compare()

    def _next(self) -> None:
        self.app._next_compare()

    def _keep_and_next(self) -> None:
        self.app._keep_and_next()

    def _delete(self, side: str) -> None:
        self.app._delete_compare_side(side)

    def _fit_all(self) -> None:
        self.left_pane.fit_view()
        self.right_pane.fit_view()

    def _actual_all(self) -> None:
        self.left_pane.actual_size()
        self.right_pane.actual_size()


class ModernStudioApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.configure(bg=BG)
        self.root.geometry("1540x980")
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.job_running = False
        self.job_done_handler = None
        self.current_mode = "scan"
        self.log_lines: list[str] = []
        self.log_window: tk.Toplevel | None = None
        self.log_text: tk.Text | None = None
        self.compare_rows: list[dict] = []
        self.compare_meta: dict | None = None
        self.compare_selected_index = -1
        self.large_compare_window: LargeCompareWindow | None = None

        self.status_var = tk.StringVar(value="准备就绪")
        self.progress_var = tk.StringVar(value="等待操作")
        self.header_hint_var = tk.StringVar(value=hardware_summary(guess_device("auto")))
        self.model_var = tk.StringVar(value=self._model_summary())
        self.scan_result_var = tk.StringVar(value="请选择目录后开始。")
        self.reference_result_var = tk.StringVar(value="请选择参考图与目录后开始。")
        self.compare_result_var = tk.StringVar(value="请选择目录 A 和目录 B。")
        self.compare_recommend_var = tk.StringVar(value="系统会在这里给出删图建议。")
        self.compare_left_var = tk.StringVar(value="等待结果")
        self.compare_right_var = tk.StringVar(value="等待结果")
        self.compare_count_var = tk.StringVar(value="暂无结果")

        self.page_frames: dict[str, tk.Frame] = {}
        self.mode_buttons: dict[str, FlatButton] = {}
        self.action_buttons: list[tk.Button] = []

        self._setup_styles()
        self._build_layout()
        self._show_mode("scan")
        self._bind_shortcuts()
        self._poll_queue()

    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Studio.Horizontal.TProgressbar", troughcolor=SURFACE_SOFT, background=ACCENT, borderwidth=0, thickness=10)
        style.configure("Studio.Treeview", background=SURFACE, fieldbackground=SURFACE, foreground=TEXT, rowheight=34, font=_font(11), borderwidth=0)
        style.configure("Studio.Treeview.Heading", background=SURFACE_SOFT, foreground=TEXT, font=_font(11, "bold"), relief="flat", borderwidth=0)
        style.map("Studio.Treeview", background=[("selected", ACCENT_SOFT)], foreground=[("selected", TEXT)])

    def _build_layout(self) -> None:
        shell = tk.Frame(self.root, bg=BG)
        shell.pack(fill="both", expand=True, padx=28, pady=22)
        self._build_header(shell)
        self.body = tk.Frame(shell, bg=BG)
        self.body.pack(fill="both", expand=True)
        self._build_mode_bar(shell)
        self.page_frames["scan"] = self._build_scan_page()
        self.page_frames["reference"] = self._build_reference_page()
        self.page_frames["compare"] = self._build_compare_page()

    def _build_header(self, parent: tk.Widget) -> None:
        header = tk.Frame(parent, bg=BG)
        header.pack(fill="x")
        left = tk.Frame(header, bg=BG)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text=APP_NAME, bg=BG, fg=TEXT, font=_font(26, "bold")).pack(anchor="w")
        tk.Label(left, text="扫描、检索与复核", bg=BG, fg=MUTED, font=_font(12)).pack(anchor="w", pady=(4, 0))
        right = tk.Frame(header, bg=BG)
        right.pack(side="right", anchor="ne")
        self._pill(right, self.model_var).pack(anchor="e")
        self._pill(right, self.header_hint_var).pack(anchor="e", pady=(10, 0))
        FlatButton(right, text="查看过程", quiet=True, command=self._open_log_window).pack(anchor="e", pady=(10, 0))

        outer, card = self._card(parent)
        outer.pack(fill="x", pady=(20, 18))
        top = tk.Frame(card, bg=SURFACE)
        top.pack(fill="x")
        tk.Label(top, textvariable=self.status_var, bg=SURFACE, fg=TEXT, font=_font(14, "bold")).pack(side="left")
        tk.Label(top, textvariable=self.progress_var, bg=SURFACE, fg=MUTED, font=_font(11)).pack(side="right")
        self.progressbar = ttk.Progressbar(card, style="Studio.Horizontal.TProgressbar", mode="determinate", maximum=100)
        self.progressbar.pack(fill="x", pady=(14, 0))

    def _build_mode_bar(self, parent: tk.Widget) -> None:
        bar = tk.Frame(parent, bg=BG)
        bar.pack(fill="x", pady=(0, 18))
        self.mode_buttons["scan"] = FlatButton(bar, text="目录扫描", command=lambda: self._show_mode("scan"))
        self.mode_buttons["reference"] = FlatButton(bar, text="参考图检索", command=lambda: self._show_mode("reference"))
        self.mode_buttons["compare"] = FlatButton(bar, text="双目录复核", command=lambda: self._show_mode("compare"))
        for button in self.mode_buttons.values():
            button.pack(side="left", padx=(0, 12))

    def _show_mode(self, mode: str) -> None:
        self.current_mode = mode
        for name, frame in self.page_frames.items():
            if frame.winfo_manager():
                frame.pack_forget()
            if name == mode:
                frame.pack(fill="both", expand=True)
        for name, button in self.mode_buttons.items():
            if name == mode:
                button.configure(bg=ACCENT, fg="#FFFFFF", highlightbackground=ACCENT)
            else:
                button.configure(bg=SURFACE, fg=TEXT, highlightbackground=BORDER)

    def _base_page(self, title: str, subtitle: str) -> tuple[tk.Frame, tk.Frame]:
        page = tk.Frame(self.body, bg=BG)
        head = tk.Frame(page, bg=BG)
        head.pack(fill="x", pady=(0, 14))
        tk.Label(head, text=title, bg=BG, fg=TEXT, font=_font(22, "bold")).pack(anchor="w")
        tk.Label(head, text=subtitle, bg=BG, fg=MUTED, font=_font(12)).pack(anchor="w", pady=(4, 0))
        content = tk.Frame(page, bg=BG)
        content.pack(fill="both", expand=True)
        return page, content

    def _card(self, parent: tk.Widget) -> tuple[tk.Frame, tk.Frame]:
        outer = tk.Frame(parent, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        inner = tk.Frame(outer, bg=SURFACE)
        inner.pack(fill="both", expand=True, padx=24, pady=22)
        return outer, inner

    def _pill(self, parent: tk.Widget, variable: tk.StringVar) -> tk.Frame:
        wrap = tk.Frame(parent, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
        tk.Label(wrap, textvariable=variable, bg=SURFACE, fg=MUTED, font=_font(10), justify="left").pack(fill="x", padx=14, pady=10)
        return wrap

    def _advanced_frame(self, parent: tk.Widget) -> tk.Frame:
        frame = tk.Frame(parent, bg=SURFACE)
        frame.pack(fill="x", pady=(14, 0))
        frame.pack_forget()
        return frame

    def _toggle_frame(self, frame: tk.Frame) -> None:
        if frame.winfo_manager():
            frame.pack_forget()
        else:
            frame.pack(fill="x", pady=(14, 0))

    def _range_inputs(self, parent: tk.Widget, min_default: str, max_default: str) -> tuple[tk.StringVar, tk.StringVar]:
        tk.Label(parent, text="相似度范围", bg=SURFACE, fg=MUTED, font=_font(11)).pack(anchor="w")
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x", pady=(6, 0))
        min_var = tk.StringVar(value=min_default)
        max_var = tk.StringVar(value=max_default)
        self._labeled_entry(row, "最低", min_var).pack(side="left", fill="x", expand=True)
        self._labeled_entry(row, "最高", max_var).pack(side="left", fill="x", expand=True, padx=(12, 0))
        return min_var, max_var

    def _single_small_field(self, parent: tk.Widget, label: str, default: str) -> tk.StringVar:
        tk.Label(parent, text=label, bg=SURFACE, fg=MUTED, font=_font(11)).pack(anchor="w", pady=(14, 0))
        var = tk.StringVar(value=default)
        self._labeled_entry(parent, "", var, compact=True).pack(fill="x", pady=(6, 0))
        return var

    def _labeled_entry(self, parent: tk.Widget, label: str, variable: tk.StringVar, compact: bool = False) -> tk.Frame:
        wrap = tk.Frame(parent, bg=SURFACE)
        if label:
            tk.Label(wrap, text=label, bg=SURFACE, fg=MUTED, font=_font(10)).pack(anchor="w")
        entry = tk.Entry(
            wrap,
            textvariable=variable,
            font=_font(12),
            relief="flat",
            bd=0,
            bg="#FBFAF7",
            fg=TEXT,
            insertbackground=TEXT,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        entry.pack(fill="x", ipady=8, pady=(6 if label else 0, 0))
        if compact:
            wrap.configure(width=160)
        return wrap

    def _check_row(self, parent: tk.Widget, variable: tk.BooleanVar, text: str) -> tk.Checkbutton:
        return tk.Checkbutton(parent, text=text, variable=variable, bg=SURFACE, fg=TEXT, activebackground=SURFACE, activeforeground=TEXT, selectcolor=SURFACE, font=_font(11), highlightthickness=0, bd=0)

    def _append_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_lines.append(f"[{stamp}] {message}")
        self.log_lines = self.log_lines[-400:]
        if self.log_text is not None and self.log_window is not None and self.log_window.winfo_exists():
            self.log_text.delete("1.0", "end")
            self.log_text.insert("1.0", "\n".join(self.log_lines))
            self.log_text.see("end")

    def _open_log_window(self) -> None:
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.lift()
            return
        self.log_window = tk.Toplevel(self.root)
        self.log_window.title("详细过程")
        self.log_window.configure(bg=BG)
        self.log_window.geometry("980x560")
        self.log_text = tk.Text(self.log_window, wrap="word", bg=SURFACE, fg=TEXT, font=_font(11), relief="flat", bd=0, padx=18, pady=18)
        self.log_text.pack(fill="both", expand=True, padx=18, pady=18)
        self.log_text.insert("1.0", "\n".join(self.log_lines))

    def _model_summary(self) -> str:
        resolved = resolve_clip_pretrained("laion2b_s34b_b79k")
        model_path = Path(resolved)
        if model_path.exists():
            return f"模型：{model_path.name}"
        return f"模型：未发现本地权重，可放入根目录的 {MODEL_FILE_NAMES[0]}"

    def _task_args(self, *, high_precision: bool, **extra) -> argparse.Namespace:
        data = dict(
            mode="gui",
            device="auto",
            clip_model="ViT-B-32",
            clip_pretrained="laion2b_s34b_b79k",
            clip_mirror="auto",
            clip_endpoint="",
            model_cache_dir=str(default_model_cache_dir()),
            no_openclip=not high_precision,
            workers=None,
            batch_size=None,
            extensions="jpg,jpeg,png,bmp,webp,tif,tiff,gif",
            non_recursive=False,
            tab=self.current_mode,
        )
        data.update(extra)
        return argparse.Namespace(**data)

    def _read_float_value(self, raw: str, field_name: str, *, minimum: float, maximum: float) -> float | None:
        try:
            value = float(raw)
        except ValueError:
            messagebox.showwarning("输入有误", f"{field_name} 必须是数字。")
            return None
        if not (minimum <= value <= maximum):
            messagebox.showwarning("输入有误", f"{field_name} 必须在 {minimum} 到 {maximum} 之间。")
            return None
        return value

    def _read_int_value(self, raw: str, field_name: str, *, minimum: int, maximum: int) -> int | None:
        try:
            value = int(raw)
        except ValueError:
            messagebox.showwarning("输入有误", f"{field_name} 必须是整数。")
            return None
        if not (minimum <= value <= maximum):
            messagebox.showwarning("输入有误", f"{field_name} 必须在 {minimum} 到 {maximum} 之间。")
            return None
        return value

    def _set_busy(self, busy: bool) -> None:
        self.job_running = busy
        for button in self.action_buttons:
            button.configure(state="disabled" if busy else "normal")

    def _start_worker(self, worker, done_handler) -> None:
        if self.job_running:
            messagebox.showinfo("请稍候", "当前任务尚未结束。")
            return
        self._set_busy(True)
        self.job_done_handler = done_handler
        self.status_var.set("任务进行中")
        self.progress_var.set("正在准备")
        self.progressbar.configure(mode="indeterminate", value=0)
        self.progressbar.start(10)

        def status_cb(message: str) -> None:
            self.queue.put(("status", message))

        def progress_cb(current: int, total: int, stage: str) -> None:
            self.queue.put(("progress", (current, total, stage)))

        def runner() -> None:
            try:
                result = worker(status_cb, progress_cb)
            except Exception as exc:
                self.queue.put(("error", str(exc)))
            else:
                self.queue.put(("done", result))

        threading.Thread(target=runner, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "status":
                    message = str(payload)
                    parts = message.splitlines()
                    self.status_var.set(parts[0][:120])
                    self.progress_var.set(parts[-1][:160])
                    self._append_log(message)
                elif kind == "progress":
                    current, total, stage = payload
                    if total and total > 1:
                        self.progressbar.stop()
                        self.progressbar.configure(mode="determinate", maximum=max(total, 1), value=min(current, total))
                        self.status_var.set(stage)
                        self.progress_var.set(f"{current}/{total}")
                    else:
                        self.progressbar.configure(mode="indeterminate")
                        self.progressbar.start(10)
                        self.status_var.set(stage)
                    self._append_log(f"{stage} {current}/{total}")
                elif kind == "error":
                    self._set_busy(False)
                    self.progressbar.stop()
                    self.progressbar.configure(mode="determinate", maximum=100, value=0)
                    self.status_var.set("任务失败")
                    self.progress_var.set("请查看详细过程")
                    self._append_log(str(payload))
                    messagebox.showerror("任务失败", str(payload))
                elif kind == "done":
                    self._set_busy(False)
                    self.progressbar.stop()
                    self.progressbar.configure(mode="determinate", maximum=100, value=100)
                    self.status_var.set("已完成")
                    self.progress_var.set("可以继续下一步")
                    self._append_log("任务完成")
                    if self.job_done_handler is not None:
                        self.job_done_handler(payload)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _build_scan_page(self) -> tk.Frame:
        page, content = self._base_page("目录扫描", "扫描目录中的相似图片，导出 Excel。")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        outer, card = self._card(content)
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.scan_dir = InputField(card, "图片目录", browse_kind="dir")
        self.scan_dir.pack(fill="x")
        FlatButton(card, text="更多设置", quiet=True, command=lambda: self._toggle_frame(self.scan_advanced)).pack(anchor="w", pady=(16, 0))
        self.scan_advanced = self._advanced_frame(card)
        self.scan_output = InputField(self.scan_advanced, "Excel 输出", browse_kind="xlsx")
        self.scan_output.pack(fill="x")
        self.scan_min, self.scan_max = self._range_inputs(self.scan_advanced, "0.90", "1.00")
        self.scan_topk = self._single_small_field(self.scan_advanced, "每张图片保留", "20")
        self.scan_recursive = tk.BooleanVar(value=True)
        self.scan_high_precision = tk.BooleanVar(value=True)
        self._check_row(self.scan_advanced, self.scan_recursive, "扫描子目录").pack(anchor="w", pady=(14, 0))
        self._check_row(self.scan_advanced, self.scan_high_precision, "使用高精度模型").pack(anchor="w", pady=(8, 0))
        button = FlatButton(card, text="开始扫描", primary=True, command=self._start_scan)
        button.pack(anchor="w", pady=(22, 0))
        self.action_buttons.append(button)

        outer, result = self._card(content)
        outer.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        tk.Label(result, text="当前状态", bg=SURFACE, fg=MUTED, font=_font(11)).pack(anchor="w")
        tk.Label(result, textvariable=self.scan_result_var, bg=SURFACE, fg=TEXT, font=_font(14), justify="left", wraplength=520).pack(anchor="w", pady=(8, 0))
        tk.Label(result, text="默认优先读取根目录本地模型，未找到时再在线获取。", bg=SURFACE, fg=MUTED, font=_font(11), justify="left", wraplength=520).pack(anchor="w", pady=(18, 0))
        return page

    def _build_reference_page(self) -> tk.Frame:
        page, content = self._base_page("参考图检索", "用一张参考图在目录中查找相似图片，导出 TXT。")
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        outer, card = self._card(content)
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.reference_file = InputField(card, "参考图片", browse_kind="image")
        self.reference_file.pack(fill="x")
        self.reference_dir = InputField(card, "搜索目录", browse_kind="dir")
        self.reference_dir.pack(fill="x", pady=(16, 0))
        FlatButton(card, text="更多设置", quiet=True, command=lambda: self._toggle_frame(self.reference_advanced)).pack(anchor="w", pady=(16, 0))
        self.reference_advanced = self._advanced_frame(card)
        self.reference_output = InputField(self.reference_advanced, "TXT 输出", browse_kind="txt")
        self.reference_output.pack(fill="x")
        self.reference_min, self.reference_max = self._range_inputs(self.reference_advanced, "0.90", "1.00")
        self.reference_recursive = tk.BooleanVar(value=True)
        self.reference_high_precision = tk.BooleanVar(value=True)
        self._check_row(self.reference_advanced, self.reference_recursive, "扫描子目录").pack(anchor="w", pady=(14, 0))
        self._check_row(self.reference_advanced, self.reference_high_precision, "使用高精度模型").pack(anchor="w", pady=(8, 0))
        button = FlatButton(card, text="开始检索", primary=True, command=self._start_reference)
        button.pack(anchor="w", pady=(22, 0))
        self.action_buttons.append(button)

        outer, result = self._card(content)
        outer.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        tk.Label(result, text="当前状态", bg=SURFACE, fg=MUTED, font=_font(11)).pack(anchor="w")
        tk.Label(result, textvariable=self.reference_result_var, bg=SURFACE, fg=TEXT, font=_font(14), justify="left", wraplength=520).pack(anchor="w", pady=(8, 0))
        tk.Label(result, text="参考图会自动排除自身，再与目录中的图片逐一比较。", bg=SURFACE, fg=MUTED, font=_font(11), justify="left", wraplength=520).pack(anchor="w", pady=(18, 0))
        return page

    def _build_compare_page(self) -> tk.Frame:
        page, content = self._base_page("双目录复核", "查看两组目录中的候选相似图，快速保留或删除。")
        content.grid_rowconfigure(0, weight=0)
        content.grid_rowconfigure(1, weight=4)
        content.grid_columnconfigure(0, weight=5)
        content.grid_columnconfigure(1, weight=7)

        outer, top_card = self._card(content)
        outer.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        paths = tk.Frame(top_card, bg=SURFACE)
        paths.pack(fill="x")
        paths.grid_columnconfigure(0, weight=1)
        paths.grid_columnconfigure(1, weight=1)
        self.compare_dir_a = InputField(paths, "目录 A", browse_kind="dir")
        self.compare_dir_a.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.compare_dir_b = InputField(paths, "目录 B", browse_kind="dir")
        self.compare_dir_b.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        FlatButton(top_card, text="更多设置", quiet=True, command=lambda: self._toggle_frame(self.compare_advanced)).pack(anchor="w", pady=(12, 0))
        self.compare_advanced = self._advanced_frame(top_card)
        self.compare_min, self.compare_max = self._range_inputs(self.compare_advanced, "0.92", "1.00")
        self.compare_topk = self._single_small_field(self.compare_advanced, "每张 A 保留", "3")
        self.compare_recursive = tk.BooleanVar(value=True)
        self.compare_high_precision = tk.BooleanVar(value=True)
        self._check_row(self.compare_advanced, self.compare_recursive, "扫描子目录").pack(anchor="w", pady=(14, 0))
        self._check_row(self.compare_advanced, self.compare_high_precision, "使用高精度模型").pack(anchor="w", pady=(8, 0))
        actions = tk.Frame(top_card, bg=SURFACE)
        actions.pack(fill="x", pady=(16, 0))
        start_button = FlatButton(actions, text="开始复核", primary=True, command=self._start_compare)
        start_button.pack(side="left")
        self.compare_export_button = FlatButton(actions, text="导出报告", command=self._export_compare_report, state="disabled")
        self.compare_export_button.pack(side="left", padx=(12, 0))
        self.action_buttons.extend([start_button, self.compare_export_button])

        outer, list_card = self._card(content)
        outer.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        head = tk.Frame(list_card, bg=SURFACE)
        head.pack(fill="x")
        tk.Label(head, text="候选结果", bg=SURFACE, fg=MUTED, font=_font(11)).pack(side="left")
        tk.Label(head, textvariable=self.compare_count_var, bg=SURFACE, fg=MUTED, font=_font(10)).pack(side="right")
        columns = ("index", "score", "type", "left", "left_info", "right", "right_info", "state")
        self.compare_table = ttk.Treeview(list_card, columns=columns, show="headings", style="Studio.Treeview")
        headings = {
            "index": "序号",
            "score": "相似度",
            "type": "类型",
            "left": "A 文件",
            "left_info": "A 信息",
            "right": "B 文件",
            "right_info": "B 信息",
            "state": "状态",
        }
        widths = {"index": 68, "score": 90, "type": 120, "left": 190, "left_info": 130, "right": 190, "right_info": 130, "state": 90}
        for name in columns:
            self.compare_table.heading(name, text=headings[name])
            self.compare_table.column(name, width=widths[name], stretch=name in {"left", "right"})
        self.compare_table.pack(fill="both", expand=True, pady=(12, 0))
        self.compare_table.bind("<<TreeviewSelect>>", lambda _event: self._refresh_selected_compare())
        self.compare_table.bind("<Double-Button-1>", lambda _event: self._open_large_compare())

        outer, preview_card = self._card(content)
        outer.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        tk.Label(preview_card, text="当前结果", bg=SURFACE, fg=MUTED, font=_font(11)).pack(anchor="w")
        tk.Label(preview_card, textvariable=self.compare_result_var, bg=SURFACE, fg=TEXT, font=_font(14), justify="left", wraplength=500).pack(anchor="w", pady=(8, 0))
        tk.Label(preview_card, textvariable=self.compare_recommend_var, bg=SURFACE, fg=ACCENT, font=_font(11, "bold"), justify="left", wraplength=500).pack(anchor="w", pady=(8, 0))
        controls = tk.Frame(preview_card, bg=SURFACE)
        controls.pack(fill="x", pady=(12, 0))
        prev_button = FlatButton(controls, text="上一组", quiet=True, command=self._prev_compare)
        prev_button.pack(side="left")
        next_button = FlatButton(controls, text="下一组", quiet=True, command=self._next_compare)
        next_button.pack(side="left", padx=(10, 0))
        fit_button = FlatButton(controls, text="适应窗口", quiet=True, command=self._fit_compare_previews)
        fit_button.pack(side="left", padx=(10, 0))
        actual_button = FlatButton(controls, text="原始大小", quiet=True, command=self._actual_compare_previews)
        actual_button.pack(side="left", padx=(10, 0))
        large_button = FlatButton(controls, text="查看大图", quiet=True, command=self._open_large_compare)
        large_button.pack(side="left", padx=(10, 0))
        self.action_buttons.extend([prev_button, next_button, fit_button, actual_button, large_button])

        previews = tk.Frame(preview_card, bg=SURFACE)
        previews.pack(fill="both", expand=True, pady=(16, 0))
        previews.grid_columnconfigure(0, weight=1)
        previews.grid_columnconfigure(1, weight=1)
        previews.grid_rowconfigure(0, weight=1)
        self.preview_a = ZoomPreviewPane(previews, "A 图", open_callback=self._open_large_compare)
        self.preview_a.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.preview_b = ZoomPreviewPane(previews, "B 图", open_callback=self._open_large_compare)
        self.preview_b.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        info = tk.Frame(preview_card, bg=SURFACE)
        info.pack(fill="x", pady=(16, 0))
        info.grid_columnconfigure(0, weight=1)
        info.grid_columnconfigure(1, weight=1)
        tk.Label(info, textvariable=self.compare_left_var, bg=SURFACE, fg=MUTED, font=_font(11), justify="left", anchor="nw", wraplength=420).grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Label(info, textvariable=self.compare_right_var, bg=SURFACE, fg=MUTED, font=_font(11), justify="left", anchor="nw", wraplength=420).grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        bottom_actions = tk.Frame(preview_card, bg=SURFACE)
        bottom_actions.pack(fill="x", pady=(16, 0))
        keep_button = FlatButton(bottom_actions, text="保留并下一组", primary=True, command=self._keep_and_next)
        keep_button.pack(side="left")
        recommend_button = FlatButton(bottom_actions, text="按建议删除", command=self._delete_recommended)
        recommend_button.pack(side="left", padx=(10, 0))
        delete_left_button = FlatButton(bottom_actions, text="删左图", command=lambda: self._delete_compare_side("A"))
        delete_left_button.pack(side="left", padx=(10, 0))
        delete_right_button = FlatButton(bottom_actions, text="删右图", command=lambda: self._delete_compare_side("B"))
        delete_right_button.pack(side="left", padx=(10, 0))
        open_left_button = FlatButton(bottom_actions, text="打开左图位置", command=lambda: self._open_compare_side("A"))
        open_left_button.pack(side="left", padx=(10, 0))
        open_right_button = FlatButton(bottom_actions, text="打开右图位置", command=lambda: self._open_compare_side("B"))
        open_right_button.pack(side="left", padx=(10, 0))
        self.action_buttons.extend([keep_button, recommend_button, delete_left_button, delete_right_button, open_left_button, open_right_button])
        return page

    def _start_scan(self) -> None:
        input_dir = self.scan_dir.get()
        if not input_dir:
            messagebox.showinfo("请先选择目录", "请先选择图片目录。")
            return
        min_sim = self._read_float_value(self.scan_min.get(), "最低相似度", minimum=0.0, maximum=1.0)
        max_sim = self._read_float_value(self.scan_max.get(), "最高相似度", minimum=0.0, maximum=1.0)
        top_k = self._read_int_value(self.scan_topk.get().strip() or "20", "每张图片保留", minimum=1, maximum=100)
        if min_sim is None or max_sim is None or top_k is None:
            return
        if min_sim > max_sim:
            messagebox.showwarning("输入有误", "最低相似度不能高于最高相似度。")
            return
        args = self._task_args(
            high_precision=self.scan_high_precision.get(),
            input_dir=input_dir,
            output_xlsx=self.scan_output.get() or None,
            min_sim=min_sim,
            max_sim=max_sim,
            top_k=top_k,
            non_recursive=not self.scan_recursive.get(),
        )
        self.scan_result_var.set("正在扫描，请稍候。")

        def worker(status_cb, progress_cb):
            return run_scan_job(args, status_cb=status_cb, progress_cb=progress_cb)

        def done(result: dict) -> None:
            self.scan_result_var.set(f"已完成\nExcel：{result['output_path']}")
            self.model_var.set(self._model_summary())
            messagebox.showinfo("扫描完成", f"Excel 已生成：\n\n{result['output_path']}")

        self._start_worker(worker, done)

    def _start_reference(self) -> None:
        reference_image = self.reference_file.get()
        search_dir = self.reference_dir.get()
        if not reference_image or not search_dir:
            messagebox.showinfo("请补全路径", "请先选择参考图片和搜索目录。")
            return
        min_sim = self._read_float_value(self.reference_min.get(), "最低相似度", minimum=0.0, maximum=1.0)
        max_sim = self._read_float_value(self.reference_max.get(), "最高相似度", minimum=0.0, maximum=1.0)
        if min_sim is None or max_sim is None:
            return
        if min_sim > max_sim:
            messagebox.showwarning("输入有误", "最低相似度不能高于最高相似度。")
            return
        args = self._task_args(
            high_precision=self.reference_high_precision.get(),
            reference_image=reference_image,
            search_dir=search_dir,
            output_txt=self.reference_output.get() or None,
            min_sim=min_sim,
            max_sim=max_sim,
            non_recursive=not self.reference_recursive.get(),
        )
        self.reference_result_var.set("正在检索，请稍候。")

        def worker(status_cb, progress_cb):
            return run_reference_job(args, status_cb=status_cb, progress_cb=progress_cb)

        def done(result: dict) -> None:
            self.reference_result_var.set(f"已完成\nTXT：{result['output_path']}")
            self.model_var.set(self._model_summary())
            messagebox.showinfo("检索完成", f"TXT 已生成：\n\n{result['output_path']}")

        self._start_worker(worker, done)

    def _start_compare(self) -> None:
        dir_a = self.compare_dir_a.get()
        dir_b = self.compare_dir_b.get()
        if not dir_a or not dir_b:
            messagebox.showinfo("请补全路径", "请先选择目录 A 和目录 B。")
            return
        if os.path.normcase(os.path.abspath(dir_a)) == os.path.normcase(os.path.abspath(dir_b)):
            messagebox.showwarning("目录重复", "目录 A 和目录 B 不能是同一个目录。")
            return
        min_sim = self._read_float_value(self.compare_min.get(), "最低相似度", minimum=0.0, maximum=1.0)
        max_sim = self._read_float_value(self.compare_max.get(), "最高相似度", minimum=0.0, maximum=1.0)
        top_k_per_a = self._read_int_value(self.compare_topk.get().strip() or "3", "每张 A 保留", minimum=1, maximum=100)
        if min_sim is None or max_sim is None or top_k_per_a is None:
            return
        if min_sim > max_sim:
            messagebox.showwarning("输入有误", "最低相似度不能高于最高相似度。")
            return
        args = self._task_args(
            high_precision=self.compare_high_precision.get(),
            dir_a=dir_a,
            dir_b=dir_b,
            min_sim=min_sim,
            max_sim=max_sim,
            top_k_per_a=top_k_per_a,
            non_recursive=not self.compare_recursive.get(),
        )
        self.compare_result_var.set("正在复核，请稍候。")

        def worker(status_cb, progress_cb):
            return run_ab_compare_job(args, status_cb=status_cb, progress_cb=progress_cb)

        def done(result: dict) -> None:
            self.compare_meta = result
            self.compare_rows = list(result["rows"])
            self._reload_compare_table()
            self.model_var.set(self._model_summary())
            if self.compare_rows:
                self._select_compare_index(0)
                self._open_large_compare()
            else:
                self.compare_result_var.set("没有符合阈值的结果。")
                self.compare_recommend_var.set("可以调低阈值后再试。")

        self._start_worker(worker, done)

    def _reload_compare_table(self) -> None:
        self.compare_selected_index = -1
        self.compare_table.delete(*self.compare_table.get_children())
        self.compare_export_button.configure(state="normal" if self.compare_rows else "disabled")
        for index, row in enumerate(self.compare_rows, start=1):
            state = row.get("deleted_side", "") or "待处理"
            self.compare_table.insert(
                "",
                "end",
                iid=str(index - 1),
                values=(
                    index,
                    f"{row['similarity_score']:.4f}",
                    match_type_label(row.get("match_type", "")),
                    row.get("name_1", Path(row["path_1"]).name),
                    self._format_side_brief(row, "A"),
                    row.get("name_2", Path(row["path_2"]).name),
                    self._format_side_brief(row, "B"),
                    state,
                ),
            )
        self.compare_count_var.set(f"共 {len(self.compare_rows)} 组")
        if not self.compare_rows:
            self.preview_a.clear("暂无结果")
            self.preview_b.clear("暂无结果")
            self.compare_left_var.set("等待结果")
            self.compare_right_var.set("等待结果")
        self._refresh_large_compare_window()

    def _select_compare_index(self, index: int) -> None:
        if not self.compare_rows:
            return
        index = max(0, min(index, len(self.compare_rows) - 1))
        self.compare_selected_index = index
        iid = str(index)
        self.compare_table.selection_set(iid)
        self.compare_table.focus(iid)
        self.compare_table.see(iid)
        self._refresh_selected_compare()

    def _current_compare_row(self) -> dict | None:
        selection = self.compare_table.selection()
        if not selection:
            return None
        index = int(selection[0])
        if 0 <= index < len(self.compare_rows):
            self.compare_selected_index = index
            return self.compare_rows[index]
        return None

    def _refresh_selected_compare(self) -> None:
        row = self._current_compare_row()
        if row is None:
            return
        if Path(row["path_1"]).exists():
            self.preview_a.set_image(row["path_1"])
        else:
            self.preview_a.clear("文件已删除")
        if Path(row["path_2"]).exists():
            self.preview_b.set_image(row["path_2"])
        else:
            self.preview_b.clear("文件已删除")
        deleted_side = row.get("deleted_side", "")
        suffix = f"\n已删除：{deleted_side}" if deleted_side else ""
        self.compare_result_var.set(
            f"第 {self.compare_selected_index + 1} / {len(self.compare_rows)} 组\n"
            f"相似度 {row['similarity_score']:.4f} · {match_type_label(row.get('match_type', ''))}{suffix}"
        )
        recommendation = self._recommend_delete(row)
        self.compare_recommend_var.set(recommendation["text"])
        self.compare_left_var.set(self._format_side_info(row, "A"))
        self.compare_right_var.set(self._format_side_info(row, "B"))
        self._refresh_large_compare_window()

    def _format_side_info(self, row: dict, side: str) -> str:
        prefix = "1" if side == "A" else "2"
        path = row[f"path_{prefix}"]
        parent = Path(path).parent
        return "\n".join(
            [
                f"{side}：{row.get(f'name_{prefix}', Path(path).name)}",
                f"大小：{human_size(int(row.get(f'file_size_{prefix}', 0) or 0))}",
                f"分辨率：{row.get(f'resolution_{prefix}', '-')}",
                f"修改时间：{self._format_mtime(path)}",
                f"目录：{parent}",
                f"路径：{path}",
            ]
        )

    def _format_side_brief(self, row: dict, side: str) -> str:
        prefix = "1" if side == "A" else "2"
        return f"{row.get(f'resolution_{prefix}', '-')} | {human_size(int(row.get(f'file_size_{prefix}', 0) or 0))}"

    def _format_mtime(self, path: str) -> str:
        try:
            return datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "未知"

    def _parse_resolution(self, text: str) -> tuple[int, int]:
        left, _, right = str(text).partition("x")
        try:
            return int(left), int(right)
        except Exception:
            return 0, 0

    def _recommend_delete(self, row: dict) -> dict:
        if row.get("deleted_side"):
            return {"side": None, "text": f"当前已删除 {row['deleted_side']} 侧，不再给出建议。"}
        match_type = row.get("match_type", "")
        w1, h1 = self._parse_resolution(row.get("resolution_1", "0x0"))
        w2, h2 = self._parse_resolution(row.get("resolution_2", "0x0"))
        px1 = w1 * h1
        px2 = w2 * h2
        size1 = int(row.get("file_size_1", 0) or 0)
        size2 = int(row.get("file_size_2", 0) or 0)
        if match_type in {"exact_file", "cross_same_file_hash"}:
            return {"side": "B", "text": "建议删右图：两边文件内容完全一致，默认保留 A。"}
        if match_type in {"exact_pixels", "cross_same_pixel_hash"} and px1 == px2 and size1 == size2:
            return {"side": "B", "text": "建议删右图：像素完全一致，默认保留 A。"}
        score_a = 4.0 if px1 > px2 else 0.0
        score_b = 4.0 if px2 > px1 else 0.0
        score_a += 1.5 if size1 > size2 else 0.0
        score_b += 1.5 if size2 > size1 else 0.0
        try:
            mt1 = Path(row["path_1"]).stat().st_mtime
            mt2 = Path(row["path_2"]).stat().st_mtime
            score_a += 0.6 if mt1 > mt2 else 0.0
            score_b += 0.6 if mt2 > mt1 else 0.0
        except Exception:
            pass
        if score_a >= score_b:
            return {"side": "B", "text": "建议删右图：左图分辨率或质量更优。"}
        return {"side": "A", "text": "建议删左图：右图分辨率或质量更优。"}

    def _prev_compare(self) -> None:
        if self.compare_selected_index > 0:
            self._select_compare_index(self.compare_selected_index - 1)

    def _next_compare(self) -> None:
        if self.compare_selected_index < len(self.compare_rows) - 1:
            self._select_compare_index(self.compare_selected_index + 1)

    def _keep_and_next(self) -> None:
        self._next_compare()

    def _delete_recommended(self) -> None:
        row = self._current_compare_row()
        if row is None:
            return
        recommendation = self._recommend_delete(row)
        if recommendation["side"] in {"A", "B"}:
            self._delete_compare_side(recommendation["side"])

    def _delete_compare_side(self, side: str) -> None:
        row = self._current_compare_row()
        if row is None:
            return
        if row.get("deleted_side"):
            messagebox.showinfo("已处理", "这一组结果已经处理过。")
            return
        target = row["path_1"] if side == "A" else row["path_2"]
        if not messagebox.askyesno("确认删除", f"将文件移入回收站：\n\n{target}"):
            return
        try:
            action = move_to_trash(target)
        except Exception as exc:
            messagebox.showerror("删除失败", str(exc))
            return
        row["deleted_side"] = side
        self._append_log(f"{action}: {target}")
        current_index = self.compare_selected_index
        self._reload_compare_table()
        if not self.compare_rows:
            return
        next_index = current_index + 1 if current_index < len(self.compare_rows) - 1 else current_index
        self._select_compare_index(next_index)

    def _fit_compare_previews(self) -> None:
        self.preview_a.fit_view()
        self.preview_b.fit_view()

    def _actual_compare_previews(self) -> None:
        self.preview_a.actual_size()
        self.preview_b.actual_size()

    def _open_large_compare(self) -> None:
        if self._current_compare_row() is None:
            return
        if self.large_compare_window is None or not self.large_compare_window.win.winfo_exists():
            self.large_compare_window = LargeCompareWindow(self.root, self)
        else:
            self.large_compare_window.win.deiconify()
            self.large_compare_window.win.lift()
            self.large_compare_window.win.focus_force()
            self.large_compare_window.refresh()

    def _refresh_large_compare_window(self) -> None:
        if self.large_compare_window is None:
            return
        try:
            if self.large_compare_window.win.winfo_exists():
                self.large_compare_window.refresh()
        except Exception:
            self.large_compare_window = None

    def _open_compare_side(self, side: str) -> None:
        row = self._current_compare_row()
        if row is None:
            return
        target = row["path_1"] if side == "A" else row["path_2"]
        try:
            open_file_location(target)
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def _export_compare_report(self) -> None:
        if not self.compare_meta or not self.compare_rows:
            messagebox.showinfo("暂无结果", "请先开始复核。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("文本", "*.txt")])
        if not path:
            return
        try:
            write_cross_compare_txt_report(
                output_txt=Path(path),
                dir_a=Path(self.compare_meta["dir_a"]),
                dir_b=Path(self.compare_meta["dir_b"]),
                rows=self.compare_rows,
                errors_a=self.compare_meta["errors_a"],
                errors_b=self.compare_meta["errors_b"],
                min_sim=self.compare_meta["min_sim"],
                max_sim=self.compare_meta["max_sim"],
            )
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
            return
        self._append_log(f"已导出复核报告：{path}")
        messagebox.showinfo("导出完成", f"复核报告已生成：\n\n{path}")

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Left>", lambda _event: self._compare_shortcut(self._prev_compare))
        self.root.bind("<Right>", lambda _event: self._compare_shortcut(self._next_compare))
        self.root.bind("<space>", lambda _event: self._compare_shortcut(self._keep_and_next))
        self.root.bind("<Key-a>", lambda _event: self._compare_shortcut(lambda: self._delete_compare_side("A")))
        self.root.bind("<Key-d>", lambda _event: self._compare_shortcut(lambda: self._delete_compare_side("B")))
        self.root.bind("<Key-f>", lambda _event: self._compare_shortcut(self._fit_compare_previews))
        self.root.bind("<Key-v>", lambda _event: self._compare_shortcut(self._open_large_compare))

    def _compare_shortcut(self, callback) -> None:
        if self.current_mode != "compare" or self.job_running:
            return
        callback()


def launch_app() -> None:
    root = tk.Tk()
    ModernStudioApp(root)
    root.mainloop()
