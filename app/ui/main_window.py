from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.services import backend
from app.settings.recent_store import RecentStore
from app.workers.jobs import JobRunner
from image_similarity_check_python import write_cross_compare_txt_report


class ZoomImageView(QGraphicsView):
    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.item: Optional[QGraphicsPixmapItem] = None
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def set_image(self, path: str) -> None:
        img = QImage(path)
        self.scene().clear()
        if img.isNull():
            self.item = None
            return
        pix = QPixmap.fromImage(img)
        self.item = QGraphicsPixmapItem(pix)
        self.scene().addItem(self.item)
        self.fitInView(self.item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        if self.item is None:
            return
        if abs(self.transform().m11() - 1.0) < 0.2:
            self.fitInView(self.item, Qt.KeepAspectRatio)
        else:
            self.resetTransform()
        super().mouseDoubleClickEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("图片相似度工作台 2026")
        self.resize(1800, 1100)
        self.base_args = backend.build_common_args()
        self.recent = RecentStore()
        self.job: Optional[JobRunner] = None
        self.compare_rows = []
        self.compare_meta = None
        self._build_ui()
        self._load_recent()

    def _build_ui(self) -> None:
        self._build_menu()
        wrap = QWidget()
        root = QVBoxLayout(wrap)

        top = QHBoxLayout()
        self.status_label = QLabel("准备就绪")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        top.addWidget(self.status_label, 2)
        top.addWidget(self.progress, 1)
        root.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._scan_tab(), "目录扫描 → Excel")
        self.tabs.addTab(self._reference_tab(), "参考图检索 → TXT")
        self.tabs.addTab(self._compare_tab(), "A/B 双目录复核")
        root.addWidget(self.tabs)

        self.setCentralWidget(wrap)

    def _build_menu(self) -> None:
        m = self.menuBar()
        file_menu = m.addMenu("文件")
        install = QAction("一键安装依赖", self)
        install.triggered.connect(self._install_deps)
        file_menu.addAction(install)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

    def _scan_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        form = QFormLayout()
        self.scan_dir = QLineEdit(); self.scan_dir.setMinimumHeight(40)
        self.scan_out = QLineEdit(); self.scan_out.setMinimumHeight(40)
        form.addRow("图片目录", self._with_browse(self.scan_dir, True))
        form.addRow("输出 Excel", self._with_save(self.scan_out, "*.xlsx"))
        self.scan_min = QLineEdit("0.90"); self.scan_max = QLineEdit("1.00"); self.scan_topk = QLineEdit("20")
        form.addRow("最低相似度", self.scan_min); form.addRow("最高相似度", self.scan_max); form.addRow("Top-K", self.scan_topk)
        self.scan_recursive = QCheckBox("递归扫描子目录"); self.scan_recursive.setChecked(True)
        self.scan_deep = QCheckBox("启用深度特征"); self.scan_deep.setChecked(True)
        form.addRow(self.scan_recursive); form.addRow(self.scan_deep)
        self.scan_device = QComboBox(); self.scan_device.addItems(["auto", "cuda", "cpu"]); self.scan_device.setCurrentText("auto")
        form.addRow("计算设备（自动优先GPU）", self.scan_device)
        self.scan_recent = QComboBox(); form.addRow("最近目录", self.scan_recent)
        v.addLayout(form)
        btn = QPushButton("开始扫描")
        btn.setMinimumHeight(44)
        btn.clicked.connect(self._start_scan)
        v.addWidget(btn)
        self.scan_log = QTextEdit(); self.scan_log.setReadOnly(True)
        v.addWidget(self.scan_log, 1)
        return w

    def _reference_tab(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w); form = QFormLayout()
        self.ref_file = QLineEdit(); self.ref_file.setMinimumHeight(40)
        self.ref_dir = QLineEdit(); self.ref_dir.setMinimumHeight(40)
        self.ref_out = QLineEdit(); self.ref_out.setMinimumHeight(40)
        form.addRow("参考图", self._with_browse(self.ref_file, False))
        form.addRow("搜索目录", self._with_browse(self.ref_dir, True))
        form.addRow("输出 TXT", self._with_save(self.ref_out, "*.txt"))
        self.ref_min = QLineEdit("0.90"); self.ref_max = QLineEdit("1.00")
        form.addRow("最低相似度", self.ref_min); form.addRow("最高相似度", self.ref_max)
        self.ref_recursive = QCheckBox("递归扫描子目录"); self.ref_recursive.setChecked(True)
        self.ref_deep = QCheckBox("启用深度特征"); self.ref_deep.setChecked(True)
        form.addRow(self.ref_recursive); form.addRow(self.ref_deep)
        self.ref_device = QComboBox(); self.ref_device.addItems(["auto", "cuda", "cpu"]); self.ref_device.setCurrentText("auto")
        form.addRow("计算设备（自动优先GPU）", self.ref_device)
        self.ref_recent = QComboBox(); form.addRow("最近参考图", self.ref_recent)
        v.addLayout(form)
        btn = QPushButton("开始检索")
        btn.setMinimumHeight(44)
        btn.clicked.connect(self._start_ref)
        v.addWidget(btn)
        self.ref_log = QTextEdit(); self.ref_log.setReadOnly(True)
        v.addWidget(self.ref_log, 1)
        return w

    def _compare_tab(self) -> QWidget:
        w = QWidget(); layout = QVBoxLayout(w)
        form_box = QGroupBox("对比参数")
        form = QFormLayout(form_box)
        self.cmp_a = QLineEdit(); self.cmp_a.setMinimumHeight(40)
        self.cmp_b = QLineEdit(); self.cmp_b.setMinimumHeight(40)
        form.addRow("目录 A", self._with_browse(self.cmp_a, True))
        form.addRow("目录 B", self._with_browse(self.cmp_b, True))
        self.cmp_min = QLineEdit("0.92"); self.cmp_max = QLineEdit("1.00"); self.cmp_topk = QLineEdit("3")
        form.addRow("最低相似度", self.cmp_min); form.addRow("最高相似度", self.cmp_max); form.addRow("每张 A 候选", self.cmp_topk)
        self.cmp_recursive = QCheckBox("递归扫描子目录"); self.cmp_recursive.setChecked(True)
        self.cmp_deep = QCheckBox("启用深度特征"); self.cmp_deep.setChecked(True)
        form.addRow(self.cmp_recursive); form.addRow(self.cmp_deep)
        self.cmp_device = QComboBox(); self.cmp_device.addItems(["auto", "cuda", "cpu"]); self.cmp_device.setCurrentText("auto")
        form.addRow("计算设备（自动优先GPU）", self.cmp_device)
        self.filter_type = QComboBox(); self.filter_type.addItems(["全部", "exact_file", "exact_pixels", "near_duplicate", "strong_match", "possible_match"])
        self.filter_type.currentTextChanged.connect(self._apply_compare_filter)
        form.addRow("结果筛选", self.filter_type)
        layout.addWidget(form_box)

        act = QHBoxLayout()
        start = QPushButton("开始 A/B 对比"); start.setMinimumHeight(44); start.clicked.connect(self._start_compare)
        export = QPushButton("导出复核报告"); export.clicked.connect(self._export_compare)
        next_btn = QPushButton("下一组"); next_btn.clicked.connect(self._next_row)
        act.addWidget(start); act.addWidget(export); act.addWidget(next_btn)
        layout.addLayout(act)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget(); l1 = QVBoxLayout(left); self.view_a = ZoomImageView(); l1.addWidget(QLabel("A 预览")); l1.addWidget(self.view_a)
        right = QWidget(); l2 = QVBoxLayout(right); self.view_b = ZoomImageView(); l2.addWidget(QLabel("B 预览")); l2.addWidget(self.view_b)
        splitter.addWidget(left); splitter.addWidget(right); splitter.setSizes([900, 900])
        layout.addWidget(splitter, 3)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["#", "相似度", "类型", "A 路径", "B 路径", "A 分辨率", "B 分辨率", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSortingEnabled(True)
        self.table.itemSelectionChanged.connect(self._show_selected_row)
        layout.addWidget(self.table, 2)

        op = QHBoxLayout()
        del_a = QPushButton("删除 A"); del_b = QPushButton("删除 B")
        del_a.clicked.connect(lambda: self._delete_side("A")); del_b.clicked.connect(lambda: self._delete_side("B"))
        open_a = QPushButton("打开 A 所在位置"); open_b = QPushButton("打开 B 所在位置")
        open_a.clicked.connect(lambda: self._open_side("A")); open_b.clicked.connect(lambda: self._open_side("B"))
        op.addWidget(del_a); op.addWidget(del_b); op.addWidget(open_a); op.addWidget(open_b)
        layout.addLayout(op)

        self.cmp_log = QTextEdit(); self.cmp_log.setReadOnly(True)
        layout.addWidget(self.cmp_log, 1)
        return w

    def _with_browse(self, edit: QLineEdit, is_dir: bool) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("浏览…")
        btn.setMinimumHeight(40)
        btn.clicked.connect(lambda: self._pick_path(edit, is_dir))
        h.addWidget(edit, 1); h.addWidget(btn)
        return w

    def _with_save(self, edit: QLineEdit, pattern: str) -> QWidget:
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("另存为…")
        btn.setMinimumHeight(40)
        btn.clicked.connect(lambda: self._save_path(edit, pattern))
        h.addWidget(edit, 1); h.addWidget(btn)
        return w

    def _pick_path(self, edit: QLineEdit, is_dir: bool) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录") if is_dir else QFileDialog.getOpenFileName(self, "选择图片", filter="图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.gif)")[0]
        if path:
            edit.setText(path)

    def _save_path(self, edit: QLineEdit, pattern: str) -> None:
        path = QFileDialog.getSaveFileName(self, "保存", filter=pattern)[0]
        if path:
            edit.setText(path)

    def _run_job(self, fn, kwargs, on_done, log: QTextEdit) -> None:
        if self.job is not None and self.job.isRunning():
            QMessageBox.information(self, "任务运行中", "请等待当前任务完成。")
            return
        self.progress.setRange(0, 0)
        self.progress.setValue(0)
        self.status_label.setText("任务启动中…")
        self.job = JobRunner(fn, kwargs)
        self.job.signals.status.connect(lambda x: (self.status_label.setText(x), log.append(x)))
        self.job.signals.progress.connect(self._on_progress)
        self.job.signals.failed.connect(lambda e: self._on_failed(e))
        self.job.signals.done.connect(lambda res: self._on_done(res, on_done))
        self.job.start()

    def _on_progress(self, current: int, total: int, stage: str) -> None:
        if total > 1:
            self.progress.setRange(0, total)
            self.progress.setValue(min(current, total))
            self.status_label.setText(f"{stage} {current}/{total}")
        else:
            self.progress.setRange(0, 0)
            self.status_label.setText(stage)

    def _on_failed(self, message: str) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        QMessageBox.critical(self, "任务失败", message)

    def _on_done(self, result: dict, on_done) -> None:
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        on_done(result)

    def _start_scan(self) -> None:
        in_dir = self.scan_dir.text().strip()
        if not in_dir:
            return
        self.recent.push_dir(in_dir)
        self.base_args.device = self.scan_device.currentText()
        self.status_label.setText(backend.runtime_summary(self.base_args.device))
        self._run_job(
            backend.run_scan,
            dict(
                base=self.base_args,
                input_dir=in_dir,
                output_xlsx=self.scan_out.text().strip() or None,
                min_sim=float(self.scan_min.text()),
                max_sim=float(self.scan_max.text()),
                top_k=int(self.scan_topk.text()),
                non_recursive=not self.scan_recursive.isChecked(),
                no_openclip=not self.scan_deep.isChecked(),
            ),
            lambda r: QMessageBox.information(self, "完成", f"Excel 已生成：\n{r['output_path']}"),
            self.scan_log,
        )

    def _start_ref(self) -> None:
        ref = self.ref_file.text().strip(); d = self.ref_dir.text().strip()
        if not ref or not d:
            return
        self.recent.push_file(ref); self.recent.push_dir(d)
        self.base_args.device = self.ref_device.currentText()
        self.status_label.setText(backend.runtime_summary(self.base_args.device))
        self._run_job(
            backend.run_reference,
            dict(
                base=self.base_args,
                reference_image=ref,
                search_dir=d,
                output_txt=self.ref_out.text().strip() or None,
                min_sim=float(self.ref_min.text()),
                max_sim=float(self.ref_max.text()),
                non_recursive=not self.ref_recursive.isChecked(),
                no_openclip=not self.ref_deep.isChecked(),
            ),
            lambda r: QMessageBox.information(self, "完成", f"TXT 已生成：\n{r['output_path']}"),
            self.ref_log,
        )

    def _start_compare(self) -> None:
        a = self.cmp_a.text().strip(); b = self.cmp_b.text().strip()
        if not a or not b:
            return
        self.recent.push_dir(a); self.recent.push_dir(b)
        self.base_args.device = self.cmp_device.currentText()
        self.status_label.setText(backend.runtime_summary(self.base_args.device))
        self._run_job(
            backend.run_compare,
            dict(
                base=self.base_args,
                dir_a=a,
                dir_b=b,
                min_sim=float(self.cmp_min.text()),
                max_sim=float(self.cmp_max.text()),
                top_k_per_a=int(self.cmp_topk.text()),
                non_recursive=not self.cmp_recursive.isChecked(),
                no_openclip=not self.cmp_deep.isChecked(),
            ),
            self._on_compare_done,
            self.cmp_log,
        )

    def _on_compare_done(self, result: dict) -> None:
        self.compare_rows = result["rows"]
        self.compare_meta = result
        self._apply_compare_filter()
        self.status_label.setText(f"完成：{len(self.compare_rows)} 组")

    def _apply_compare_filter(self) -> None:
        rows = self.compare_rows
        current = self.filter_type.currentText() if hasattr(self, "filter_type") else "全部"
        if current != "全部":
            rows = [r for r in rows if r.get("match_type") == current]
        self.table.setRowCount(0)
        for i, row in enumerate(rows):
            rid = self.table.rowCount()
            self.table.insertRow(rid)
            vals = [
                str(i + 1),
                f"{row['similarity_score']:.4f}",
                row.get("match_type", ""),
                row.get("path_1", ""),
                row.get("path_2", ""),
                row.get("resolution_1", ""),
                row.get("resolution_2", ""),
                row.get("deleted_side", "") or "待处理",
            ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c == 0:
                    item.setData(Qt.UserRole, row)
                self.table.setItem(rid, c, item)
        if self.table.rowCount() > 0:
            self.table.selectRow(0)

    def _current_row_data(self) -> Optional[dict]:
        r = self.table.currentRow()
        if r < 0:
            return None
        item = self.table.item(r, 0)
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def _show_selected_row(self) -> None:
        row = self._current_row_data()
        if not row:
            return
        self.view_a.set_image(row["path_1"])
        self.view_b.set_image(row["path_2"])

    def _next_row(self) -> None:
        r = self.table.currentRow()
        if r < self.table.rowCount() - 1:
            self.table.selectRow(r + 1)

    def _delete_side(self, side: str) -> None:
        row = self._current_row_data()
        if not row:
            return
        target = row["path_1"] if side == "A" else row["path_2"]
        if QMessageBox.question(self, "确认删除", f"将文件移入回收站？\n\n{target}") != QMessageBox.Yes:
            return
        try:
            msg = backend.safe_delete(target)
        except Exception as e:
            QMessageBox.critical(self, "删除失败", str(e))
            return
        row["deleted_side"] = side
        self.cmp_log.append(f"{msg}: {target}")
        self._apply_compare_filter()

    def _open_side(self, side: str) -> None:
        row = self._current_row_data()
        if not row:
            return
        path = row["path_1"] if side == "A" else row["path_2"]
        try:
            backend.open_location(path)
        except Exception as e:
            QMessageBox.critical(self, "打开失败", str(e))

    def _export_compare(self) -> None:
        if not self.compare_meta:
            return
        path = QFileDialog.getSaveFileName(self, "导出复核报告", filter="TXT (*.txt)")[0]
        if not path:
            return
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
        QMessageBox.information(self, "导出成功", path)

    def _install_deps(self) -> None:
        QMessageBox.information(self, "说明", "请双击 launch_studio.bat（Windows）或运行 launch_studio.sh 来自动安装依赖并启动。")

    def _load_recent(self) -> None:
        state = self.recent.load()
        self.scan_recent.addItems(state.recent_dirs)
        self.ref_recent.addItems(state.recent_files)
        self.scan_recent.currentTextChanged.connect(self.scan_dir.setText)
        self.ref_recent.currentTextChanged.connect(self.ref_file.setText)
