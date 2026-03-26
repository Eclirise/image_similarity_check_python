from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Signal


class JobSignals(QObject):
    status = Signal(str)
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)


class JobRunner(QThread):
    def __init__(self, fn: Callable[..., Any], kwargs: dict) -> None:
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs
        self.signals = JobSignals()

    def run(self) -> None:
        try:
            runtime_kwargs = dict(self.kwargs)
            runtime_kwargs["status_cb"] = lambda msg: self.signals.status.emit(str(msg))
            runtime_kwargs["progress_cb"] = lambda cur, total, stage: self.signals.progress.emit(int(cur), int(total), str(stage))
            result = self.fn(**runtime_kwargs)
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        else:
            self.signals.done.emit(result)
