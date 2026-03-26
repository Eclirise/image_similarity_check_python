from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List


@dataclass
class RecentState:
    recent_dirs: List[str] = field(default_factory=list)
    recent_files: List[str] = field(default_factory=list)


class RecentStore:
    def __init__(self, path: Path | None = None, max_items: int = 12) -> None:
        self.path = path or (Path.home() / ".image_similarity_workbench" / "recent.json")
        self.max_items = max_items
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> RecentState:
        if not self.path.exists():
            return RecentState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return RecentState(
                recent_dirs=list(data.get("recent_dirs", [])),
                recent_files=list(data.get("recent_files", [])),
            )
        except Exception:
            return RecentState()

    def save(self, state: RecentState) -> None:
        self.path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")

    def push_dir(self, value: str) -> RecentState:
        state = self.load()
        state.recent_dirs = [x for x in state.recent_dirs if x != value]
        state.recent_dirs.insert(0, value)
        state.recent_dirs = state.recent_dirs[: self.max_items]
        self.save(state)
        return state

    def push_file(self, value: str) -> RecentState:
        state = self.load()
        state.recent_files = [x for x in state.recent_files if x != value]
        state.recent_files.insert(0, value)
        state.recent_files = state.recent_files[: self.max_items]
        self.save(state)
        return state
