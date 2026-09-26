"""
memory.py — Short-term and Long-term Memory for MedRAGAgents
─────────────────────────────────────────────────────────────
Short-term memory: stores intermediate results within a single
                   question turn (held in RAM, discarded after).
Long-term memory : persistent JSON cache mapping question hash →
                   predicted answer, to avoid re-computation on
                   repeated questions across runs.
"""

import os
import json
import hashlib
from typing import Any, Dict, Optional


# ─────────────────────────────────────────────────────────────
# Short-term Memory
# ─────────────────────────────────────────────────────────────

class ShortTermMemory:
    """
    Stores intermediate agent outputs during one question's inference.
    Cleared between questions.
    """

    def __init__(self):
        self._store: Dict[str, Any] = {}

    # ── write ────────────────────────────────────────────────
    def store(self, key: str, value: Any) -> None:
        """Save an intermediate result under `key`."""
        self._store[key] = value

    # ── read ─────────────────────────────────────────────────
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a stored result (returns `default` if missing)."""
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    # ── lifecycle ────────────────────────────────────────────
    def clear(self) -> None:
        """Reset memory between questions."""
        self._store.clear()

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of the current memory state (for logging)."""
        return dict(self._store)

    def __repr__(self) -> str:
        keys = list(self._store.keys())
        return f"ShortTermMemory(keys={keys})"


# ─────────────────────────────────────────────────────────────
# Long-term Memory
# ─────────────────────────────────────────────────────────────

class LongTermMemory:
    """
    Persistent cache stored as a JSON file on disk.
    Maps SHA-256(question + options) → cached prediction record.

    Record format:
    {
        "pred_answer": "A",
        "syn_report":  "...",
        "variant":     "V3",
        "timestamp":   "2025-01-01T00:00:00"
    }
    """

    def __init__(self, cache_path: str = "./memory/long_term_cache.json"):
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self._cache: Dict[str, Any] = self._load()

    # ── persistence ──────────────────────────────────────────
    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def save(self) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    # ── key ──────────────────────────────────────────────────
    @staticmethod
    def make_key(question: str, options: Any) -> str:
        raw = question.strip() + str(options)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    # ── read / write ─────────────────────────────────────────
    def get(self, question: str, options: Any) -> Optional[Dict]:
        key = self.make_key(question, options)
        return self._cache.get(key)

    def store(self, question: str, options: Any, record: Dict, autosave: bool = True) -> None:
        """
        Cache a prediction record.
        `autosave=True` flushes to disk immediately (safe but slower).
        """
        from datetime import datetime
        key = self.make_key(question, options)
        record["timestamp"] = datetime.utcnow().isoformat()
        self._cache[key] = record
        if autosave:
            self.save()

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"LongTermMemory(entries={len(self._cache)}, path='{self.cache_path}')"


# ─────────────────────────────────────────────────────────────
# Combined Memory Manager
# ─────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Convenience wrapper that holds both memory types.
    Used by the pipeline to pass a single memory object around.
    """

    def __init__(self, long_term_path: str = "./memory/long_term_cache.json"):
        self.short = ShortTermMemory()
        self.long  = LongTermMemory(cache_path=long_term_path)

    def reset_short(self) -> None:
        """Call at the start of each question."""
        self.short.clear()

    def __repr__(self) -> str:
        return f"MemoryManager(short={self.short}, long={self.long})"
