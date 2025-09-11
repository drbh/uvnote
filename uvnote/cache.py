"""SQLite-backed cache index and eviction for uvnote.

Responsibilities:
- Track cache entries (key -> size, atime, mtime, success, checksum)
- Track artifacts per entry
- Provide integrity checks on read
- Enforce a size cap with simple LRU eviction
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


INDEX_REL = Path(".uvnote/index.db")
CACHE_REL = Path(".uvnote/cache")


def _db_path(work_dir: Path) -> Path:
    return work_dir / INDEX_REL


def _cache_dir(work_dir: Path) -> Path:
    return work_dir / CACHE_REL


def init_db(work_dir: Path) -> None:
    db = _db_path(work_dir)
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entries (
                key TEXT PRIMARY KEY,
                size_bytes INTEGER NOT NULL,
                atime REAL NOT NULL,
                mtime REAL NOT NULL,
                success INTEGER NOT NULL,
                checksum TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                key TEXT NOT NULL,
                relpath TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                PRIMARY KEY (key, relpath)
            )
            """
        )
        conn.commit()


def _now() -> float:
    return time.time()


def _dir_size_bytes(p: Path) -> int:
    total = 0
    if not p.exists():
        return 0
    for root, dirs, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except FileNotFoundError:
                # File might have been removed concurrently
                pass
    return total


def _sha256_file(p: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None


def record_write(
    work_dir: Path,
    key: str,
    cache_dir: Path,
    success: bool,
    artifacts: Iterable[str],
) -> None:
    """Record/refresh an entry after writing results and artifacts.

    Computes entry dir size and result.json checksum, updates atime/mtime,
    and replaces artifact rows for the key.
    """
    init_db(work_dir)
    db = _db_path(work_dir)
    size_bytes = _dir_size_bytes(cache_dir)
    checksum = _sha256_file(cache_dir / "result.json")
    now = _now()

    with sqlite3.connect(db) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO entries(key, size_bytes, atime, mtime, success, checksum)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
              size_bytes=excluded.size_bytes,
              atime=excluded.atime,
              mtime=excluded.mtime,
              success=excluded.success,
              checksum=excluded.checksum
            """,
            (key, size_bytes, now, now, 1 if success else 0, checksum),
        )
        # Replace artifacts set
        cur.execute("DELETE FROM artifacts WHERE key=?", (key,))
        for rel in artifacts:
            ap = cache_dir / rel
            try:
                s = ap.stat().st_size if ap.is_file() else 0
            except FileNotFoundError:
                s = 0
            cur.execute(
                "INSERT INTO artifacts(key, relpath, size_bytes) VALUES(?, ?, ?)",
                (key, rel, s),
            )
        conn.commit()


def record_access(work_dir: Path, key: str) -> None:
    """Update access time for an entry (LRU)."""
    init_db(work_dir)
    db = _db_path(work_dir)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE entries SET atime=? WHERE key=?",
            (_now(), key),
        )
        conn.commit()


def get_artifacts(work_dir: Path, key: str) -> List[str]:
    init_db(work_dir)
    db = _db_path(work_dir)
    with sqlite3.connect(db) as conn:
        cur = conn.execute(
            "SELECT relpath FROM artifacts WHERE key=? ORDER BY relpath", (key,)
        )
        return [row[0] for row in cur.fetchall()]


def get_total_size_bytes(work_dir: Path) -> int:
    init_db(work_dir)
    db = _db_path(work_dir)
    with sqlite3.connect(db) as conn:
        cur = conn.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM entries")
        row = cur.fetchone()
        return int(row[0] or 0)


def _parse_size_bytes(val: Optional[str], default_bytes: int) -> int:
    if not val:
        return default_bytes
    s = val.strip().lower()
    try:
        if s.endswith("kb") or s.endswith("k"):
            return int(float(s.rstrip("kbk"))) * 1024
        if s.endswith("mb") or s.endswith("m"):
            return int(float(s.rstrip("mbm"))) * 1024 * 1024
        if s.endswith("gb") or s.endswith("g"):
            return int(float(s.rstrip("gbg"))) * 1024 * 1024 * 1024
        return int(s)
    except ValueError:
        return default_bytes


def get_cache_cap_bytes() -> int:
    # Default 10 GiB
    return _parse_size_bytes(
        os.environ.get("UVNOTE_CACHE_SIZE"), 10 * 1024 * 1024 * 1024
    )


def evict_to_target(work_dir: Path, target_total_bytes: int) -> Tuple[int, List[str]]:
    """Evict least-recently-used entries until under target size.

    Returns (bytes_freed, keys_removed)
    """
    init_db(work_dir)
    db = _db_path(work_dir)
    removed_keys: List[str] = []
    freed = 0
    cache_root = _cache_dir(work_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db) as conn:
        # Fetch keys ordered by atime asc (oldest first)
        cur = conn.execute("SELECT key, size_bytes FROM entries ORDER BY atime ASC")
        rows = cur.fetchall()

        total = get_total_size_bytes(work_dir)
        for key, size in rows:
            if total <= target_total_bytes:
                break
            entry_dir = cache_root / key
            try:
                shutil.rmtree(entry_dir, ignore_errors=True)
            finally:
                conn.execute("DELETE FROM artifacts WHERE key=?", (key,))
                conn.execute("DELETE FROM entries WHERE key=?", (key,))
                removed_keys.append(key)
                # Recompute total conservatively
                total = max(0, total - int(size or 0))
                freed += int(size or 0)
        conn.commit()

    return freed, removed_keys


def integrity_check(work_dir: Path, key: str) -> bool:
    """Basic integrity check for a cached entry.

    Verifies that result.json, stdout.txt, stderr.txt, and recorded artifacts exist.
    If index has no artifacts, treat as pass (legacy entries).
    """
    cache_dir = _cache_dir(work_dir) / key
    if not cache_dir.exists():
        return False
    must = [
        cache_dir / "result.json",
        cache_dir / "stdout.txt",
        cache_dir / "stderr.txt",
    ]
    for p in must:
        if not p.exists():
            return False
    rec = get_artifacts(work_dir, key)
    for rel in rec:
        if not (cache_dir / rel).exists():
            return False
    return True


__all__ = [
    "init_db",
    "record_write",
    "record_access",
    "get_artifacts",
    "get_total_size_bytes",
    "get_cache_cap_bytes",
    "evict_to_target",
    "integrity_check",
]
