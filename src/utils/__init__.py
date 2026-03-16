from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import fcntl


PathLike = str | os.PathLike[str] | Path


def _to_path(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def read_text(path: PathLike, *, encoding: str = "utf-8") -> str | None:
    p = _to_path(path)
    try:
        return p.read_text(encoding=encoding)
    except Exception:
        return None


def read_json(path: PathLike, *, encoding: str = "utf-8") -> Any:
    s = read_text(path, encoding=encoding)
    if s is None:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def read_json_dict(path: PathLike, *, encoding: str = "utf-8") -> dict[str, Any] | None:
    obj = read_json(path, encoding=encoding)
    return obj if isinstance(obj, dict) else None


def atomic_write_text(path: PathLike, text: str, *, encoding: str = "utf-8") -> None:
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(str(text), encoding=encoding)
    os.replace(str(tmp), str(p))


def atomic_write_json(
    path: PathLike,
    obj: Any,
    *,
    encoding: str = "utf-8",
    indent: int | None = 2,
) -> None:
    atomic_write_text(
        path,
        json.dumps(obj, ensure_ascii=False, indent=indent),
        encoding=encoding,
    )


def append_jsonl(path: PathLike, obj: Any, *, encoding: str = "utf-8") -> None:
    p = _to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(obj, ensure_ascii=False) + "\n").encode(encoding)
    fd = None
    try:
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        off = 0
        while off < len(line):
            n = os.write(fd, line[off:])
            if not isinstance(n, int) or n <= 0:
                break
            off += n
    finally:
        try:
            if fd is not None:
                os.close(fd)
        except Exception:
            pass


def locked_update_json(
    path: PathLike,
    *,
    lock_path: PathLike,
    update: Callable[[dict[str, Any]], dict[str, Any]],
    encoding: str = "utf-8",
    indent: int | None = 2,
) -> dict[str, Any]:
    p = _to_path(path)
    lp = _to_path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            existing = read_json_dict(p, encoding=encoding) or {}
            next_obj = update(dict(existing))
            if not isinstance(next_obj, dict):
                next_obj = {}
            atomic_write_json(p, next_obj, encoding=encoding, indent=indent)
            return next_obj
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
