from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from attacks.common.config import load_json, save_json


def _lock_age_hours(path: Path) -> float:
    try:
        payload = load_json(path)
        started = datetime.fromisoformat(payload["started_at"])

        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)

        return (datetime.now(timezone.utc) - started).total_seconds() / 3600.0
    except Exception:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return (datetime.now(timezone.utc) - modified).total_seconds() / 3600.0


def acquire_lock(
    path: str | Path,
    *,
    reference_id: int,
    mechanism_fingerprint: str,
    timeout_hours: float = 24.0,
    force_unlock_stale: bool = False,
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        age_hours = _lock_age_hours(path)

        if not force_unlock_stale or age_hours <= timeout_hours:
            raise RuntimeError(
                f"Lock already exists: {path} (age={age_hours:.2f} hours)"
            )

        path.unlink()

    payload = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reference_id": int(reference_id),
        "mechanism_fingerprint": mechanism_fingerprint,
    }

    try:
        with path.open("x", encoding="utf-8") as f:
            import json

            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    except FileExistsError as exc:
        raise RuntimeError(f"Lock was acquired concurrently: {path}") from exc

    return payload


def release_lock(path: str | Path, expected_pid: int | None = None) -> None:
    path = Path(path)

    if not path.exists():
        return

    if expected_pid is not None:
        try:
            payload = load_json(path)

            if int(payload.get("pid", -1)) != int(expected_pid):
                raise RuntimeError(f"Refusing to release a foreign lock: {path}")
        except ValueError:
            raise

    path.unlink()
