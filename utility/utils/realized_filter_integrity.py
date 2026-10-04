"""Locate required legacy files and record their current hashes."""

import hashlib
from pathlib import Path
from typing import Dict


REQUIRED_LEGACY_FILES = (
    "algorithm/DPSGD.py",
    "algorithm/DPSGD_HF.py",
    "utils/dp_optimizer.py",
    "privacy_analysis/PLD/FindSigmaSmallGMM.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_legacy_core(root: Path) -> Dict[str, str]:
    actual = {}
    missing = []
    for relative in REQUIRED_LEGACY_FILES:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual[relative] = _sha256(path)
    if missing:
        raise RuntimeError(
            "Required legacy source files are missing:\n" + "\n".join(missing)
        )
    return actual
