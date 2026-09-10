#!/usr/bin/env python3
"""Download and verify the compact cSCC example inputs."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


RELEASE_BASE = "https://github.com/KuangQuanS/Spagraph/releases/download/v1.0.0"
FILES = {
    "GSE144236_P2_SC.h5ad": "adee03bc6f599471b1887ee41e34fb3991c051694caab5dd9a90e37cdc3a9473",
    "GSE144239_P2_ST.h5ad": "f5855a4974bb421deb9b6da01bb087fdfc9f15729248536a56dc55a929f1b51c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, expected in FILES.items():
        destination = data_dir / name
        if not destination.exists() or sha256(destination) != expected:
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(f"{RELEASE_BASE}/{name}", destination)
        observed = sha256(destination)
        if observed != expected:
            destination.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256 mismatch for {name}: {observed}")
        print(f"Verified {destination}")


if __name__ == "__main__":
    main()
