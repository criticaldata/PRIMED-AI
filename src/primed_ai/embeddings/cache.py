"""On-disk embedding cache for the frozen-embedding pipeline.

Stores EchoJEPA-L and ECG-FM embeddings so downstream probes and evaluations can
reuse them without re-encoding. Each modality has one .npy file per
study/record ID and an append-only metadata manifest::

<root>/<modality>/<id>.npy
<root>/<modality>/manifest.jsonl

Cached IDs are determined from the files on disk, so incomplete manifests do not
hide existing embeddings.

Expected disk usage ~= n_records * prod(shape) * 4 bytes → ~5 MB per modality
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

__all__ = ["EmbeddingCache"]

_ARRAY_SUFFIX = ".npy"
_MANIFEST_NAME = "manifest.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_safe_name(name: str, kind: str) -> str:
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"{kind} is not a safe path component: {name!r}")
    return name


class EmbeddingCache:
    """Read/write embedding cache for one modality, keyed by record id.

    ``root`` is the base dir (e.g. ``embeddings/``); ``modality`` names the sub-dir
    (``"echo"`` / ``"ecg"``); ``encoder_version`` is recorded on every write so
    cached vectors stay traceable to the checkpoint that produced them.
    """

    def __init__(self, root, modality: str, encoder_version: str = "unknown") -> None:
        self.root = Path(root)
        self.modality = _check_safe_name(modality, "modality")
        self.encoder_version = encoder_version
        self.dir = self.root / self.modality
        self.manifest_path = self.dir / _MANIFEST_NAME

    # -- id <-> path -------------------------------------------------------
    def _path(self, record_id) -> Path:
        key = _check_safe_name(str(record_id), "record id")
        return self.dir / f"{key}{_ARRAY_SUFFIX}"

    def has(self, record_id) -> bool:
        """True if an embedding is already cached for ``record_id`` (for resume)."""
        return self._path(record_id).is_file()

    def __contains__(self, record_id) -> bool:
        return self.has(record_id)

    # -- write -------------------------------------------------------------
    def write(self, record_id, embedding, *, overwrite: bool = False,
              timestamp: str | None = None) -> None:
        """Cache one embedding; atomic (temp file + rename).

        Raises ``FileExistsError`` if present unless ``overwrite=True``. Pass a
        CPU array (GPU callers: ``tensor.cpu().numpy()``).
        """
        arr = np.asarray(embedding)
        if arr.dtype == object or arr.size == 0:
            raise ValueError("embedding must be a non-empty numeric array")
        path = self._path(record_id)
        if path.is_file() and not overwrite:
            raise FileExistsError(
                f"{record_id!r} already cached (pass overwrite=True to replace)")
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        # Save via a handle so numpy keeps our temp name (else it appends '.npy').
        with open(tmp, "wb") as f:
            np.save(f, arr)
        tmp.replace(path)
        self._append_manifest(record_id, arr, timestamp or _utc_now_iso())

    def _append_manifest(self, record_id, arr: np.ndarray, timestamp: str) -> None:
        record = {
            "id": str(record_id),
            "modality": self.modality,
            "encoder_version": self.encoder_version,
            "timestamp": timestamp,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "path": self._path(record_id).name,
        }
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # -- read --------------------------------------------------------------
    def read(self, record_id, *, mmap: bool = True) -> np.ndarray:
        """Return the cached embedding (``KeyError`` if absent).

        ``mmap=True`` (default) memory-maps for fast random access; ``mmap=False``
        returns a writable in-memory copy.
        """
        path = self._path(record_id)
        if not path.is_file():
            raise KeyError(record_id)
        return np.load(path, mmap_mode="r" if mmap else None)

    def __getitem__(self, record_id) -> np.ndarray:
        return self.read(record_id)

    # -- introspection -----------------------------------------------------
    def keys(self) -> list[str]:
        """Sorted ids currently cached (derived from the .npy files on disk)."""
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob(f"*{_ARRAY_SUFFIX}"))

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def disk_usage_bytes(self) -> int:
        """Total bytes of cached embedding files (excludes the manifest)."""
        if not self.dir.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.dir.glob(f"*{_ARRAY_SUFFIX}"))
