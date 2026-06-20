# Unit tests for the on-disk embedding cache (task M05, issue #23).

import json

import numpy as np
import pytest

from primed_ai.embeddings.cache import EmbeddingCache


def _cache(tmp_path, modality="echo", version="enc-v1"):
    return EmbeddingCache(tmp_path / "embeddings", modality, encoder_version=version)


def test_write_read_roundtrip_preserves_values_shape_dtype(tmp_path):
    cache = _cache(tmp_path)
    emb = np.random.default_rng(0).standard_normal((16, 1024)).astype(np.float32)
    cache.write("101", emb, timestamp="2026-06-19T00:00:00+00:00")
    out = np.asarray(cache.read("101"))
    assert np.array_equal(out, emb)
    assert out.shape == (16, 1024)
    assert out.dtype == np.float32


def test_has_supports_resume(tmp_path):
    cache = _cache(tmp_path)
    assert not cache.has("7")
    cache.write("7", np.ones(8, dtype=np.float32), timestamp="t")
    assert cache.has("7")
    assert "7" in cache


def test_read_missing_raises_keyerror(tmp_path):
    with pytest.raises(KeyError):
        _cache(tmp_path).read("nope")


def test_write_no_overwrite_then_overwrite(tmp_path):
    cache = _cache(tmp_path)
    cache.write("1", np.zeros(4, dtype=np.float32), timestamp="t")
    with pytest.raises(FileExistsError):
        cache.write("1", np.ones(4, dtype=np.float32), timestamp="t")
    cache.write("1", np.ones(4, dtype=np.float32), overwrite=True, timestamp="t")
    assert np.array_equal(np.asarray(cache.read("1")), np.ones(4, dtype=np.float32))


def test_keys_len_iter_sorted(tmp_path):
    cache = _cache(tmp_path)
    for i in ["3", "1", "2"]:
        cache.write(i, np.zeros(2, dtype=np.float32), timestamp="t")
    assert cache.keys() == ["1", "2", "3"]
    assert len(cache) == 3
    assert sorted(iter(cache)) == ["1", "2", "3"]


def test_manifest_records_metadata(tmp_path):
    cache = _cache(tmp_path, version="echojepa-l-abc123")
    cache.write("42", np.zeros((4, 8), dtype=np.float32),
                timestamp="2026-06-19T12:00:00+00:00")
    lines = cache.manifest_path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "id": "42",
        "modality": "echo",
        "encoder_version": "echojepa-l-abc123",
        "timestamp": "2026-06-19T12:00:00+00:00",
        "shape": [4, 8],
        "dtype": "float32",
        "path": "42.npy",
    }


def test_modalities_are_isolated(tmp_path):
    root = tmp_path / "embeddings"
    echo = EmbeddingCache(root, "echo", "v")
    ecg = EmbeddingCache(root, "ecg", "v")
    echo.write("1", np.full(3, 1.0, dtype=np.float32), timestamp="t")
    ecg.write("1", np.full(3, 2.0, dtype=np.float32), timestamp="t")
    assert np.array_equal(np.asarray(echo.read("1")), np.full(3, 1.0, dtype=np.float32))
    assert np.array_equal(np.asarray(ecg.read("1")), np.full(3, 2.0, dtype=np.float32))
    assert "1" not in EmbeddingCache(root, "other", "v")


def test_mmap_and_inmemory_reads_match(tmp_path):
    cache = _cache(tmp_path)
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    cache.write("m", emb, timestamp="t")
    assert np.array_equal(np.asarray(cache.read("m", mmap=True)), emb)
    assert np.array_equal(cache.read("m", mmap=False), emb)


def test_atomic_write_leaves_no_tmp_files(tmp_path):
    cache = _cache(tmp_path)
    cache.write("1", np.zeros(4, dtype=np.float32), timestamp="t")
    assert list(cache.dir.glob("*.tmp")) == []
    assert cache.keys() == ["1"]  # tmp + manifest excluded from keys


def test_disk_usage_grows_with_writes(tmp_path):
    cache = _cache(tmp_path)
    assert cache.disk_usage_bytes() == 0
    cache.write("1", np.zeros(256, dtype=np.float32), timestamp="t")
    assert cache.disk_usage_bytes() > 0


def test_unsafe_modality_and_id_rejected(tmp_path):
    with pytest.raises(ValueError):
        EmbeddingCache(tmp_path, "a/b")
    cache = _cache(tmp_path)
    with pytest.raises(ValueError):
        cache.write("../evil", np.zeros(2, dtype=np.float32), timestamp="t")
