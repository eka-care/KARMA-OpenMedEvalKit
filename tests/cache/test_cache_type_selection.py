import pytest

from karma.cache.cache_manager import _create_cache_io
from karma.cache.duckdb_cache_io import DuckDBCacheIO


def test_create_cache_io_duckdb(tmp_path, monkeypatch):
    monkeypatch.delenv("KARMA_CACHE_TYPE", raising=False)
    db_path = tmp_path / "test.duckdb"

    cache_io = _create_cache_io(cache_type="duckdb", cache_path=str(db_path))

    assert isinstance(cache_io, DuckDBCacheIO)
    cache_io.close_connections()


def test_create_cache_io_dynamodb(monkeypatch):
    monkeypatch.setenv("KARMA_CACHE_TYPE", "dynamodb")
    sentinel = object()
    monkeypatch.setattr(
        "karma.cache.cache_manager.DynamoDBCacheIO",
        lambda *args, **kwargs: sentinel,
    )

    cache_io = _create_cache_io()

    assert cache_io is sentinel
