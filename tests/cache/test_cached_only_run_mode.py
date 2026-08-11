"""Cached-only replay: gap beyond known model errors → dev downgrade."""

from types import SimpleNamespace

from karma.benchmark import Benchmark


def _bench(known_errors, getter_raises=False, has_getter=True):
    bench = object.__new__(Benchmark)
    bench.logger = SimpleNamespace(exception=lambda *a, **k: None)

    if has_getter:
        def getter(config_hash):
            if getter_raises:
                raise RuntimeError("db down")
            return known_errors

        cache_io = SimpleNamespace(get_completed_run_error_count=getter)
    else:
        cache_io = SimpleNamespace()
    bench.cache_manager = SimpleNamespace(cache_io=cache_io, config_hash="abc")
    return bench


def test_gap_within_known_model_errors_keeps_run_mode():
    assert _bench(known_errors=42)._replay_gap_beyond_known_errors(42) is False
    assert _bench(known_errors=42)._replay_gap_beyond_known_errors(10) is False


def test_gap_beyond_known_model_errors_downgrades():
    assert _bench(known_errors=42)._replay_gap_beyond_known_errors(43) is True
    assert _bench(known_errors=0)._replay_gap_beyond_known_errors(1) is True


def test_no_completed_run_downgrades():
    assert _bench(known_errors=None)._replay_gap_beyond_known_errors(1) is True


def test_backend_without_getter_downgrades():
    assert _bench(None, has_getter=False)._replay_gap_beyond_known_errors(1) is True


def test_getter_failure_downgrades():
    assert _bench(42, getter_raises=True)._replay_gap_beyond_known_errors(1) is True
