import threading
from unittest.mock import MagicMock

import pytest

from src.shared import lazy_import


def test_lazy_import_success():
    # Reset stats
    lazy_import.reset_import_stats()

    mock_module = MagicMock()
    # Mocking os.path.join as our target
    import_map = {"join": "os.path"}

    res = lazy_import.lazy_import("os", import_map, "join", mock_module)

    from os.path import join

    assert res == join
    assert mock_module.join == join

    stats = lazy_import.get_import_stats()
    assert stats["successful_imports"] == 1


def test_lazy_import_missing_attr():
    mock_module = MagicMock()
    import_map = {"exists": "os.path"}

    with pytest.raises(AttributeError):
        lazy_import.lazy_import("os", import_map, "missing", mock_module)


def test_lazy_import_failure_caching():
    lazy_import.reset_import_stats()
    mock_module = MagicMock()
    import_map = {"fail": "non_existent_module_engineer"}

    # First attempt fails
    with pytest.raises(lazy_import.LazyImportError):
        lazy_import.lazy_import("test", import_map, "fail", mock_module)

    # Second attempt fails immediately from cache
    with pytest.raises(lazy_import.LazyImportError) as excinfo:
        lazy_import.lazy_import("test", import_map, "fail", mock_module)
    assert "Previous import" in str(excinfo.value)


def test_circular_import_detection():
    # Manually trigger the circular stack
    with lazy_import._track_import_stack("module_a"):
        with pytest.raises(lazy_import.CircularImportError):
            with lazy_import._track_import_stack("module_a"):
                pass


def test_preload_modules():
    lazy_import.reset_import_stats()
    mock_module = MagicMock()
    import_map = {"path": "os"}

    lazy_import.preload_modules("os", import_map, ["path"], cache_module_override=mock_module)
    assert hasattr(mock_module, "path")


def test_thread_safe_import():
    lazy_import.reset_import_stats()
    mock_module = MagicMock()
    import_map = {"sep": "os"}

    def worker():
        for _ in range(10):
            lazy_import.lazy_import("os", import_map, "sep", mock_module)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    from os import sep

    assert mock_module.sep == sep
