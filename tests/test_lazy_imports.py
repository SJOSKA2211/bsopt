"""
Test suite for lazy import functionality.
"""
import pytest
import sys
import threading
import time
from src.utils.lazy_import import (
    reset_import_stats, 
    get_import_stats, 
    lazy_import, 
    LazyImportError, 
    CircularImportError,
    preload_modules
)

class TestLazyImports:
    """Test lazy import behavior."""

    def setup_method(self):
        """Reset state before each test."""
        reset_import_stats()
        # Clear any cached imports from previous tests if they exist
        modules_to_clear = [
            mod for mod in sys.modules.keys() 
            if mod.startswith('mock_package')
        ]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]

    def test_lazy_import_success(self, tmp_path, monkeypatch):
        """Verify successful lazy import."""
        # Create a mock package
        pkg_dir = tmp_path / "mock_package"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "import sys\n"
            "from src.utils.lazy_import import lazy_import\n"
            "_import_map = {'MyClass': '.submodule'}\n"
            "def __getattr__(name): return lazy_import(__name__, _import_map, name, sys.modules[__name__])"
        )
        (pkg_dir / "submodule.py").write_text("class MyClass: pass")
        
        monkeypatch.syspath_prepend(str(tmp_path))
        
        import mock_package
        assert 'mock_package.submodule' not in sys.modules
        
        obj = mock_package.MyClass
        assert obj.__name__ == 'MyClass'
        assert 'mock_package.submodule' in sys.modules
        
        stats = get_import_stats()
        assert stats['successful_imports'] == 1

    def test_attribute_error_for_nonexistent(self, tmp_path, monkeypatch):
        """Verify proper error for non-existent attributes."""
        pkg_dir = tmp_path / "mock_package_err"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "import sys\n"
            "from src.utils.lazy_import import lazy_import\n"
            "_import_map = {'Valid': '.submodule'}\n"
            "def __getattr__(name): return lazy_import(__name__, _import_map, name, sys.modules[__name__])"
        )
        
        monkeypatch.syspath_prepend(str(tmp_path))
        import mock_package_err
        
        with pytest.raises(AttributeError, match="has no attribute 'NonExistent'"):
            _ = mock_package_err.NonExistent

    def test_circular_import_detection(self, tmp_path, monkeypatch):
        """Verify circular import detection."""
        pkg_dir = tmp_path / "mock_package_circ"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "import sys\n"
            "from src.utils.lazy_import import lazy_import\n"
            "_import_map = {'A': '.a', 'B': '.b'}\n"
            "def __getattr__(name): return lazy_import(__name__, _import_map, name, sys.modules[__name__])"
        )
        (pkg_dir / "a.py").write_text("from mock_package_circ import B\nclass A: pass")
        (pkg_dir / "b.py").write_text("from mock_package_circ import A\nclass B: pass")
        
        monkeypatch.syspath_prepend(str(tmp_path))
        import mock_package_circ
        
        with pytest.raises(CircularImportError):
            _ = mock_package_circ.A

    def test_failed_import_cached(self, tmp_path, monkeypatch):
        """Verify failed imports are cached."""
        pkg_dir = tmp_path / "mock_package_fail"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "import sys\n"
            "from src.utils.lazy_import import lazy_import\n"
            "_import_map = {'Broken': '.nonexistent'}\n"
            "def __getattr__(name): return lazy_import(__name__, _import_map, name, sys.modules[__name__])"
        )
        
        monkeypatch.syspath_prepend(str(tmp_path))
        import mock_package_fail
        
        with pytest.raises(LazyImportError):
            _ = mock_package_fail.Broken
            
        # Second attempt should be near instant and still fail
        start = time.perf_counter()
        with pytest.raises(LazyImportError):
            _ = mock_package_fail.Broken
        assert time.perf_counter() - start < 0.01
        
        stats = get_import_stats()
        assert stats['failed_imports'] == 1

    def test_thread_safety(self, tmp_path, monkeypatch):
        """Verify thread-safe imports."""
        pkg_dir = tmp_path / "mock_package_thread"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "import sys\n"
            "from src.utils.lazy_import import lazy_import\n"
            "_import_map = {'Slow': '.slow'}\n"
            "def __getattr__(name): return lazy_import(__name__, _import_map, name, sys.modules[__name__])"
        )
        (pkg_dir / "slow.py").write_text("import time\ntime.sleep(0.1)\nclass Slow: pass")
        
        monkeypatch.syspath_prepend(str(tmp_path))
        import mock_package_thread
        
        results = []
        def do_import():
            try:
                results.append(mock_package_thread.Slow)
            except Exception:
                pass
            
        threads = [threading.Thread(target=do_import) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        
        assert len(results) == 5
        assert all(r is results[0] for r in results)

    def test_preload_modules(self, tmp_path, monkeypatch):
        """Verify preloading functionality."""
        pkg_dir = tmp_path / "mock_package_preload"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text(
            "import sys\n"
            "from src.utils.lazy_import import lazy_import, preload_modules\n"
            "_import_map = {'C1': '.c1', 'C2': '.c2'}\n"
            "def __getattr__(name): return lazy_import(__name__, _import_map, name, sys.modules[__name__])\n"
            "def preload(): preload_modules(__name__, _import_map, list(_import_map.keys()))"
        )
        (pkg_dir / "c1.py").write_text("class C1: pass")
        (pkg_dir / "c2.py").write_text("class C2: pass")
        
        monkeypatch.syspath_prepend(str(tmp_path))
        import mock_package_preload
        
        # Initially not loaded
        assert 'mock_package_preload.c1' not in sys.modules
        assert 'mock_package_preload.c2' not in sys.modules
        
        mock_package_preload.preload()
        
        # Now they should be in sys.modules
        assert 'mock_package_preload.c1' in sys.modules
        assert 'mock_package_preload.c2' in sys.modules
        # And cached in the module
        assert 'C1' in mock_package_preload.__dict__
        assert 'C2' in mock_package_preload.__dict__