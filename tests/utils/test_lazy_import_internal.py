import pytest
import threading
import time
import os
import sys
import importlib
from unittest.mock import MagicMock, patch
import src.utils.lazy_import
from src.utils.lazy_import import (
    _get_import_lock, 
    _track_import_stack, 
    _import_times, 
    _failed_imports,
    _import_stack,
    LazyImportError,
    CircularImportError,
    get_import_stats,
    reset_import_stats
)
import structlog
import io
from contextlib import redirect_stdout
import re
from typing import TYPE_CHECKING
import builtins

# Save original import_module
original_import_module = importlib.import_module

def test_get_import_lock_concurrency():
    lock1 = _get_import_lock("mod1")
    lock2 = _get_import_lock("mod1")
    assert lock1 is lock2
    
    lock3 = _get_import_lock("mod2")
    assert lock1 is not lock3

def test_circular_import_detection():
    if not hasattr(_import_stack, 'modules'):
        _import_stack.modules = []
    _import_stack.modules = []
        
    with _track_import_stack("mod_a"):
        with _track_import_stack("mod_b"):
            with pytest.raises(CircularImportError) as excinfo:
                with _track_import_stack("mod_a"):
                    pass
            assert "Circular import detected: mod_a" in str(excinfo.value)

def test_lazy_import_invalid_attribute():
    import_map = {"valid": "path"}
    with pytest.raises(AttributeError) as excinfo:
        src.utils.lazy_import.lazy_import("pkg", import_map, "invalid", MagicMock())
    assert "has no attribute 'invalid'" in str(excinfo.value)

def test_lazy_import_previously_failed():
    _failed_imports["pkg.fail"] = Exception("Boom")
    import_map = {"fail": "path"}
    with pytest.raises(LazyImportError) as excinfo:
        src.utils.lazy_import.lazy_import("pkg", import_map, "fail", MagicMock())
    assert "Previous import of pkg.fail failed" in str(excinfo.value)
    del _failed_imports["pkg.fail"]

def test_lazy_import_success():
    reset_import_stats()
    import_map = {"name": "os"}
    
    class MockCache:
        def __init__(self):
            self.__dict__ = {}
    
    cache = MockCache()
    res = src.utils.lazy_import.lazy_import("os", import_map, "name", cache)
    
    import os
    assert res is os.name
    assert cache.name is os.name
    
    stats = get_import_stats()
    assert stats['successful_imports'] >= 1

def test_lazy_import_module_attribute_error():
    import_map = {"NON_EXISTENT_ATTR": "os"}
    cache = MagicMock()
    
    with pytest.raises(LazyImportError) as excinfo:
        src.utils.lazy_import.lazy_import("src", import_map, "NON_EXISTENT_ATTR", cache)
    assert "Failed to import NON_EXISTENT_ATTR" in str(excinfo.value)

def test_preload_modules():
    import_map = {"name": "os"}
    class RealObject: pass
    obj = RealObject()
    src.utils.lazy_import.preload_modules("os", import_map, ["name"], cache_module_override=obj)
    assert hasattr(obj, "name")
    import os
    assert obj.name is os.name

def test_get_import_stats_complex():
    reset_import_stats()
    _import_times["a"] = 0.5
    _import_times["b"] = 0.1
    _failed_imports["c"] = Exception("Error")
    
    stats = get_import_stats()
    assert stats['successful_imports'] == 2
    assert stats['failed_imports'] == 1
    assert stats['slowest_imports'][0][0] == "a"
    assert "c" in stats['failures']

def test_lazy_import_double_check():
    import_map = {"name": "os"}
    class MockCache: pass
    cache = MockCache()
    cache.name = "already_here"
    
    res = src.utils.lazy_import.lazy_import("os", import_map, "name", cache)
    assert res == "already_here"

def test_lazy_import_circular_error_propagation(mocker):
    import_map = {"a": "my_circular_module"} 
    package_name = "pkg"
    expected_full_module_path = "my_circular_module"

    if not hasattr(_import_stack, 'modules'):
        _import_stack.modules = []
    
    mocker.patch.object(_import_stack, 'modules', [expected_full_module_path]) 
    
    with pytest.raises(CircularImportError) as excinfo:
        src.utils.lazy_import.lazy_import(package_name, import_map, "a", MagicMock())
    assert "Circular import detected" in str(excinfo.value)

def test_ml_init_logic():
    import src.ml
    # Patch in src.utils.lazy_import so that reload gets the mock
    with patch('src.utils.lazy_import.preload_modules') as mock_preload:
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "PRELOAD_ML_MODULES": "true"}):
            importlib.reload(src.ml)
            # The reload will have called src.ml.preload_critical_modules()
            # which calls preload_modules (the mock)
            assert mock_preload.called

def test_import_stack_cleanup_on_error():
    if not hasattr(_import_stack, 'stack'):
        _import_stack.stack = []
    _import_stack.stack = []
    
    try:
        with _track_import_stack("bad"):
            raise ValueError("Boom")
    except ValueError:
        pass
    assert "bad" not in _import_stack.stack

def test_track_import_stack_cleanup():
    if not hasattr(_import_stack, 'modules'):
        _import_stack.modules = []
    _import_stack.modules.clear()

    initial_len = len(_import_stack.modules)
    with _track_import_stack("temp_module"):
        assert len(_import_stack.modules) == initial_len + 1
    assert len(_import_stack.modules) == initial_len

def test_preload_modules_failure_logging(mocker):
    reset_import_stats()
    
    mock_logger = MagicMock()
    mocker.patch.object(src.utils.lazy_import, 'logger', mock_logger)
    
    def side_effect(name, package=None):
        if name == 'bad_module':
            raise ModuleNotFoundError("No module named 'bad_module'")
        return original_import_module(name, package)

    mocker.patch('src.utils.lazy_import.importlib.import_module', side_effect=side_effect)
    
    package_name = "test_pkg"
    import_map = {"bad_attr": "bad_module"}
    attributes = ["bad_attr"]
    mock_cache_module = MagicMock()

    src.utils.lazy_import.preload_modules(package_name, import_map, attributes, cache_module_override=mock_cache_module)
    
    assert mock_logger.warning.called
    _, kwargs = mock_logger.warning.call_args
    assert kwargs['package'] == package_name
    assert "No module named 'bad_module'" in str(kwargs['error'])

def test_ml_getattr(mocker):
    expected_returned_object = MagicMock(name="MockedDataNormalizerInstance")
    import src.ml
    # Ensure fresh state
    if 'DataNormalizer' in src.ml.__dict__:
        del src.ml.DataNormalizer
        
    mock_lazy_import = mocker.patch('src.ml.lazy_import', return_value=expected_returned_object)

    # Trigger __getattr__
    result = src.ml.DataNormalizer
    
    mock_lazy_import.assert_called_once()
    assert result is expected_returned_object
    
def test_ml_get_ml_import_stats():
    import src.ml
    stats = src.ml.get_ml_import_stats()
    assert isinstance(stats, dict)

def test_ml_type_checking_imports_with_dummy_module(tmp_path, mocker):
    dummy_ml_path = tmp_path / "dummy_ml_module.py"
    dummy_ml_content = """
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ml.forecasting.tft_model import PriceTFTModel, TFTModel
    from src.ml.reinforcement_learning.trading_env import TradingEnvironment
    from src.ml.rl.augmented_agent import AugmentedRLAgent
    from src.ml.federated_learning.coordinator import FederatedLearningCoordinator
    from src.ml.data_loader import DataNormalizer

DUMMY_VAR = True
    """
    dummy_ml_path.write_text(dummy_ml_content)

    sys.path.insert(0, str(tmp_path))
    original_sys_modules = sys.modules.copy()
    mocker.patch('typing.TYPE_CHECKING', True)

    attempted_imports = set()
    original_builtin_import = builtins.__import__

    def mock_builtin_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'dummy_ml_module':
            return original_builtin_import(name, globals, locals, fromlist, level)
        if name.startswith('src.ml.'):
            attempted_imports.add(name)
            return MagicMock()
        if name in ['sys', 'typing', 'os', 'builtins', 'importlib']:
             return original_builtin_import(name, globals, locals, fromlist, level)
        return MagicMock()

    mocker.patch('builtins.__import__', side_effect=mock_builtin_import)

    try:
        import dummy_ml_module
        assert 'src.ml.forecasting.tft_model' in attempted_imports
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.clear()
        sys.modules.update(original_sys_modules)

def test_pricing_getattr(mocker):
    expected_returned_object = MagicMock()
    import src.pricing
    if 'BlackScholesEngine' in src.pricing.__dict__:
        del src.pricing.BlackScholesEngine
    mock_lazy_import = mocker.patch('src.pricing.lazy_import', return_value=expected_returned_object)
    result = src.pricing.BlackScholesEngine
    assert result is expected_returned_object

def test_pricing_dir():
    import src.pricing
    assert sorted(src.pricing.__all__) == sorted(dir(src.pricing))

def test_preload_classical_pricers(mocker):
    import src.pricing
    with patch('src.utils.lazy_import.preload_modules') as mock_preload:
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "PRELOAD_PRICING": "true"}):
            importlib.reload(src.pricing)
            assert mock_preload.called

def test_pricing_type_checking_imports_with_dummy_module(tmp_path, mocker):
    dummy_pricing_path = tmp_path / "dummy_pricing_module.py"
    dummy_pricing_content = """
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.pricing.black_scholes import BlackScholesEngine
DUMMY_VAR = True
    """
    dummy_pricing_path.write_text(dummy_pricing_content)
    sys.path.insert(0, str(tmp_path))
    original_sys_modules = sys.modules.copy()
    mocker.patch('typing.TYPE_CHECKING', True)
    attempted_imports = set()
    original_builtin_import = builtins.__import__
    def mock_builtin_import(name, *args, **kwargs):
        if name.startswith('src.pricing.'): attempted_imports.add(name); return MagicMock()
        return original_builtin_import(name, *args, **kwargs)
    mocker.patch('builtins.__import__', side_effect=mock_builtin_import)
    try:
        import dummy_pricing_module
        assert 'src.pricing.black_scholes' in attempted_imports
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.clear(); sys.modules.update(original_sys_modules)

def test_streaming_getattr(mocker):
    expected = MagicMock()
    import src.streaming
    if 'MarketDataProducer' in src.streaming.__dict__:
        del src.streaming.MarketDataProducer
    mocker.patch('src.streaming.lazy_import', return_value=expected)
    assert src.streaming.MarketDataProducer is expected

def test_streaming_dir():
    import src.streaming
    assert sorted(src.streaming.__all__) == sorted(dir(src.streaming))

def test_preload_streaming_modules(mocker):
    import src.streaming
    mock_preload = mocker.patch('src.streaming.preload_modules')
    src.streaming.preload_streaming_modules()
    mock_preload.assert_called()

def test_streaming_type_checking_imports_with_dummy_module(tmp_path, mocker):
    dummy_path = tmp_path / "dummy_streaming_module.py"
    dummy_content = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from src.streaming.kafka_producer import MarketDataProducer\nDUMMY_VAR = True"
    dummy_path.write_text(dummy_content)
    sys.path.insert(0, str(tmp_path))
    original_sys_modules = sys.modules.copy()
    mocker.patch('typing.TYPE_CHECKING', True)
    attempted = set()
    orig_import = builtins.__import__
    def mock_imp(name, *args, **kwargs):
        if name.startswith('src.streaming.'): attempted.add(name); return MagicMock()
        return orig_import(name, *args, **kwargs)
    mocker.patch('builtins.__import__', side_effect=mock_imp)
    try:
        import dummy_streaming_module
        assert 'src.streaming.kafka_producer' in attempted
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.clear(); sys.modules.update(original_sys_modules)

def test_blockchain_getattr(mocker):
    expected = MagicMock()
    import src.blockchain
    if 'DeFiOptionsProtocol' in src.blockchain.__dict__:
        del src.blockchain.DeFiOptionsProtocol
    mocker.patch('src.blockchain.lazy_import', return_value=expected)
    assert src.blockchain.DeFiOptionsProtocol is expected

def test_blockchain_dir():
    import src.blockchain
    assert sorted(src.blockchain.__all__) == sorted(dir(src.blockchain))

def test_blockchain_type_checking_imports_with_dummy_module(tmp_path, mocker):
    dummy_path = tmp_path / "dummy_blockchain_module.py"
    dummy_content = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from src.blockchain.defi_options import DeFiOptionsProtocol\nDUMMY_VAR = True"
    dummy_path.write_text(dummy_content)
    sys.path.insert(0, str(tmp_path))
    original_sys_modules = sys.modules.copy()
    mocker.patch('typing.TYPE_CHECKING', True)
    attempted = set()
    orig_import = builtins.__import__
    def mock_imp(name, *args, **kwargs):
        if name.startswith('src.blockchain.'): attempted.add(name); return MagicMock()
        return orig_import(name, *args, **kwargs)
    mocker.patch('builtins.__import__', side_effect=mock_imp)
    try:
        import dummy_blockchain_module
        assert 'src.blockchain.defi_options' in attempted
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.clear(); sys.modules.update(original_sys_modules)

def test_pricing_init(tmp_path, mocker):
    pkg_name = "my_pricing_pkg_4"
    pkg_path = tmp_path / pkg_name
    pkg_path.mkdir()
    (pkg_path / "__init__.py").write_text(
        "import sys\n"
        "from src.utils.lazy_import import lazy_import\n"
        "_import_map = {'BlackScholesEngine': '.black_scholes'}\n"
        "def __getattr__(name): return lazy_import(__name__, _import_map, name, sys.modules[__name__])\n"
    )
    (pkg_path / "black_scholes.py").write_text("class BlackScholesEngine: pass")
    def side_effect(name, package=None):
        if name == f'{pkg_name}.black_scholes': return MagicMock()
        return original_import_module(name, package)
    mocker.patch('src.utils.lazy_import.importlib.import_module', side_effect=side_effect)
    sys.path.insert(0, str(tmp_path))
    try:
        my_pkg = importlib.import_module(pkg_name)
        assert hasattr(my_pkg, "BlackScholesEngine")
    finally:
        sys.path.remove(str(tmp_path))
        if pkg_name in sys.modules: del sys.modules[pkg_name]

def test_lazy_import_full_module_path_starts_with_dot(tmp_path, mocker):
    pkg_name = "my_package_temp_4"
    pkg_path = tmp_path / pkg_name
    pkg_path.mkdir()
    (pkg_path / "__init__.py").write_text(
        f"import sys\nfrom src.utils.lazy_import import lazy_import\n_import_map={{'my_attr':'.submodule'}}\n"
        f"def __getattr__(name): return lazy_import('{pkg_name}', _import_map, name, sys.modules[__name__])\n"
    )
    (pkg_path / "submodule.py").write_text("my_attr = 'mocked_relative_attr'")
    def side_effect(name, package=None):
        if name == f'{pkg_name}.submodule': 
            m = MagicMock(); m.my_attr = 'mocked_relative_attr'; return m
        return original_import_module(name, package)
    mocker.patch('src.utils.lazy_import.importlib.import_module', side_effect=side_effect)
    sys.path.insert(0, str(tmp_path))
    try:
        my_pkg = importlib.import_module(pkg_name)
        res = src.utils.lazy_import.lazy_import(pkg_name, {"my_attr":".submodule"}, "my_attr", my_pkg)
        assert res == "mocked_relative_attr"
    finally:
        sys.path.remove(str(tmp_path))
        if pkg_name in sys.modules: del sys.modules[pkg_name]
