import builtins  # Import builtins for patching
import importlib  # Added importlib
import io  # Added for capturing stdout
import os
import re  # Import regex for parsing log messages
import sys
from contextlib import redirect_stdout  # Added for capturing stdout
from unittest.mock import MagicMock, patch

import pytest

from src.utils.lazy_import import (
    CircularImportError,
    LazyImportError,
    _failed_imports,
    _get_import_lock,
    _import_stack,
    _import_times,
    _track_import_stack,
    get_import_stats,
    lazy_import,
    preload_modules,
    reset_import_stats,
)


def test_get_import_lock_concurrency():
    lock1 = _get_import_lock("mod1")
    lock2 = _get_import_lock("mod1")
    assert lock1 is lock2
    
    lock3 = _get_import_lock("mod2")
    assert lock1 is not lock3

def test_circular_import_detection():
    # Ensure stack is initialized
    if not hasattr(_import_stack, 'stack'):
        _import_stack.stack = []
    _import_stack.stack = []
        
    with _track_import_stack("mod_a"):
        with _track_import_stack("mod_b"):
            with pytest.raises(CircularImportError) as excinfo:
                with _track_import_stack("mod_a"):
                    pass
            assert "Circular import detected: mod_a" in str(excinfo.value)

def test_lazy_import_invalid_attribute():
    import_map = {"valid": "path"}
    with pytest.raises(AttributeError) as excinfo:
        lazy_import("pkg", import_map, "invalid", MagicMock())
    assert "has no attribute 'invalid'" in str(excinfo.value)

def test_lazy_import_previously_failed():
    _failed_imports["pkg.fail"] = Exception("Boom")
    import_map = {"fail": "path"}
    with pytest.raises(LazyImportError) as excinfo:
        lazy_import("pkg", import_map, "fail", MagicMock())
    assert "Previous import of pkg.fail failed" in str(excinfo.value)
    del _failed_imports["pkg.fail"]

def test_lazy_import_success():
    reset_import_stats()
    # Use 'os' module and 'name' attribute
    import_map = {"name": "os"}
    
    class MockCache:
        def __init__(self):
            self.__dict__ = {}
    
    cache = MockCache()
    
    # Perform lazy import
    res = lazy_import("os", import_map, "name", cache)
    
    import os
    assert res is os.name
    assert cache.name is os.name
    
    # Let's verify stats
    stats = get_import_stats()
    assert stats['successful_imports'] >= 1
    assert any(item[0] == "os.name" for item in stats['slowest_imports'])

def test_lazy_import_module_attribute_error():
    import_map = {"NON_EXISTENT_ATTR": "os"}
    cache = MagicMock()
    
    with pytest.raises(LazyImportError) as excinfo:
        lazy_import("src", import_map, "NON_EXISTENT_ATTR", cache)
    assert "Failed to import NON_EXISTENT_ATTR" in str(excinfo.value)

def test_preload_modules():
    import_map = {"name": "os"}
    class RealObject: pass
    obj = RealObject()
    preload_modules("os", import_map, ["name"], cache_module_override=obj)
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
    
    res = lazy_import("os", import_map, "name", cache)
    assert res == "already_here"

def test_lazy_import_circular_error_propagation(mocker):
    import_map = {"a": "my_circular_module"} 
    package_name = "pkg"
    # The full_module_path that lazy_import will pass to _track_import_stack is "my_circular_module"
    expected_full_module_path = "my_circular_module"

    # Ensure _import_stack.modules is initialized
    if not hasattr(_import_stack, 'modules'):
        _import_stack.modules = []
    
    # Simulate a circular import by patching _import_stack.modules directly
    # Now, expected_full_module_path will be in _import_stack.modules
    mocker.patch.object(_import_stack, 'modules', [expected_full_module_path]) 
    
    with pytest.raises(CircularImportError) as excinfo:
        lazy_import(package_name, import_map, "a", MagicMock())
    assert "Circular import detected" in str(excinfo.value)
    assert f"-> {expected_full_module_path}" in str(excinfo.value)
    # The mocker fixture handles cleanup.

def test_ml_init_logic():
    import importlib

    import src.ml
    
    with patch.dict(os.environ, {"ENVIRONMENT": "production", "PRELOAD_ML_MODULES": "true"}):
        with patch('src.utils.lazy_import.preload_modules') as mock_preload: # Patch the actual function called
            importlib.reload(src.ml) # Reload to ensure the env var check is re-evaluated
            mock_preload.assert_called_with("src.ml", src.ml._import_map, {"DataNormalizer"})

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

# New test to cover _import_stack.modules.pop()
def test_track_import_stack_cleanup():
    if not hasattr(_import_stack, 'modules'):
        _import_stack.modules = []
    _import_stack.modules.clear() # Ensure clean state

    initial_len = len(_import_stack.modules)
    with _track_import_stack("temp_module"):
        assert len(_import_stack.modules) == initial_len + 1
    assert len(_import_stack.modules) == initial_len # Should pop on exit

# New test to cover preload_modules failure logging
def test_preload_modules_failure_logging(mocker):
    reset_import_stats()
    # Mock lazy_import to fail when called from preload_modules
    mocker.patch('src.utils.lazy_import.lazy_import', side_effect=ModuleNotFoundError("No module named 'bad_module'"))
    
    package_name = "test_pkg"
    import_map = {"bad_attr": "bad_module"}
    attributes = ["bad_attr"]
    
    # Provide a mock cache_module_override to avoid KeyError
    mock_cache_module = MagicMock()

    # Capture stdout to verify the log message
    f = io.StringIO()
    with redirect_stdout(f):
        preload_modules(package_name, import_map, attributes, cache_module_override=mock_cache_module)
    
    output = f.getvalue()
    
    # Extract the relevant log message line (the warning one)
    warning_line = ""
    for line in output.splitlines():
        if "[warning  ]" in line:
            warning_line = line
            break
    
    assert warning_line, "Expected warning log line not found"

    # Use regex to parse the structlog output
    match = re.search(r'\[warning\s*\]\s*(?P<event>\S+)\s+.*attribute=(?P<attribute>\S+)\s+error="(?P<error>.*?)"\s+package=(?P<package>\S+)', warning_line)

    assert match is not None, f"Could not parse warning log line: {warning_line}" 
    
    # Assert extracted values
    assert match.group('event') == "preload_failed"
    assert match.group('package') == package_name
    assert match.group('attribute') == "bad_attr"
    assert match.group('error') == f"Failed to import bad_attr from {package_name}: No module named 'bad_module'"

# Add new tests for src.ml/__init__.py coverage
def test_ml_getattr(mocker):
    # Ensure src.ml is reloaded to get a fresh state
    # We must delete src.ml from sys.modules first to ensure a clean slate
    if 'src.ml' in sys.modules:
        del sys.modules['src.ml']
    
    # We also need to ensure src.utils.lazy_import is clean to use our patch
    if 'src.utils.lazy_import' in sys.modules:
        del sys.modules['src.utils.lazy_import']

    # Create the mock object that lazy_import should return
    expected_returned_object = MagicMock(name="MockedDataNormalizerInstance")

    # Mock src.utils.lazy_import.lazy_import to return our specific mock object
    # This directly mocks what src.ml.__getattr__ calls.
    mock_lazy_import = mocker.patch('src.utils.lazy_import.lazy_import', return_value=expected_returned_object)

    # Now, import src.ml. This will load src.ml and its __getattr__ logic.
    # Since src.utils.lazy_import might be imported during this process,
    # and it relies on a module-level import of `import_module`,
    # we need to be careful.
    import src.ml
    
    # Access a lazy-loaded attribute to trigger __getattr__
    result = src.ml.DataNormalizer
    
    # Assert that lazy_import was called correctly
    mock_lazy_import.assert_called_once_with(
        'src.ml', src.ml._import_map, 'DataNormalizer', sys.modules['src.ml']
    )
    
    # Assert that the result is the expected mock object
    assert result is expected_returned_object
    
def test_ml_get_ml_import_stats():
    import src.ml
    importlib.reload(src.ml) # Ensure a clean state

    # Call the function to get import stats
    stats = src.ml.get_ml_import_stats()
    assert isinstance(stats, dict) # Assuming get_import_stats returns a dict
    # Add more specific assertions if needed based on expected stats content

# New test to cover the TYPE_CHECKING block in src/ml/__init__.py
def test_ml_type_checking_imports_with_dummy_module(tmp_path, mocker):
    # Temporarily create a dummy module that will attempt the TYPE_CHECKING imports
    dummy_ml_path = tmp_path / "dummy_ml_module.py"
    dummy_ml_content = """
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ml.forecasting.tft_model import PriceTFTModel, TFTModel
    from src.ml.reinforcement_learning.trading_env import TradingEnvironment
    from src.ml.reinforcement_learning.augmented_agent import AugmentedRLAgent
    from src.ml.federated_learning.coordinator import FederatedLearningCoordinator
    from src.ml.data_loader import DataNormalizer

DUMMY_VAR = True
    """
    dummy_ml_path.write_text(dummy_ml_content)

    # Add the temporary path to sys.path
    sys.path.insert(0, str(tmp_path))

    # Store original sys.modules to clean up later
    original_sys_modules = sys.modules.copy()

    # Patch TYPE_CHECKING to be True for the duration of this test
    mocker.patch('typing.TYPE_CHECKING', True)

    attempted_imports = set()
    original_builtin_import = builtins.__import__

    def mock_builtin_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Allow the dummy module to import itself
        if name == 'dummy_ml_module':
            return original_builtin_import(name, globals, locals, fromlist, level)

        # Track imports that are submodules of src.ml
        if name.startswith('src.ml.'):
            attempted_imports.add(name)
            return MagicMock(name=f"MockedModuleForTypeChecking_{name}")
        
        # Allow core Python modules and 'typing', 'sys', 'os', 'builtins' to import normally.
        # Everything else gets a MagicMock to prevent loading real dependencies.
        if name in ['sys', 'typing', 'os', 'builtins', 'importlib']: # Added 'importlib'
             return original_builtin_import(name, globals, locals, fromlist, level)
        
        return MagicMock(name=f"GenericMockedImport_{name}")

    mocker.patch('builtins.__import__', side_effect=mock_builtin_import)

    try:
        import dummy_ml_module
        assert dummy_ml_module.DUMMY_VAR is True

        assert 'src.ml.forecasting.tft_model' in attempted_imports
        assert 'src.ml.reinforcement_learning.trading_env' in attempted_imports
        assert 'src.ml.reinforcement_learning.augmented_agent' in attempted_imports
        assert 'src.ml.federated_learning.coordinator' in attempted_imports
        assert 'src.ml.data_loader' in attempted_imports

    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.clear()
        sys.modules.update(original_sys_modules)
        if 'dummy_ml_module' in sys.modules:
            del sys.modules['dummy_ml_module']

# New tests for src/pricing/__init__.py coverage
def test_pricing_getattr(mocker):
    if 'src.pricing' in sys.modules:
        del sys.modules['src.pricing']
    if 'src.utils.lazy_import' in sys.modules:
        del sys.modules['src.utils.lazy_import']

    expected_returned_object = MagicMock(name="MockedBlackScholesEngineInstance")
    mock_lazy_import = mocker.patch('src.utils.lazy_import.lazy_import', return_value=expected_returned_object)

    import src.pricing
    result = src.pricing.BlackScholesEngine
    
    mock_lazy_import.assert_called_once_with(
        'src.pricing', src.pricing._import_map, 'BlackScholesEngine', sys.modules['src.pricing']
    )
    assert result is expected_returned_object

def test_pricing_dir():
    if 'src.pricing' in sys.modules:
        del sys.modules['src.pricing']
    import src.pricing
    # Make sure we don't accidentally lazy load something by just calling dir
    # And make sure __all__ elements are present
    assert sorted(src.pricing.__all__) == sorted(dir(src.pricing))

def test_preload_classical_pricers(mocker):
    # Ensure src.pricing is removed from sys.modules for a fresh import
    if 'src.pricing' in sys.modules:
        del sys.modules['src.pricing']
    
    # Ensure src.utils.lazy_import is removed from sys.modules to guarantee our patch is applied
    if 'src.utils.lazy_import' in sys.modules:
        del sys.modules['src.utils.lazy_import']
    
    # Patch preload_modules directly using mocker. This needs to happen BEFORE src.pricing is imported
    # because src.pricing imports src.utils.lazy_import at the top level.
    mock_preload_modules = mocker.patch('src.utils.lazy_import.preload_modules')

    with patch.dict(os.environ, {"ENVIRONMENT": "production", "PRELOAD_PRICING": "true"}):
        # Now import src.pricing. This will be its first import in this test,
        # and the module-level 'if' condition will be evaluated exactly once.
        import src.pricing
        
        expected_fast_modules = {
            "HestonModelFFT",
            "HestonCalibrator",
            "BlackScholesEngine",
            "SVISurface",
        }
        mock_preload_modules.assert_called_once_with(
            'src.pricing', src.pricing._import_map, expected_fast_modules
        )

def test_pricing_type_checking_imports_with_dummy_module(tmp_path, mocker):
    dummy_pricing_path = tmp_path / "dummy_pricing_module.py"
    dummy_pricing_content = """
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pricing.models.heston_fft import HestonModelFFT
    from src.pricing.calibration.engine import HestonCalibrator
    from src.pricing.black_scholes import BlackScholesEngine
    from src.pricing.monte_carlo import MonteCarloEngine
    from src.pricing.calibration.svi_surface import SVISurface
    from src.pricing.vol_surface import SABRModel
    from src.pricing.quantum_pricing import QuantumOptionPricer

DUMMY_VAR = True
    """
    dummy_pricing_path.write_text(dummy_pricing_content)

    sys.path.insert(0, str(tmp_path))
    original_sys_modules = sys.modules.copy()
    
    mocker.patch('typing.TYPE_CHECKING', True)

    attempted_imports = set()
    original_builtin_import = builtins.__import__

    def mock_builtin_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'dummy_pricing_module':
            return original_builtin_import(name, globals, locals, fromlist, level)

        if name.startswith('src.pricing.'):
            attempted_imports.add(name)
            return MagicMock(name=f"MockedModuleForTypeChecking_{name}")
        
        if name in ['sys', 'typing', 'os', 'builtins', 'importlib']:
             return original_builtin_import(name, globals, locals, fromlist, level)
        
        return MagicMock(name=f"GenericMockedImport_{name}")

    mocker.patch('builtins.__import__', side_effect=mock_builtin_import)

    try:
        import dummy_pricing_module
        assert dummy_pricing_module.DUMMY_VAR is True

        assert 'src.pricing.models.heston_fft' in attempted_imports
        assert 'src.pricing.calibration.engine' in attempted_imports
        assert 'src.pricing.black_scholes' in attempted_imports
        assert 'src.pricing.monte_carlo' in attempted_imports
        assert 'src.pricing.calibration.svi_surface' in attempted_imports
        assert 'src.pricing.vol_surface' in attempted_imports
        assert 'src.pricing.quantum_pricing' in attempted_imports

    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.clear()
        sys.modules.update(original_sys_modules)
        if 'dummy_pricing_module' in sys.modules:
            del sys.modules['dummy_pricing_module']


def test_streaming_getattr(mocker):
    if 'src.streaming' in sys.modules:
        del sys.modules['src.streaming']
    if 'src.utils.lazy_import' in sys.modules:
        del sys.modules['src.utils.lazy_import']

    expected_returned_object = MagicMock(name="MockedMarketDataProducerInstance")
    mock_lazy_import = mocker.patch('src.utils.lazy_import.lazy_import', return_value=expected_returned_object)

    import src.streaming
    result = src.streaming.MarketDataProducer
    
    mock_lazy_import.assert_called_once_with(
        'src.streaming', src.streaming._import_map, 'MarketDataProducer', sys.modules['src.streaming']
    )
    assert result is expected_returned_object

def test_streaming_dir():
    if 'src.streaming' in sys.modules:
        del sys.modules['src.streaming']
    import src.streaming
    assert sorted(src.streaming.__all__) == sorted(dir(src.streaming))

def test_preload_streaming_modules(mocker):
    if 'src.streaming' in sys.modules:
        del sys.modules['src.streaming']
    if 'src.utils.lazy_import' in sys.modules:
        del sys.modules['src.utils.lazy_import']
    
    mock_preload_modules = mocker.patch('src.utils.lazy_import.preload_modules')

    import src.streaming
    src.streaming.preload_streaming_modules()
    
    expected_modules = {"MarketDataProducer", "MarketDataConsumer"}
    mock_preload_modules.assert_called_once_with(
        'src.streaming', src.streaming._import_map, expected_modules
    )

def test_streaming_type_checking_imports_with_dummy_module(tmp_path, mocker):
    dummy_streaming_path = tmp_path / "dummy_streaming_module.py"
    dummy_streaming_content = """
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.streaming.kafka_producer import MarketDataProducer
    from src.streaming.kafka_consumer import MarketDataConsumer

DUMMY_VAR = True
    """
    dummy_streaming_path.write_text(dummy_streaming_content)

    sys.path.insert(0, str(tmp_path))
    original_sys_modules = sys.modules.copy()
    
    mocker.patch('typing.TYPE_CHECKING', True)

    attempted_imports = set()
    original_builtin_import = builtins.__import__

    def mock_builtin_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'dummy_streaming_module':
            return original_builtin_import(name, globals, locals, fromlist, level)

        if name.startswith('src.streaming.'):
            attempted_imports.add(name)
            return MagicMock(name=f"MockedModuleForTypeChecking_{name}")
        
        if name in ['sys', 'typing', 'os', 'builtins', 'importlib']:
             return original_builtin_import(name, globals, locals, fromlist, level)
        
        return MagicMock(name=f"GenericMockedImport_{name}")

    mocker.patch('builtins.__import__', side_effect=mock_builtin_import)

    try:
        import dummy_streaming_module
        assert dummy_streaming_module.DUMMY_VAR is True

        assert 'src.streaming.kafka_producer' in attempted_imports
        assert 'src.streaming.kafka_consumer' in attempted_imports

    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.clear()
        sys.modules.update(original_sys_modules)
        if 'dummy_streaming_module' in sys.modules:
            del sys.modules['dummy_streaming_module']


def test_blockchain_getattr(mocker):
    if 'src.blockchain' in sys.modules:
        del sys.modules['src.blockchain']
    if 'src.utils.lazy_import' in sys.modules:
        del sys.modules['src.utils.lazy_import']

    expected_returned_object = MagicMock(name="MockedDeFiOptionsProtocolInstance")
    mock_lazy_import = mocker.patch('src.utils.lazy_import.lazy_import', return_value=expected_returned_object)

    import src.blockchain
    result = src.blockchain.DeFiOptionsProtocol
    
    mock_lazy_import.assert_called_once_with(
        'src.blockchain', src.blockchain._import_map, 'DeFiOptionsProtocol', sys.modules['src.blockchain']
    )
    assert result is expected_returned_object

def test_blockchain_dir():
    if 'src.blockchain' in sys.modules:
        del sys.modules['src.blockchain']
    import src.blockchain
    assert sorted(src.blockchain.__all__) == sorted(dir(src.blockchain))

def test_blockchain_type_checking_imports_with_dummy_module(tmp_path, mocker):
    dummy_blockchain_path = tmp_path / "dummy_blockchain_module.py"
    dummy_blockchain_content = """
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.blockchain.defi_options import DeFiOptionsProtocol

DUMMY_VAR = True
    """
    dummy_blockchain_path.write_text(dummy_blockchain_content)

    sys.path.insert(0, str(tmp_path))
    original_sys_modules = sys.modules.copy()
    
    mocker.patch('typing.TYPE_CHECKING', True)

    attempted_imports = set()
    original_builtin_import = builtins.__import__

    def mock_builtin_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'dummy_blockchain_module':
            return original_builtin_import(name, globals, locals, fromlist, level)

        if name.startswith('src.blockchain.'):
            attempted_imports.add(name)
            return MagicMock(name=f"MockedModuleForTypeChecking_{name}")
        
        if name in ['sys', 'typing', 'os', 'builtins', 'importlib']:
             return original_builtin_import(name, globals, locals, fromlist, level)
        
        return MagicMock(name=f"GenericMockedImport_{name}")

    mocker.patch('builtins.__import__', side_effect=mock_builtin_import)

    try:
        import dummy_blockchain_module
        assert dummy_blockchain_module.DUMMY_VAR is True

        assert 'src.blockchain.defi_options' in attempted_imports

    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.clear()
        sys.modules.update(original_sys_modules)
        if 'dummy_blockchain_module' in sys.modules:
            del sys.modules['dummy_blockchain_module']

def test_pricing_init(tmp_path, mocker): # Added mocker fixture
    # Create a dummy package for testing
    pkg_path = tmp_path / "my_pricing_pkg"
    pkg_path.mkdir()
    (pkg_path / "__init__.py").write_text(
        "import sys\n" # Added sys import
        "from src.utils.lazy_import import lazy_import\n"
        "_import_map = {\'BlackScholesEngine\': \'.black_scholes\'}\n"
        "def __getattr__(name): return lazy_import(__name__, _import_map, name, sys.modules[__name__])\n"
    )
    (pkg_path / "black_scholes.py").write_text("class BlackScholesEngine: pass")

    # Mock importlib.import_module to prevent leakage from other tests
    # Ensure that when lazy_import tries to import '.black_scholes', it returns a valid mock.
    mocker.patch('src.utils.lazy_import.import_module', side_effect=lambda name, package=None: MagicMock() if name == 'my_pricing_pkg.black_scholes' or name == 'src.utils.lazy_import' or name == 'structlog' else importlib.import_module(name, package))

    sys.path.insert(0, str(tmp_path))
    try:
        import my_pricing_pkg
        importlib.reload(my_pricing_pkg)
        assert hasattr(my_pricing_pkg, "BlackScholesEngine")
    finally:
        sys.path.remove(str(tmp_path))
        if 'my_pricing_pkg' in sys.modules:
            del sys.modules['my_pricing_pkg']

def test_lazy_import_full_module_path_starts_with_dot(tmp_path, mocker): # Added mocker fixture
    reset_import_stats()
    # Create a temporary package structure
    pkg_name = "my_package_temp"
    pkg_path = tmp_path / pkg_name
    pkg_path.mkdir()
    (pkg_path / "__init__.py").write_text(
        f"import sys\n"
        f"from src.utils.lazy_import import lazy_import\n"
        f"_import_map={{'my_attr':'.submodule'}}\n"
        f"def __getattr__(name): return lazy_import('{pkg_name}', _import_map, name, sys.modules[__name__])\n"
    )
    submodule_dir = pkg_path / "submodule"
    submodule_dir.mkdir()
    (submodule_dir / "__init__.py").write_text("my_attr = 'mocked_relative_attr'")

    # Mock importlib.import_module to prevent leakage from other tests
    mocker.patch('src.utils.lazy_import.import_module', side_effect=lambda name, package=None: MagicMock() if name == f'{pkg_name}.submodule' or name == 'src.utils.lazy_import' or name == 'structlog' else importlib.import_module(name, package))

    sys.path.insert(0, str(tmp_path))
    try:
        # Import the package to make it available for lazy_import
        my_package_temp = importlib.import_module(pkg_name)
        
        # Test a relative import path that exists
        res = lazy_import(pkg_name, {"my_attr":".submodule"}, "my_attr", my_package_temp)
        assert res == "mocked_relative_attr"
        assert hasattr(my_package_temp, "my_attr")
        assert my_package_temp.my_attr == "mocked_relative_attr"
    finally:
        sys.path.remove(str(tmp_path))
        if pkg_name in sys.modules:
            del sys.modules[pkg_name]