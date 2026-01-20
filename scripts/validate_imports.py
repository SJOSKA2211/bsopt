import os
import sys
import importlib.util
from typing import Dict, Any, List, Optional

def _find_lazy_loaded_modules(base_path: str) -> Dict[str, Dict[str, str]]:
    lazy_modules = {}
    # base_path is the package directory (e.g., .../src)
    # package_name is the name of that directory (e.g., 'src')
    package_name = os.path.basename(os.path.abspath(base_path))

    for root, _, files in os.walk(base_path):
        if "__init__.py" in files:
            relative_path = os.path.relpath(root, base_path)
            if relative_path == ".":
                full_module_name = package_name
            else:
                full_module_name = f"{package_name}.{relative_path.replace(os.sep, '.')}"

            init_file_path = os.path.join(root, "__init__.py")
            try:
                spec = importlib.util.spec_from_file_location(full_module_name, init_file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Use a fresh sys.modules context for inspection if possible, 
                    # but simple assignment is usually enough for init execution
                    original_module = sys.modules.get(full_module_name)
                    sys.modules[full_module_name] = module
                    try:
                        spec.loader.exec_module(module)
                        if hasattr(module, '_import_map') and hasattr(module, '__getattr__'):
                            lazy_modules[full_module_name] = module._import_map
                    except Exception:
                        pass
                    finally:
                        if original_module is not None:
                            sys.modules[full_module_name] = original_module
                        else:
                            del sys.modules[full_module_name]
            except Exception:
                pass
    return lazy_modules

def validate_lazy_imports(base_path: str):
    broken_imports = {}
    original_sys_path = sys.path.copy()
    project_root = os.path.dirname(os.path.abspath(base_path))
    package_name = os.path.basename(os.path.abspath(base_path))
    
    sys.path.insert(0, project_root)
    
    # Try to get exceptions from the target package's utils.lazy_import
    try:
        utils_mod_name = f"{package_name}.utils.lazy_import"
        if utils_mod_name in sys.modules: del sys.modules[utils_mod_name]
        lazy_mod = importlib.import_module(utils_mod_name)
        LazyImportError = getattr(lazy_mod, 'LazyImportError', Exception)
        CircularImportError = getattr(lazy_mod, 'CircularImportError', Exception)
    except ImportError:
        LazyImportError = Exception
        CircularImportError = Exception

    try:
        lazy_modules_info = _find_lazy_loaded_modules(base_path)
        if not lazy_modules_info:
            print("No lazy-loaded modules with _import_map and __getattr__ found under the given path.")
            return

        for module_name, import_map in lazy_modules_info.items():
            print(f"Validating lazy imports for module: {module_name}")
            try:
                if module_name in sys.modules: del sys.modules[module_name]
                parent_module = importlib.import_module(module_name)
            except Exception as e:
                print(f"Error: Could not dynamically import parent module {module_name}: {e}")
                broken_imports[f"{module_name} (parent module)"] = str(e)
                continue

            for attr_name, relative_path in import_map.items():
                try:
                    _ = getattr(parent_module, attr_name)
                except (LazyImportError, CircularImportError, ImportError, AttributeError) as e:
                    full_import_path = f"{module_name}.{attr_name} (maps to {relative_path})"
                    broken_imports[full_import_path] = str(e)
                except Exception as e:
                    full_import_path = f"{module_name}.{attr_name} (maps to {relative_path})"
                    broken_imports[full_import_path] = f"Unexpected error: {str(e)}"

        if broken_imports:
            print("\nValidation failed: Found broken lazy import mappings.")
            for imp, error in broken_imports.items():
                print(f"- {imp}: {error}")
            sys.exit(1)
        else:
            print("\nValidation successful: All lazy imports confirmed valid.")
    finally:
        sys.path = original_sys_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = os.path.abspath(sys.argv[1])
    else:
        # Default to real project 'src'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(os.path.dirname(script_dir), "src")
    
    validate_lazy_imports(base_path=target_path)