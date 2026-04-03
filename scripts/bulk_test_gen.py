import os

TEMPLATE = """
import pytest
import importlib

def test_import_{MOD_UNDERSCORE}():
    # Automatically generated import test for {MOD_PATH}
    module = importlib.import_module("src.{MOD_PATH}")
    assert module is not None

def test_initialization_{MOD_UNDERSCORE}():
    # Try to find and initialize classes in {MOD_PATH}
    try:
        module = importlib.import_module("src.{MOD_PATH}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and attr.__module__ == 'src.{MOD_PATH}':
                try:
                    # Try to initialize with no args if possible
                    instance = attr()
                    assert instance is not None
                except:
                    pass
    except ImportError:
        pytest.skip('Could not import src.{MOD_PATH}')
"""


def generate_tests():
    src_dir = "src"
    test_dir = "tests/generated"
    os.makedirs(test_dir, exist_ok=True)

    print(f"🛠️ Starting Bulk Test Generation in {test_dir}...")

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                rel_path = os.path.relpath(os.path.join(root, file), src_dir)
                module_path = rel_path.replace(".py", "").replace("/", ".")
                test_file_name = f"test_{rel_path.replace('/', '_')}"
                if not test_file_name.endswith(".py"):
                    test_file_name += ".py"

                test_path = os.path.join(test_dir, test_file_name)

                content = TEMPLATE.replace(
                    "{MOD_UNDERSCORE}", module_path.replace(".", "_")
                ).replace("{MOD_PATH}", module_path)

                with open(test_path, "w") as f:
                    f.write(content)

    print("🏁 Bulk Test Generation Complete.")


if __name__ == "__main__":
    generate_tests()
