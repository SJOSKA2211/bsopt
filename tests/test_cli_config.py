from unittest.mock import MagicMock

from src.cli.config import ConfigManager, get_config
from tests.test_utils import assert_equal


def test_cli_config_get():
    config = get_config()
    assert config is not None


def test_cli_config_set():
    config = ConfigManager()
    # Mock file operations to avoid writing to disk
    config._load = MagicMock(return_value={})
    config.save = MagicMock()

    config.set("test_key", "test_value")
    assert_equal(config.get("test_key"), "test_value")
