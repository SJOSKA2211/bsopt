from unittest.mock import patch

import pytest

from src.cli.config import ConfigManager


@pytest.fixture
def config_manager(tmp_path):
    with patch("src.cli.config.Path.home", return_value=tmp_path):
        yield ConfigManager()

def test_config_defaults(config_manager):
    assert config_manager.get("api.base_url") == "http://localhost:8000"
    assert config_manager.get("output.format") == "table"

def test_config_set_get(config_manager):
    config_manager.set("api.base_url", "http://new-api")
    assert config_manager.get("api.base_url") == "http://new-api"
    
    config_manager.set("custom.option", True)
    assert config_manager.get("custom.option") is True

def test_config_save_load(config_manager):
    config_manager.set("api.base_url", "http://saved-api")
    config_manager.save()
    
    # New manager instance should load saved config
    with patch("src.cli.config.Path.home", return_value=config_manager.config_file.parent.parent):
        new_manager = ConfigManager()
        assert new_manager.get("api.base_url") == "http://saved-api"
