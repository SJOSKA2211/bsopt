"""
CLI Configuration Manager

Manages CLI-specific settings and preferences.
"""

from pathlib import Path
from typing import Any

import orjson


class ConfigManager:
    """Manages CLI configuration stored in the user's home directory."""

    def __init__(self) -> None:
        self.config_file = Path.home() / ".bsopt" / "config.json"
        self._ensure_config_dir()
        self.defaults: dict[str, Any] = {
            "api": {"base_url": "http://localhost:8000"},
            "output": {"format": "table", "color": True},
            "pricing": {"default_method": "bs"},
        }
        self.config: dict[str, Any] = self._load()

    def _ensure_config_dir(self) -> None:
        """Ensure the configuration directory exists."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        """Load configuration from file, falling back to defaults."""
        if not self.config_file.exists():
            return self.defaults.copy()

        try:
            with open(self.config_file, "rb") as f:
                user_config = orjson.loads(f.read())
                # Deep merge defaults with user config (simple version)
                config: dict[str, Any] = self.defaults.copy()
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in config:
                        # OPTIMIZED: Ensure we have a dict before updating
                        target = config[key]
                        if isinstance(target, dict):
                            target.update(value)
                    else:
                        config[key] = value
                return config
        except Exception:
            return self.defaults.copy()

    def save(self, scope: str = "user") -> None:
        """Save current configuration to file."""
        if scope == "user":
            with open(self.config_file, "wb") as f:
                f.write(orjson.dumps(self.config, option=orjson.OPT_INDENT_2))

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'api.base_url')."""
        parts = key.split(".")
        val: Any = self.config
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return default
        return val

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        parts = key.split(".")
        val = self.config
        for part in parts[:-1]:
            if part not in val:
                val[part] = {}
            val = val[part]
        val[parts[-1]] = value

    def get_all(self) -> dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self.config

    def reset(self, scope: str = "user") -> None:
        """Reset configuration to defaults."""
        if scope == "user":
            self.config = self.defaults.copy()
            self.save()


def get_config() -> ConfigManager:
    """Singleton-like helper to get ConfigManager instance."""
    return ConfigManager()
