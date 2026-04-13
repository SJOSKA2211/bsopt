"""
Configuration Proxy Module
Redirects to the high-performance shared configuration.
"""

from src.shared.config import Settings, get_settings, settings

__all__ = ["settings", "get_settings", "Settings"]