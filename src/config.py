"""
Configuration Proxy Module
Redirects to the high-performance shared configuration.
"""

from src.shared.config import settings, get_settings, Settings

__all__ = ["settings", "get_settings", "Settings"]
