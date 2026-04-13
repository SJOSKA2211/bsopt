import random
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class StealthHttpClient:
    """
    HTTP client using TLS fingerprint mimicry.
    OPTIMIZED: Async-native with synchronized headers and TLS fingerprints.
    """

    def __init__(self):
        try:
            from curl_cffi.requests import AsyncSession

            self.session = AsyncSession()
            self._has_cffi = True
        except ImportError:
            import httpx

            self.session = httpx.AsyncClient()
            self._has_cffi = False
            logger.warning("curl_cffi_not_found_using_httpx")

    async def get(self, url: str, **kwargs) -> Any:
        """Perform request with synchronized browser impersonation."""
        # 1. Select impersonation target
        target = random.choice(["chrome110", "chrome120", "safari15_5"])  # nosec B311

        # 2. Map headers to the specific target to avoid mismatches
        # This is a simplified map - in a true Optimized implementation, we'd have full profiles
        ua_map = {
            "chrome110": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
            "chrome120": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "safari15_5": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.5 Safari/605.1.15",
        }

        headers = {
            "User-Agent": ua_map.get(target, ua_map["chrome120"]),
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Upgrade-Insecure-Requests": "1",
        }

        if self._has_cffi:
            response = await self.session.get(url, headers=headers, impersonate=target, **kwargs)
            logger.debug("stealth_request_complete", url=url, target=target)
            return response
        return await self.session.get(url, headers=headers, **kwargs)


# Default client instance
default_stealth_client = StealthHttpClient()