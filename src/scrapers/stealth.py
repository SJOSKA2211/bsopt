import random
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

class StealthHttpClient:
    """
    SOTA: Invisible HTTP client using TLS fingerprint mimicry.
    Bypasses modern WAFs by masquerading as a specific browser.
    """
    def __init__(self):
        try:
            from curl_cffi import requests as cffi_requests
            self.session = cffi_requests.Session()
            self._has_cffi = True
        except ImportError:
            import httpx
            self.session = httpx.AsyncClient()
            self._has_cffi = False
            logger.warning("curl_cffi_missing_using_httpx_fallback")

    async def get(self, url: str, **kwargs) -> Any:
        """🚀 SINGULARITY: High-stealth GET request."""
        # SOTA: Mimic Chrome 120 on Windows
        impersonate = random.choice(["chrome110", "chrome120", "safari15_5"])
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1"
        }

        if self._has_cffi:
            # 🚀 SOTA: JA3 Fingerprinting via impersonate
            response = self.session.get(url, headers=headers, impersonate=impersonate, **kwargs)
            logger.info("stealth_request_sent", url=url, impersonate=impersonate)
            return response
        else:
            return await self.session.get(url, headers=headers, **kwargs)

# Singleton ghost client
stealth_client = StealthHttpClient()
