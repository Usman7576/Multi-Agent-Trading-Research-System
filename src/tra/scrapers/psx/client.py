"""PSX Data Portal client.

The portal's own JavaScript loads tables with AJAX calls that carry an ``X-Req-Id`` token embedded in every
page (``window.__ps._k``). This client reads the token from a page once, sends it with each data request
like the browser does, and refreshes it if a request is rejected. All traffic goes through the shared
rate-limited ``HttpClient``.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from tra.scrapers.common.http import HttpClient, HttpError

log = logging.getLogger(__name__)

BASE = "https://dps.psx.com.pk"
USER_AGENT = (
    "Usman7576-trading-research/0.1 (+https://github.com/Usman7576/Multi-Agent-Trading-Research-System)"
)
_TOKEN = re.compile(r'"_k"\s*:\s*"([^"]+)"')


class PSXClient:
    def __init__(self, http: HttpClient, backoff_seconds: float = 5.0) -> None:
        self._http = http
        self._token: str | None = None
        self._backoff = backoff_seconds

    def _headers(self) -> dict[str, str]:
        if self._token is None:
            page = self._http.get_text(f"{BASE}/historical", use_cache=False)
            match = _TOKEN.search(page)
            if match is None:
                raise HttpError("PSX page token (window.__ps._k) not found; the portal may have changed")
            self._token = match.group(1)
        return {"X-Req-Id": self._token, "X-Requested-With": "XMLHttpRequest"}

    def post(self, path: str, data: dict[str, str], use_cache: bool = False, attempts: int = 4) -> str:
        """POST a portal form. The server sometimes answers a plain ``error`` body under load; that is
        retried with backoff and a fresh page token, like a rejected request."""
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                text = self._http.post_text(BASE + path, data, self._headers(), use_cache=use_cache)
                if text.strip().lower() != "error":
                    return text
                last = HttpError(f"POST {path} {data} returned an 'error' body")
            except HttpError as exc:
                last = exc
            self._token = None
            wait = self._backoff * 2**attempt
            log.warning("%s; retrying in %.0fs with a fresh token", last, wait)
            time.sleep(wait)
        raise HttpError(f"POST {path} failed after {attempts} attempts: {last}")

    def get_json(self, path: str) -> Any:
        text = self._http.request("GET", BASE + path, headers=self._headers(), use_cache=False)
        return json.loads(text)

    def get_page(self, path: str) -> str:
        return self._http.get_text(BASE + path, use_cache=False)

    def download(self, path: str) -> bytes:
        return self._http.request("GET", BASE + path, use_cache=False)
