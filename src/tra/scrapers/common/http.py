"""One polite HTTP client shared by every scraper and API client.

Features: honest User-Agent, global rate limit, retries with exponential backoff on network errors and
429/5xx responses, timeouts, and an optional on-disk cache so historical data is never downloaded twice.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpError(RuntimeError):
    pass


class HttpClient:
    def __init__(
        self,
        user_agent: str,
        requests_per_second: float = 1.0,
        max_retries: int = 4,
        backoff_seconds: float = 2.0,
        timeout_seconds: float = 30.0,
        cache_dir: Path | None = None,
        session: requests.Session | None = None,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("A non-empty User-Agent is required")
        self._session = session or requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._min_interval = 1.0 / requests_per_second
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._timeout = timeout_seconds
        self._cache_dir = cache_dir
        self._last_request = 0.0

    def get_bytes(self, url: str, params: dict[str, str] | None = None, use_cache: bool = True) -> bytes:
        return self.request("GET", url, params=params, use_cache=use_cache)

    def get_text(self, url: str, params: dict[str, str] | None = None, use_cache: bool = True) -> str:
        return self.get_bytes(url, params, use_cache).decode("utf-8", errors="replace")

    def post_text(
        self,
        url: str,
        data: dict[str, str],
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> str:
        body = self.request("POST", url, data=data, headers=headers, use_cache=use_cache)
        return body.decode("utf-8", errors="replace")

    def request(
        self,
        method: str,
        url: str,
        params: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> bytes:
        cache_path = self._cache_path(method, url, {**(params or {}), **(data or {})}) if use_cache else None
        if cache_path is not None and cache_path.exists():
            return cache_path.read_bytes()
        body = self._fetch(method, url, params, data, headers)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(body)
        return body

    def _fetch(
        self,
        method: str,
        url: str,
        params: dict[str, str] | None,
        data: dict[str, str] | None,
        headers: dict[str, str] | None,
    ) -> bytes:
        send = getattr(self._session, method.lower())
        kwargs: dict[str, object] = {"params": params, "timeout": self._timeout}
        if data is not None:
            kwargs["data"] = data
        if headers:
            kwargs["headers"] = headers
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                resp = send(url, **kwargs)
            except requests.RequestException as exc:
                last_error = exc
            else:
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code not in RETRY_STATUS:
                    raise HttpError(f"{method} {url} returned HTTP {resp.status_code}")
                last_error = HttpError(f"{method} {url} returned HTTP {resp.status_code}")
            if attempt < self._max_retries:
                wait = self._backoff * 2**attempt
                log.warning("%s %s failed (%s); retrying in %.0fs", method, url, last_error, wait)
                time.sleep(wait)
        raise HttpError(f"{method} {url} failed after {self._max_retries + 1} attempts: {last_error}")

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _cache_path(self, method: str, url: str, params: dict[str, str]) -> Path | None:
        if self._cache_dir is None:
            return None
        key = method + " " + url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return self._cache_dir / hashlib.sha256(key.encode()).hexdigest()[:32]
