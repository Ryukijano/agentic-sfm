"""Tool client: HTTP client for the MLLM agent to call tools."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ToolClient:
    """Async HTTP client for the tool server."""

    def __init__(self, base_url: str = "http://localhost:8765", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def health(self) -> dict[str, Any]:
        r = self.client.get("/health")
        r.raise_for_status()
        return r.json()

    def register_image(self, image_id: str, path: str) -> dict[str, Any]:
        r = self.client.post(
            "/register_image", json={"image_id": image_id, "path": path}
        )
        r.raise_for_status()
        return r.json()

    def crop(self, image_id: str, bbox: list[float]) -> dict[str, Any]:
        r = self.client.post("/crop", json={"image_id": image_id, "bbox": bbox})
        r.raise_for_status()
        return r.json()

    def match(
        self, image_a: str, image_b: str, matcher: str = "mast3r", max_size: int = 512
    ) -> dict[str, Any]:
        r = self.client.post(
            "/match",
            json={"image_a": image_a, "image_b": image_b, "matcher": matcher, "max_size": max_size},
        )
        r.raise_for_status()
        return r.json()

    def doppelganger_check(self, image_a: str, image_b: str) -> dict[str, Any]:
        r = self.client.post(
            "/doppelganger_check", json={"image_a": image_a, "image_b": image_b}
        )
        r.raise_for_status()
        return r.json()

    def retrieve(self, query_image: str, k: int = 5) -> dict[str, Any]:
        r = self.client.post("/retrieve", json={"query_image": query_image, "k": k})
        r.raise_for_status()
        return r.json()

    def sfm_run(
        self,
        image_dir: str,
        pair_list: list[tuple[str, str]] | None = None,
        output_dir: str = "./outputs/sfm_run",
    ) -> dict[str, Any]:
        r = self.client.post(
            "/sfm_run",
            json={"image_dir": image_dir, "pair_list": pair_list, "output_dir": output_dir},
        )
        r.raise_for_status()
        return r.json()

    def inspect(self, recon_dir: str) -> dict[str, Any]:
        r = self.client.post("/inspect", json={"recon_dir": recon_dir})
        r.raise_for_status()
        return r.json()

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
