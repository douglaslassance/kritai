"""fal.ai cloud backend for Kritai.

The local path shells out to the ``mflux`` CLI from ``docker.py``. This module
runs the equivalent FLUX.2 models on fal.ai over HTTP instead, landing the
result as a PNG at ``output_path`` so the rest of the UI stays identical.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

# fal.ai only hosts the distilled klein checkpoints, keyed by size. No guidance, no LoRAs.
FAL_TXT2IMG_ENDPOINTS = {
    "4b": "fal-ai/flux-2/klein/4b",
    "9b": "fal-ai/flux-2/klein/9b",
}
FAL_EDIT_ENDPOINTS = {
    "4b": "fal-ai/flux-2/klein/4b/edit",
    "9b": "fal-ai/flux-2/klein/9b/edit",
}

FAL_QUEUE_HOST = "https://queue.fal.run"
FAL_API_HOST = "https://api.fal.ai"
FAL_BILLING_URL = "https://fal.ai/dashboard/billing"

PCT_UNKNOWN = -1

PROVIDER_LOCAL = "local"
PROVIDER_FAL = "fal"
PROVIDER_LABELS = {
    PROVIDER_LOCAL: "Local (mflux)",
    PROVIDER_FAL: "Cloud (fal.ai)",
}


@dataclass
class GenerationRequest:
    """A provider-neutral description of one image generation."""

    mode: str  # "generate" | "edit" | "angle"
    model_name: str  # internal key, e.g. "flux2-klein-4b"
    prompt: str
    input_image_path: str  # canvas (or cropped selection)
    output_path: str
    width: int
    height: int
    resize_output: bool  # width/height differ from the source
    steps: int
    guidance: float | None = None  # None means no guidance scale
    strength: float | None = None  # 0..1, img2img blend (generate only)
    quantize: int | None = None  # local only; ignored by fal
    seed: int | None = None  # None means random
    reference_image_paths: list[str] = field(default_factory=list)
    loras: list[tuple[str, float]] = field(default_factory=list)


LogFn = Callable[[str], None]
ProgressFn = Callable[[int, str], None]


class ProviderError(RuntimeError):
    """Raised when a backend fails to produce an image."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


ADMIN_KEY_HINT = "Reading credits needs an ADMIN-scoped key from fal.ai/dashboard/keys."


class FalProvider:
    """Runs a GenerationRequest on fal.ai via its HTTP queue API."""

    def __init__(self, api_key: str, poll_interval: float = 1.5, timeout: float = 600.0) -> None:
        self.api_key = (api_key or "").strip()
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def validate(self) -> None:
        """Raise ProviderError unless the key is accepted by fal.ai."""
        if not self.api_key:
            raise ProviderError("No fal.ai API key set.")
        try:
            self._get(f"{FAL_API_HOST}/v1/models?limit=1")
        except ProviderError as e:
            if e.http_status in (401, 403):
                raise ProviderError("fal.ai rejected this API key.", e.http_status)
            raise

    def balance(self) -> tuple[float, str]:
        """Return the remaining credit balance and its currency. Needs an ADMIN key."""
        if not self.api_key:
            raise ProviderError("No fal.ai API key set.")
        try:
            data = self._get(f"{FAL_API_HOST}/v1/account/billing?expand=credits")
        except ProviderError as e:
            if e.http_status in (401, 403):
                raise ProviderError(ADMIN_KEY_HINT, e.http_status)
            raise
        credits = data.get("credits") or {}
        if "current_balance" not in credits:
            raise ProviderError(f"fal.ai returned no balance: {json.dumps(data)[:300]}")
        return float(credits["current_balance"]), str(credits.get("currency") or "USD")

    def run(self, request: GenerationRequest, log: LogFn, progress: ProgressFn) -> None:
        if not self.api_key:
            raise ProviderError("No fal.ai API key set. Add one in the Backend section.")

        endpoint, payload = self._build_request(request, log)
        log(f"fal.ai: submitting to {endpoint}")
        progress(PCT_UNKNOWN, "Submitting to fal.ai")

        submit = self._post(f"{FAL_QUEUE_HOST}/{endpoint}", payload)
        status_url = submit.get("status_url")
        response_url = submit.get("response_url")
        if not status_url or not response_url:
            raise ProviderError(f"Unexpected fal.ai response: {json.dumps(submit)[:300]}")

        self._wait(status_url, log, progress)

        result = self._get(response_url)
        image_url = self._extract_image_url(result)
        if not image_url:
            raise ProviderError(f"fal.ai returned no image: {json.dumps(result)[:300]}")

        log("fal.ai: downloading result")
        progress(PCT_UNKNOWN, "Downloading result")
        self._download(image_url, request.output_path)
        progress(100, "Done")

    def _build_request(self, request: GenerationRequest, log: LogFn):
        size = "9b" if "9b" in request.model_name else "4b"
        if "base" in request.model_name:
            log(f"fal.ai: no base klein endpoint, running the distilled {size} instead")
        elif request.guidance is not None and request.mode != "angle":
            log("fal.ai: guidance isn't supported by the distilled klein endpoints; ignoring it")

        # The Generate tab is img2img, which only the edit endpoints can do.
        use_edit = request.mode in ("edit", "angle") or request.strength is not None
        endpoint = (FAL_EDIT_ENDPOINTS if use_edit else FAL_TXT2IMG_ENDPOINTS)[size]

        payload = {
            "prompt": request.prompt,
            "num_inference_steps": request.steps,
        }
        if use_edit:
            image_urls = [self._data_uri(request.input_image_path)]
            for ref in request.reference_image_paths:
                ref = (ref or "").strip()
                if ref and os.path.exists(ref):
                    image_urls.append(self._data_uri(ref))
            payload["image_urls"] = image_urls
            if request.strength is not None:
                log("fal.ai: image strength isn't supported by the edit endpoint; ignoring it")
        if not use_edit or request.resize_output:
            payload["image_size"] = {"width": request.width, "height": request.height}
        if request.seed is not None:
            payload["seed"] = request.seed
        for path, _scale in request.loras:
            if (path or "").strip():
                log(f"fal.ai: skipping LoRA '{os.path.basename(path)}' (not supported on fal.ai)")

        return endpoint, payload

    def _wait(self, status_url: str, log: LogFn, progress: ProgressFn) -> None:
        deadline = time.monotonic() + self.timeout
        last_status = None
        while True:
            if self._cancelled:
                raise ProviderError("Cancelled")
            status = self._get(status_url)
            state = status.get("status")
            if state != last_status:
                log(f"fal.ai: {state}")
                last_status = state
                label = "Generating on fal.ai" if state == "IN_PROGRESS" else "Queued on fal.ai"
                progress(PCT_UNKNOWN, label)
            if state == "COMPLETED":
                return
            if state in ("FAILED", "ERROR"):
                raise ProviderError(f"fal.ai request failed: {json.dumps(status)[:300]}")
            if time.monotonic() > deadline:
                raise ProviderError("fal.ai request timed out")
            time.sleep(self.poll_interval)

    def _headers(self) -> dict:
        return {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        return self._send(req)

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        return self._send(req)

    @staticmethod
    def _send(req: urllib.request.Request) -> dict:
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            raise ProviderError(f"fal.ai HTTP {e.code}: {detail or e.reason}", e.code)
        except urllib.error.URLError as e:
            raise ProviderError(f"fal.ai network error: {e.reason}")
        return json.loads(body) if body else {}

    @staticmethod
    def _download(url: str, output_path: str) -> None:
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
        except urllib.error.URLError as e:
            raise ProviderError(f"fal.ai download failed: {e.reason}")
        with open(output_path, "wb") as f:
            f.write(data)

    @staticmethod
    def _extract_image_url(result: dict) -> str | None:
        images = result.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, dict):
                return first.get("url")
            if isinstance(first, str):
                return first
        image = result.get("image")
        if isinstance(image, dict):
            return image.get("url")
        return None

    @staticmethod
    def _data_uri(path: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "image/png"
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
