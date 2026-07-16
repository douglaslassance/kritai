"""Generation backends for Kritai.

Kritai can run the same FLUX.2 models two ways:

* ``LocalProvider`` shells out to the local ``mflux`` CLI (Apple-Silicon / MLX),
  the original behaviour. No network, no cost, macOS only.
* ``FalProvider`` runs the equivalent model on fal.ai over HTTP. Works on any
  platform, requires a fal API key, and bills per image.

Both providers consume a provider-neutral :class:`GenerationRequest` and land
their result as a PNG at ``output_path`` so the rest of the UI (preview,
import-to-layer) stays identical regardless of backend.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

# --- mflux model registry (local backend) ----------------------------------

MFLUX_DIR = os.path.expanduser("~/.local/bin")

# Maps model name → (cli_binary, model_flag, supports_strength, supports_guidance, needs_reference_image).
# Distilled models (klein, schnell) don't accept a variable guidance scale.
# Models with needs_reference_image=True use --image-paths [canvas, ref…] instead of --image-path canvas.
MODEL_CLI = {
    # FLUX.2 — distilled variants: no guidance; base variants: guidance ok
    "flux2-klein-4b":      ("mflux-generate-flux2",      "flux2-klein-4b",      True,  False, False),
    "flux2-klein-9b":      ("mflux-generate-flux2",      "flux2-klein-9b",      True,  False, False),
    "flux2-klein-base-4b": ("mflux-generate-flux2",      "flux2-klein-base-4b", True,  True,  False),
    "flux2-klein-base-9b": ("mflux-generate-flux2",      "flux2-klein-base-9b", True,  True,  False),
    # FLUX.2 edit — canvas + optional reference image via --image-paths.
    # Model is chosen at runtime via the Edit tab's model selector.
    "flux2-edit":          ("mflux-generate-flux2-edit", None,                  False, True,  True),
}

# Which models belong to which tab.
GENERATE_MODELS = ["flux2-klein-4b", "flux2-klein-9b", "flux2-klein-base-4b", "flux2-klein-base-9b"]
EDIT_MODELS = ["flux2-edit"]
# Models available in the Angle tab (must be compatible with mflux-generate-flux2-edit).
ANGLE_MODELS = ["flux2-klein-4b", "flux2-klein-9b", "flux2-klein-base-4b", "flux2-klein-base-9b"]

# --- fal.ai endpoint mapping (cloud backend) --------------------------------

# Text-to-image endpoints, one per klein variant. The "base" models accept a
# guidance scale; the plain klein models are distilled (4-step, no guidance).
# NOTE: the 9b endpoint ids are inferred from fal's 4b naming and may need to
# be adjusted once fal publishes them — they're isolated here on purpose.
FAL_TXT2IMG_ENDPOINTS = {
    "flux2-klein-4b":      "fal-ai/flux-2/klein/4b/distilled",
    "flux2-klein-9b":      "fal-ai/flux-2/klein/9b/distilled",
    "flux2-klein-base-4b": "fal-ai/flux-2/klein/4b",
    "flux2-klein-base-9b": "fal-ai/flux-2/klein/9b",
}

# Image editing / reference endpoint (takes image_urls[]). Used for the Edit and
# Frame tabs, and for the Generate tab when it's acting as img2img.
FAL_EDIT_ENDPOINT = "fal-ai/flux-2/klein/4b/edit"

FAL_QUEUE_HOST = "https://queue.fal.run"

# Providers exposed in the backend dropdown.
PROVIDER_LOCAL = "local"
PROVIDER_FAL = "fal"
PROVIDER_LABELS = {
    PROVIDER_LOCAL: "Local (mflux)",
    PROVIDER_FAL: "Cloud (fal.ai)",
}


@dataclass
class GenerationRequest:
    """A provider-neutral description of one image generation."""

    mode: str                                   # "generate" | "edit" | "angle"
    model_name: str                             # internal key, e.g. "flux2-klein-4b"
    prompt: str
    input_image_path: str                       # canvas (or cropped selection)
    output_path: str
    width: int
    height: int
    resize_output: bool                         # width/height differ from the source
    steps: int
    guidance: Optional[float] = None            # None → don't send a guidance scale
    strength: Optional[float] = None            # 0..1, img2img blend (generate only)
    quantize: Optional[int] = None              # local only; ignored by fal
    seed: Optional[int] = None                  # None → random
    reference_image_paths: List[str] = field(default_factory=list)
    loras: List[Tuple[str, float]] = field(default_factory=list)


# Callbacks a provider uses to stream progress back to the UI thread.
LogFn = Callable[[str], None]
ProgressFn = Callable[[int], None]


class ProviderError(RuntimeError):
    """Raised when a backend fails to produce an image."""


class Provider:
    """Runs a :class:`GenerationRequest`, landing a PNG at ``request.output_path``."""

    def run(self, request: GenerationRequest, log: LogFn, progress: ProgressFn) -> None:
        raise NotImplementedError


# --- Local (mflux CLI) ------------------------------------------------------


class LocalProvider(Provider):
    """Runs the request through the local ``mflux`` command-line tools."""

    def run(self, request: GenerationRequest, log: LogFn, progress: ProgressFn) -> None:
        cmd = self._build_command(request)
        log("Running: " + " ".join(f'"{t}"' if " " in t else t for t in cmd))
        self._run_subprocess(cmd, log, progress)

    def _build_command(self, request: GenerationRequest) -> List[str]:
        if request.mode == "generate":
            return self._build_generate_command(request)
        # edit and angle both target mflux-generate-flux2-edit
        return self._build_edit_command(request)

    def _build_generate_command(self, request: GenerationRequest) -> List[str]:
        cli_name, model_flag, *_ = MODEL_CLI.get(
            request.model_name, ("mflux-generate-flux2", request.model_name, True, True, False)
        )
        cli_path = os.path.join(MFLUX_DIR, cli_name)

        cmd = [cli_path, "--prompt", request.prompt]
        if model_flag:
            cmd += ["--model", model_flag]
        cmd += ["--image-path", request.input_image_path]
        cmd += ["--steps", str(request.steps), "--output", request.output_path]
        if request.resize_output:
            cmd += ["--width", str(request.width), "--height", str(request.height)]
        if request.guidance is not None:
            cmd += ["--guidance", str(request.guidance)]
        if request.strength is not None:
            cmd += ["--image-strength", str(request.strength)]
        if request.quantize is not None:
            cmd += ["--quantize", str(request.quantize)]
        if request.seed is not None:
            cmd += ["--seed", str(request.seed)]
        cmd += self._lora_args(request.loras)
        return cmd

    def _build_edit_command(self, request: GenerationRequest) -> List[str]:
        # mflux-generate-flux2-edit derives its output size from the first image
        # and has no --width/--height, so pre-scale the canvas in place.
        if request.resize_output:
            self._prescale_input(request)

        # The Edit tab passes "flux2-edit" as the model; the Frame tab passes a
        # klein model name. Either way the binary is the edit tool.
        model_name = request.model_name
        model_flag = "flux2-edit" if request.mode == "edit" else model_name

        cli_path = os.path.join(MFLUX_DIR, "mflux-generate-flux2-edit")
        cmd = [cli_path, "--prompt", request.prompt, "--model", model_flag]
        cmd += ["--image-paths", request.input_image_path]
        for ref in request.reference_image_paths:
            ref = (ref or "").strip()
            if ref:
                cmd.append(ref)
        cmd += ["--steps", str(request.steps), "--output", request.output_path]
        if request.guidance is not None:
            cmd += ["--guidance", str(request.guidance)]
        if request.quantize is not None:
            cmd += ["--quantize", str(request.quantize)]
        if request.seed is not None:
            cmd += ["--seed", str(request.seed)]
        cmd += self._lora_args(request.loras)
        return cmd

    @staticmethod
    def _prescale_input(request: GenerationRequest) -> None:
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QImage

        img = QImage(request.input_image_path)
        if not img.isNull():
            img.scaled(
                request.width, request.height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            ).save(request.input_image_path, "PNG")

    @staticmethod
    def _lora_args(loras: List[Tuple[str, float]]) -> List[str]:
        paths, scales = [], []
        for path, scale in loras:
            path = (path or "").strip()
            if path:
                paths.append(path)
                scales.append(str(scale))
        if not paths:
            return []
        return ["--lora-paths"] + paths + ["--lora-scales"] + scales

    @staticmethod
    def _run_subprocess(cmd: List[str], log: LogFn, progress: ProgressFn) -> None:
        # Strip Krita's Python environment variables so they don't bleed into the
        # mflux subprocess (causes an SRE module mismatch otherwise).
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE")
        }
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=clean_env
        )

        stderr_lines: List[str] = []

        def drain_stderr() -> None:
            for line in proc.stderr:
                stderr_lines.append(line)
                log(line.rstrip())
                m = re.search(r"(\d+)%\|", line)
                if m:
                    progress(int(m.group(1)))

        t = threading.Thread(target=drain_stderr, daemon=True)
        t.start()
        stdout, _ = proc.communicate()
        t.join()

        if stdout.strip():
            log(stdout.strip())
        if proc.returncode != 0:
            raise ProviderError("".join(stderr_lines).strip() or "mflux-generate failed")
        progress(100)


# --- Cloud (fal.ai) ---------------------------------------------------------


class FalProvider(Provider):
    """Runs the request on fal.ai via its HTTP queue API."""

    def __init__(self, api_key: str, poll_interval: float = 1.5, timeout: float = 600.0) -> None:
        self.api_key = (api_key or "").strip()
        self.poll_interval = poll_interval
        self.timeout = timeout

    def run(self, request: GenerationRequest, log: LogFn, progress: ProgressFn) -> None:
        if not self.api_key:
            raise ProviderError("No fal.ai API key set. Add one in the Backend section.")

        endpoint, payload = self._build_request(request, log)
        log(f"fal.ai: submitting to {endpoint}")
        progress(5)

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
        self._download(image_url, request.output_path)
        progress(100)

    # -- request shaping --

    def _build_request(self, request: GenerationRequest, log: LogFn):
        use_edit = request.mode in ("edit", "angle") or (
            request.mode == "generate"
            and request.strength is not None
            and request.strength < 0.99
        )

        if use_edit:
            endpoint = FAL_EDIT_ENDPOINT
            image_urls = [self._data_uri(request.input_image_path)]
            for ref in request.reference_image_paths:
                ref = (ref or "").strip()
                if ref and os.path.exists(ref):
                    image_urls.append(self._data_uri(ref))
            payload = {
                "prompt": request.prompt,
                "image_urls": image_urls,
                "num_inference_steps": request.steps,
            }
            # The edit endpoint sizes the output from the input image, so we don't
            # send image_size. mflux's --image-strength has no fal equivalent here.
            if request.mode == "generate" and request.strength is not None:
                log("fal.ai: image strength isn't supported by the edit endpoint; ignoring it")
        else:
            endpoint = FAL_TXT2IMG_ENDPOINTS.get(request.model_name)
            if not endpoint:
                raise ProviderError(f"No fal.ai endpoint mapped for model '{request.model_name}'")
            payload = {
                "prompt": request.prompt,
                "image_size": {"width": request.width, "height": request.height},
                "num_inference_steps": request.steps,
            }

        if request.guidance is not None:
            payload["guidance_scale"] = request.guidance
        if request.seed is not None:
            payload["seed"] = request.seed

        # LoRAs on fal must be hosted (URL); local file paths can't be uploaded here.
        lora_specs = []
        for path, scale in request.loras:
            path = (path or "").strip()
            if not path:
                continue
            if re.match(r"^https?://", path):
                lora_specs.append({"path": path, "scale": scale})
            else:
                log(f"fal.ai: skipping local LoRA '{os.path.basename(path)}' (cloud needs a URL)")
        if lora_specs:
            payload["loras"] = lora_specs

        if request.quantize is not None:
            log("fal.ai: quantize is a local-only setting; ignoring it")

        return endpoint, payload

    def _wait(self, status_url: str, log: LogFn, progress: ProgressFn) -> None:
        deadline = time.monotonic() + self.timeout
        pct = 5
        last_status = None
        while True:
            status = self._get(status_url)
            state = status.get("status")
            if state != last_status:
                log(f"fal.ai: {state}")
                last_status = state
            if state == "COMPLETED":
                return
            if state in ("FAILED", "ERROR"):
                raise ProviderError(f"fal.ai request failed: {json.dumps(status)[:300]}")
            if time.monotonic() > deadline:
                raise ProviderError("fal.ai request timed out")
            # No true percentage from the queue; creep toward 90 while we wait.
            pct = min(90, pct + 3)
            progress(pct)
            time.sleep(self.poll_interval)

    # -- HTTP helpers --

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
            raise ProviderError(f"fal.ai HTTP {e.code}: {detail or e.reason}")
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
    def _extract_image_url(result: dict) -> Optional[str]:
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


def make_provider(provider_id: str, fal_api_key: str = "") -> Provider:
    """Factory: return the provider for ``provider_id`` (``local`` or ``fal``)."""
    if provider_id == PROVIDER_FAL:
        return FalProvider(fal_api_key)
    return LocalProvider()
