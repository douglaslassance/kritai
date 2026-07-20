from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Optional

from krita import DockWidget, InfoObject
from PyQt5.QtCore import (
    QByteArray,
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtGui import QBrush, QColor, QFontDatabase, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QScrollArea,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

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

# --- External tool discovery ------------------------------------------------

# `uv tool install` and Homebrew drop binaries in a handful of well-known dirs
# that Krita's minimal (Finder-launched) PATH often doesn't include, so we
# search them explicitly rather than relying on PATH alone.
_TOOL_SEARCH_DIRS = [
    MFLUX_DIR,                              # uv tool install / pipx default
    "/opt/homebrew/bin",                   # Homebrew on Apple Silicon
    "/usr/local/bin",                      # Homebrew on Intel
    os.path.expanduser("~/.cargo/bin"),
]


def find_executable(name: str) -> Optional[str]:
    """Full path to *name* if it's installed, else ``None``.

    Checks PATH first, then the common install dirs above, since Krita launched
    from Finder inherits a stripped-down PATH.
    """
    found = shutil.which(name)
    if found:
        return found
    for directory in _TOOL_SEARCH_DIRS:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


# --- rembg (background removal, Mask tab) -----------------------------------

# Console script installed by ``uv tool install "rembg[cli]"``.
REMBG_CLI = "rembg"

# Curated subset of rembg's models, best-first: quality / balanced / fast.
# Names must match rembg's ``-m`` values exactly. Weights download on first
# use into ~/.u2net/. (rembg ships many more niche models — anime, portrait,
# human-seg, etc. — deliberately not exposed here to keep the choice simple.)
MASK_MODELS = [
    "birefnet-general",
    "isnet-general-use",
    "u2net",
]

AZIMUTH_MAP = [
    (0,    "front view"),
    (45,   "front-left quarter view"),
    (90,   "left side view"),
    (135,  "back-left quarter view"),
    (180,  "back view"),
    (-135, "back-right quarter view"),
    (-90,  "right side view"),
    (-45,  "front-right quarter view"),
]
ELEVATION_MAP = [
    (-90, "extreme low angle, looking up"),
    (-30, "low angle, slightly looking up"),
    (0,   "eye level"),
    (30,  "slightly elevated, looking down"),
    (60,  "high angle, looking down"),
    (90,  "top-down, bird's eye view"),
]
DISTANCE_MAP = [
    (60,  "close-up"),
    (100, "medium shot"),
    (180, "wide shot"),
]
ANGLE_PRESETS = {
    "Front":  (0,    0,   100),
    "Right":  (90,   0,   100),
    "Back":   (180,  0,   100),
    "Left":   (-90,  0,   100),
    "Top":    (0,    90,  100),
    "Bottom": (0,    -90, 100),
}

# How often (ms) to poll canvas for changes when auto-mode is on.
POLL_INTERVAL_MS = 1500
# How long (ms) to wait after the last detected change before generating.
DEBOUNCE_MS = 2000


class PreviewLabel(QLabel):
    """QLabel that paints its pixmap centered with correct aspect ratio."""

    def __init__(self) -> None:
        super().__init__()
        self._ratio: float = 1.0
        self._source: Optional[QPixmap] = None
        self._height_update_pending: bool = False

    def clearPixmap(self) -> None:
        self._source = None
        self.setMinimumHeight(0)
        self.setMaximumHeight(240)
        self.setFixedHeight(0)
        self.setVisible(False)
        self.update()

    def setRatio(self, ratio: float) -> None:
        if ratio > 0 and ratio != self._ratio:
            self._ratio = ratio
            self._schedule_height_update()

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        self._schedule_height_update()

    def _schedule_height_update(self) -> None:
        """Defer height update to the next event-loop tick to avoid recursive
        layout invalidation (PreviewLabel.setFixedHeight -> parent layout ->
        QPlainTextEdit.resizeEvent -> layout -> PreviewLabel.resizeEvent …)."""
        if self._height_update_pending:
            return
        self._height_update_pending = True
        QTimer.singleShot(0, self._apply_fixed_height)

    def _apply_fixed_height(self) -> None:
        self._height_update_pending = False
        target = max(1, self.heightForWidth(self.width()))
        if target != self.height():
            self.setFixedHeight(target)

    def setPixmap(self, pixmap: QPixmap) -> None:
        self._source = pixmap
        self.update()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return int(width * self._ratio)

    def sizeHint(self) -> QSize:
        return QSize(1, self.heightForWidth(1))

    def minimumSizeHint(self) -> QSize:
        return QSize(1, 1)

    def paintEvent(self, event: QEvent) -> None:
        from PyQt5.QtGui import QPainter
        painter = QPainter(self)
        if self._source:
            scaled = self._source.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)


class _AdaptiveTabWidget(QTabWidget):
    """QTabWidget whose size hint tracks the active tab only.

    Both width and height are derived from the active tab's content so the
    docker's natural minimum width equals the tab widget's minimum width —
    they always agree.  Call updateGeometry() after switching tabs.
    """

    def sizeHint(self) -> QSize:
        return self._hint_for(preferred=True)

    def minimumSizeHint(self) -> QSize:
        return self._hint_for(preferred=False)

    def _hint_for(self, preferred: bool) -> QSize:
        w = self.currentWidget()
        if w is None:
            return super().sizeHint() if preferred else super().minimumSizeHint()
        content = w.sizeHint() if preferred else w.minimumSizeHint()
        tab_bar  = self.tabBar()
        margins  = self.contentsMargins()
        w_extra  = margins.left() + margins.right()
        h_extra  = tab_bar.sizeHint().height() + margins.top() + margins.bottom() + 4
        min_tab_w = tab_bar.minimumSizeHint().width()
        return QSize(
            max(content.width() + w_extra, min_tab_w),
            max(content.height() + h_extra, tab_bar.sizeHint().height() + 20),
        )


class CameraOrbitWidget(QWidget):
    """Interactive 3D orbit widget: drag the teal dot to rotate azimuth,
    the magenta dot (camera) to change elevation, the amber dot to zoom."""

    azimuth_changed   = pyqtSignal(int)   # –180 – 180
    elevation_changed = pyqtSignal(int)   # –90 – 90
    distance_changed  = pyqtSignal(int)   # 60 – 180

    # Fixed isometric view direction (not user-controllable)
    _VY = math.radians(30)   # scene yaw  (rotate world around Y before projecting)
    _VP = math.radians(25)   # view pitch (tilt camera down)

    _R_GRID    = 1.4   # fixed grid radius — independent of camera distance

    _R_HANDLE  = 5    # handle draw radius px
    _R_HIT     = 14   # click hit radius px

    def __init__(self, parent=None):
        super().__init__(parent)
        self._az    = 0
        self._el    = 0
        self._dist  = 100        # range 60–180
        self._drag           = False
        self._drag_origin    = None
        self._drag_start_az  = 0
        self._drag_start_el  = 0
        self._hovered        = False
        self.setFixedWidth(160)
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ api

    def setAzimuth(self, v: int) -> None:
        v = max(-180, min(180, int(v)))
        if self._az != v:
            self._az = v
            self.update()

    def setElevation(self, v: int) -> None:
        v = max(-90, min(90, int(v)))
        if self._el != v:
            self._el = v
            self.update()

    def setDistance(self, v: int) -> None:
        v = max(60, min(180, int(v)))
        if self._dist != v:
            self._dist = v
            self.update()

    def azimuth(self)  -> int: return self._az
    def elevation(self) -> int: return self._el
    def distance(self)  -> int: return self._dist

    # --------------------------------------------------------- 3-D math

    def _scale(self) -> float:
        return min(self.width(), self.height()) * 0.35

    def _origin(self) -> QPointF:
        return QPointF(self.width() * 0.5, self.height() * 0.5)

    def _project(self, x: float, y: float, z: float) -> QPointF:
        vy, vp = self._VY, self._VP
        rx =  x * math.cos(vy) + z * math.sin(vy)
        rz = -x * math.sin(vy) + z * math.cos(vy)
        sx = rx
        sy = -y * math.cos(vp) + rz * math.sin(vp)
        o = self._origin()
        s = self._scale()
        return QPointF(o.x() + sx * s, o.y() + sy * s)

    def _cam_xyz(self, az=None, el=None, dist=None):
        if az   is None: az   = self._az
        if el   is None: el   = self._el
        if dist is None: dist = self._dist
        a = math.radians(az)
        e = math.radians(el)
        r = dist / 100.0
        return (r * math.cos(e) * math.sin(a),
                r * math.sin(e),
                r * math.cos(e) * math.cos(a))

    def _screen_depth(self, x, y, z) -> float:
        """Negative = in front of viewer (use to decide front vs back)."""
        vy = self._VY
        return -x * math.sin(vy) + z * math.cos(vy)

    # ------------------------------------------------ handle screen positions

    def _pos_handle(self) -> QPointF:
        """Camera position projected onto screen — sits on the elevation arc."""
        return self._project(*self._cam_xyz())

    # --------------------------------------------------------------- painting

    def _palette_colors(self):
        """Return (bg, grid, ring, arc) derived from the live Qt palette."""
        from PyQt5.QtWidgets import QApplication
        pal     = QApplication.palette()
        bg      = pal.color(pal.Window).darker(150)
        accent  = pal.color(pal.Highlight)
        grid_c  = QColor(pal.color(pal.Midlight))
        grid_c.setAlpha(160)
        return bg, grid_c, accent

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        bg, grid_c, accent = self._palette_colors()
        p.fillRect(self.rect(), bg)
        self._draw_grid(p, grid_c)
        self._draw_orbit_ring(p, accent)
        self._draw_elevation_arc(p, accent)
        self._draw_handles(p, accent, grid_c)

    def _draw_grid(self, p: QPainter, grid_c: QColor) -> None:
        r = self._R_GRID
        y = -0.35 * r
        p.setPen(QPen(grid_c, 1, Qt.DotLine))
        n = 5
        for i in range(-n, n + 1):
            t = i * r / n
            p.drawLine(self._project(-r, y, t), self._project(r,  y, t))
            p.drawLine(self._project(t,  y, -r), self._project(t, y, r))

    def _fog_alpha(self, depth: float, r: float) -> int:
        """Map screen depth → alpha: front (+r) = 255, back (–r) = 35."""
        t = (depth + r) / (2.0 * r) if r > 0 else 0.5
        return int(35 + max(0.0, min(1.0, t)) * 220)

    def _draw_orbit_ring(self, p: QPainter, base: QColor) -> None:
        r = self._dist / 100.0
        steps = 80
        for i in range(steps):
            a1 = 2 * math.pi * i       / steps
            a2 = 2 * math.pi * (i + 1) / steps
            x1, z1 = r * math.sin(a1), r * math.cos(a1)
            x2, z2 = r * math.sin(a2), r * math.cos(a2)
            depth  = (self._screen_depth(x1, 0, z1) + self._screen_depth(x2, 0, z2)) * 0.5
            col    = QColor(base)
            col.setAlpha(self._fog_alpha(depth, r))
            p.setPen(QPen(col, 1.5))
            p.drawLine(self._project(x1, 0, z1), self._project(x2, 0, z2))

    def _draw_elevation_arc(self, p: QPainter, base: QColor) -> None:
        r  = self._dist / 100.0
        a  = math.radians(self._az)
        steps = 40
        for i in range(steps):
            e1 = math.pi * i       / steps - math.pi / 2
            e2 = math.pi * (i + 1) / steps - math.pi / 2
            def pt(e):
                return (r * math.cos(e) * math.sin(a),
                        r * math.sin(e),
                        r * math.cos(e) * math.cos(a))
            x1, y1, z1 = pt(e1)
            x2, y2, z2 = pt(e2)
            depth  = (self._screen_depth(x1, y1, z1) + self._screen_depth(x2, y2, z2)) * 0.5
            col    = QColor(base)
            col.setAlpha(self._fog_alpha(depth, r))
            p.setPen(QPen(col, 1.5))
            p.drawLine(self._project(x1, y1, z1), self._project(x2, y2, z2))

    def _draw_handles(self, p: QPainter, accent: QColor, grid_c: QColor) -> None:
        r = float(self._R_HANDLE)
        if self._hovered or self._drag:
            from PyQt5.QtWidgets import QApplication
            text_c = QApplication.palette().color(QApplication.palette().WindowText)
            p.setPen(QPen(text_c, 1.5))
        else:
            p.setPen(Qt.NoPen)
        p.setBrush(QBrush(accent))
        p.drawEllipse(self._pos_handle(), r, r)

    # -------------------------------------------------------- mouse events

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        self._drag           = 'free'
        self._drag_origin    = QPointF(event.pos())
        self._drag_start_az  = self._az
        self._drag_start_el  = self._el
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if not self._drag:
            return
        pos = QPointF(event.pos())
        dx  = pos.x() - self._drag_origin.x()
        dy  = pos.y() - self._drag_origin.y()

        new_az = max(-180, min(180, int(round(self._drag_start_az + dx))))
        new_el = max(-90,  min(90,  int(round(self._drag_start_el - dy * 0.8))))
        changed = False
        if self._az != new_az:
            self._az = new_az
            self.azimuth_changed.emit(new_az)
            changed = True
        if self._el != new_el:
            self._el = new_el
            self.elevation_changed.emit(new_el)
            changed = True
        if changed:
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = None
            self.setCursor(Qt.ArrowCursor)


class DropThumbnail(QLabel):
    """64x64 label that accepts image drops and clicks to browse."""

    pathChanged = pyqtSignal(str)

    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tiff")

    def __init__(self) -> None:
        super().__init__()
        self._path: str = ""
        self.setFixedSize(64, 64)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self._apply_placeholder()

    def _apply_placeholder(self) -> None:
        self.setText("+")
        self.setStyleSheet(
            "border: 2px dashed #888; background: #333; color: #aaa; font-size: 24px;"
        )

    def imagePath(self) -> str:
        return self._path

    def setImagePath(self, path: str) -> None:
        self._path = path
        if path and os.path.isfile(path):
            pix = QPixmap(path)
            if not pix.isNull():
                scaled = pix.scaled(
                    self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
                )
                x = (scaled.width() - self.width()) // 2
                y = (scaled.height() - self.height()) // 2
                self.setPixmap(scaled.copy(x, y, self.width(), self.height()))
                self.setStyleSheet("border: 2px solid #666;")
                self.pathChanged.emit(path)
                return
        self._apply_placeholder()
        self.pathChanged.emit(path)

    def mousePressEvent(self, event: QEvent) -> None:
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select reference image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.tiff)"
        )
        if path:
            self.setImagePath(path)

    def dragEnterEvent(self, event: QEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if any(url.toLocalFile().lower().endswith(e) for e in self._IMAGE_EXTS):
                    event.acceptProposedAction()
                    self.setStyleSheet(
                        "border: 2px solid #4a9; background: #333; color: #aaa; font-size: 24px;"
                    )
                    return

    def dragLeaveEvent(self, event: QEvent) -> None:
        if not self._path:
            self._apply_placeholder()
        else:
            self.setStyleSheet("border: 2px solid #666;")

    def dropEvent(self, event: QEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if any(path.lower().endswith(e) for e in self._IMAGE_EXTS):
                self.setImagePath(path)
                return


class CollapsibleSection(QWidget):
    """A full-width accordion-style collapsible section."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._title = title

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header row: toggle + optional extra widgets
        self._header_row = QHBoxLayout()
        self._header_row.setContentsMargins(0, 0, 0, 0)
        self._header_row.setSpacing(2)

        self._toggle = QPushButton(f"▶  {title}")
        self._toggle.setCheckable(True)
        self._toggle.setFlat(True)
        self._toggle.setStyleSheet("text-align: left; padding: 2px 4px;")
        self._toggle.toggled.connect(self._on_toggled)
        self._header_row.addWidget(self._toggle)

        outer.addLayout(self._header_row)

        # Content area
        self._content = QWidget()
        self._content.setVisible(False)
        outer.addWidget(self._content)

    def addHeaderWidget(self, widget: QWidget) -> None:
        """Add a widget to the right side of the header (e.g. a Clear button)."""
        widget.setVisible(False)
        self._toggle.toggled.connect(widget.setVisible)
        self._header_row.addWidget(widget)

    def setContentLayout(self, layout: QVBoxLayout) -> None:
        self._content.setLayout(layout)

    def setExpanded(self, expanded: bool) -> None:
        self._toggle.setChecked(expanded)

    def _on_toggled(self, checked: bool) -> None:
        self._content.setVisible(checked)
        self._toggle.setText(("▼" if checked else "▶") + f"  {self._title}")


class GenerateThread(QThread):
    finished = pyqtSignal(str)        # output path
    errored = pyqtSignal(str)         # error message
    logged = pyqtSignal(str)          # line of stdout/stderr for the log panel
    progress = pyqtSignal(int)        # 0–100

    def __init__(self, cmd: list[str], output_path: str) -> None:
        super().__init__()
        self.cmd = cmd
        self.output_path = output_path

    def run(self) -> None:
        try:
            # Strip Krita's Python environment variables so they don't bleed
            # into the mflux subprocess (causes SRE module mismatch otherwise).
            clean_env = {
                k: v for k, v in os.environ.items()
                if k not in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE")
            }
            proc = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=clean_env,
            )

            stderr_lines = []

            def drain_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line)
                    self.logged.emit(line.rstrip())
                    m = re.search(r"(\d+)%\|", line)
                    if m:
                        self.progress.emit(int(m.group(1)))

            t = threading.Thread(target=drain_stderr, daemon=True)
            t.start()

            stdout, _ = proc.communicate()
            t.join()

            if stdout.strip():
                self.logged.emit(stdout.strip())

            if proc.returncode != 0:
                self.errored.emit("".join(stderr_lines).strip() or "mflux-generate failed")
            else:
                self.progress.emit(100)
                self.finished.emit(self.output_path)
        except Exception as e:
            self.logged.emit(str(e))
            self.errored.emit(str(e))


# ======================================================================
# Dependency detection + one-click install
# ======================================================================


class _Dependency:
    """A CLI tool Kritai shells out to, plus how to install it."""

    def __init__(self, label: str, package: str, executables: list[str],
                 docs_url: str, reason: str, install_args: list[str] = None) -> None:
        self.label = label              # human name, e.g. "mflux"
        self.package = package          # uv package spec, e.g. "rembg[cli]"
        self.executables = executables  # CLI names that must exist once installed
        self.docs_url = docs_url
        self.reason = reason            # one-line "why it's needed"
        self.install_args = install_args or []  # extra flags for `uv tool install`


DEP_MFLUX = _Dependency(
    "mflux",
    "mflux",
    ["mflux-generate-flux2"],
    "https://github.com/filipstrand/mflux",
    "Local generation, editing, framing and upscaling run through the mflux CLI.",
)
DEP_REMBG = _Dependency(
    "rembg",
    # [cli] gives the command; [cpu] pulls the onnxruntime inference backend
    # (rembg 2.x split it into cpu/gpu extras) — without it `rembg i` errors
    # with "No onnxruntime backend found".
    "rembg[cli,cpu]",
    [REMBG_CLI],
    "https://github.com/danielgatis/rembg",
    "Background removal in the Mask tab runs through the rembg CLI.",
    # rembg -> pymatting -> numba: without this floor uv backtracks to an
    # ancient numba whose llvmlite has no Python 3.12 wheel and fails to build.
    install_args=["--with", "numba>=0.60"],
)


class InstallThread(QThread):
    """Runs a sequence of shell commands, streaming output; stops on first failure."""

    logged = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(bool)           # True if every command succeeded

    def __init__(self, commands: list[list[str]]) -> None:
        super().__init__()
        self.commands = commands

    def run(self) -> None:
        clean_env = {
            k: v for k, v in os.environ.items()
            if k not in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE")
        }
        for cmd in self.commands:
            self.logged.emit("$ " + " ".join(cmd))
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=clean_env,
                )
            except OSError as e:
                self.logged.emit(str(e))
                self.done.emit(False)
                return
            for line in proc.stdout:
                self.logged.emit(line.rstrip())
                m = re.search(r"(\d+)%\|", line)
                if m:
                    self.progress.emit(int(m.group(1)))
            proc.wait()
            if proc.returncode != 0:
                self.done.emit(False)
                return
        self.done.emit(True)


class DependencyDialog(QDialog):
    """Explains a missing CLI dependency and offers a one-click install.

    Installs via ``uv tool install``, bootstrapping uv through Homebrew when uv
    isn't present. Falls back to a copy-command when neither uv nor brew exist.
    Sets ``self.installed`` and calls ``accept()`` once the tool is on disk.
    """

    def __init__(self, parent, dep: _Dependency, log_fn=None) -> None:
        super().__init__(parent)
        self._dep = dep
        self._log_fn = log_fn or (lambda _s: None)
        self.installed = False
        self._thread: Optional[InstallThread] = None

        self.setWindowTitle(f"{dep.label} required")
        layout = QVBoxLayout(self)

        msg = QLabel(f"<b>{dep.label}</b> isn't installed.<br>{dep.reason}")
        msg.setWordWrap(True)
        layout.addWidget(msg)

        uv = find_executable("uv")
        brew = find_executable("brew")
        self._commands = self._plan_commands(dep, uv, brew)

        # Shell-quote for display/copy so pasting into zsh doesn't glob on the
        # brackets in specs like rembg[cli]. (The actual install runs argv-style
        # via subprocess, so it's unaffected either way.)
        def _fmt(cmd):
            return " ".join(shlex.quote(tok) for tok in cmd)

        self._cmd_text = (
            "\n".join(_fmt(c) for c in self._commands)
            if self._commands
            else _fmt(["uv", "tool", "install", "--upgrade", dep.package])
        )
        cmd_box = QPlainTextEdit(self._cmd_text)
        cmd_box.setReadOnly(True)
        cmd_box.setFixedHeight(56)
        cmd_box.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        layout.addWidget(cmd_box)

        if not self._commands:
            hint = QLabel(
                "Neither <code>uv</code> nor Homebrew was found. Install uv "
                "(<code>brew install uv</code>, or see astral.sh/uv), then run the "
                "command above in a terminal."
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setVisible(False)
        self._log.setMinimumHeight(120)
        self._log.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        layout.addWidget(self._log)

        btns = QHBoxLayout()
        btns.addStretch()
        copy_btn = QPushButton("Copy Command")
        copy_btn.clicked.connect(self._copy)
        btns.addWidget(copy_btn)
        self._install_btn = QPushButton(f"Install {dep.label}")
        self._install_btn.setDefault(True)
        self._install_btn.setEnabled(bool(self._commands))
        self._install_btn.clicked.connect(self._start_install)
        btns.addWidget(self._install_btn)
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.reject)
        btns.addWidget(self._close_btn)
        layout.addLayout(btns)

    @staticmethod
    def _plan_commands(dep: _Dependency, uv, brew) -> list[list[str]]:
        def uv_install(uv_path):
            return [uv_path, "tool", "install", "--upgrade", dep.package, *dep.install_args]

        if uv:
            return [uv_install(uv)]
        if brew:
            # Homebrew installs uv into its own bin dir, right next to brew.
            uv_after = os.path.join(os.path.dirname(brew), "uv")
            return [[brew, "install", "uv"], uv_install(uv_after)]
        return []

    def _copy(self) -> None:
        from PyQt5.QtWidgets import QApplication
        QApplication.clipboard().setText(self._cmd_text)

    def _start_install(self) -> None:
        self._install_btn.setEnabled(False)
        self._close_btn.setEnabled(False)  # avoid tearing down a running thread
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._log.setVisible(True)
        self._thread = InstallThread(self._commands)
        self._thread.logged.connect(self._on_log)
        self._thread.progress.connect(self._progress.setValue)
        self._thread.done.connect(self._on_done)
        self._thread.start()

    def _on_log(self, text: str) -> None:
        self._log.appendPlainText(text)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
        self._log_fn(text)

    def _on_done(self, ok: bool) -> None:
        # Trust the filesystem, not just the exit code: uv may succeed yet land
        # the binary somewhere we didn't expect.
        self.installed = all(find_executable(e) for e in self._dep.executables)
        self._close_btn.setEnabled(True)
        if self.installed:
            self.accept()
        else:
            self._progress.setVisible(False)
            self._install_btn.setEnabled(True)
            self._on_log(
                "Installation did not complete — see the log above, or run the "
                "command manually."
            )


class _FocusOutSignal(QObject):
    """Emits focusLost when the watched widget loses focus."""
    focusLost = pyqtSignal()

    def __init__(self, widget: QWidget) -> None:
        super().__init__(widget)
        widget.installEventFilter(self)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.FocusOut:
            self.focusLost.emit()
        return False


# ======================================================================
# Helpers to snap continuous angle values to the nearest discrete option.
# ======================================================================

def _snap_to_nearest(value: int, mapping: list[tuple[int, str]]) -> str:
    """Return the description for the nearest key in *mapping*."""
    return min(mapping, key=lambda kv: abs(value - kv[0]))[1]


def _snap_to_nearest_wrap(value: int, mapping: list[tuple[int, str]], wrap: int = 360) -> str:
    """Return the description for the nearest key, wrapping around *wrap*."""
    def dist(kv):
        return min(abs(value - kv[0]), wrap - abs(value - kv[0]))
    return min(mapping, key=dist)[1]


class KritaiDocker(DockWidget):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Kritai")
        self._thread: Optional[GenerateThread] = None
        self._tmp_input: Optional[str] = None
        self._tmp_output: Optional[str] = None
        self._last_canvas_hash: Optional[str] = None
        self._current_doc = None  # krita.Document
        self._doc_previews: dict[str, QPixmap] = {}
        self._doc_settings: dict[str, dict] = {}
        self._upscale_settings: dict[str, dict] = {}
        self._edit_selection_bounds: Optional[tuple] = None
        # Where a result should land on import: (x, y, w, h) for a selection-
        # scoped cutout, or None for a full-canvas result. Tracked per document
        # (keyed like _doc_previews) so it survives until the user clicks Use.
        self._result_bounds: dict[str, Optional[tuple]] = {}
        self._active_result_bounds: Optional[tuple] = None  # for the in-flight run

        # Flush settings to annotation on save and on application close.
        Krita.instance().notifier().imageSaved.connect(self._on_image_saved)
        Krita.instance().notifier().applicationClosing.connect(self._on_application_closing)

        # Polling timer: checks canvas content periodically when auto is on.
        self._poll_timer = QTimer()
        self._poll_timer.setInterval(POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_canvas)

        # Debounce timer: fires after inactivity to trigger generation.
        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._generate)

        self._build_ui()
        self._connect_settings_signals()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        self.setWidget(scroll)

        root = QWidget()
        scroll.setWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # --- Tab widget ---
        self._tabs = _AdaptiveTabWidget()
        self._tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._tabs.addTab(self._build_generate_tab(), "Generate")
        self._tabs.addTab(self._build_edit_tab(), "Edit")
        self._tabs.addTab(self._build_angle_tab(), "Frame")
        self._tabs.addTab(self._build_mask_tab(), "Mask")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        outer.addWidget(self._tabs)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        outer.addLayout(btn_row)

        self._auto_btn = QToolButton()
        self._auto_btn.setCheckable(True)
        self._auto_btn.setToolTip("Live refresh — regenerate automatically whenever the canvas or a setting changes")
        self._auto_btn.setIcon(Krita.instance().icon("reload-preset"))
        self._auto_btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self._auto_btn.toggled.connect(self._on_auto_toggled)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.clicked.connect(self._generate)
        self._generate_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._generate_btn.setEnabled(False)
        self._generate_btn.setToolTip("No active document.")
        self._generate_btn.setDefault(True)
        btn_row.addWidget(self._generate_btn)
        btn_row.addWidget(self._auto_btn)


        self._clear_preview_btn = QToolButton()
        self._clear_preview_btn.setToolTip("Clear preview")
        self._clear_preview_btn.setIcon(Krita.instance().icon("edit-clear"))
        self._clear_preview_btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self._clear_preview_btn.setEnabled(False)
        self._clear_preview_btn.clicked.connect(self._clear_preview)
        btn_row.addWidget(self._clear_preview_btn)

        self._use_btn = QToolButton()
        self._use_btn.setToolTip("Add result as a new layer in the document")
        self._use_btn.setIcon(Krita.instance().icon("document-new"))
        self._use_btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self._use_btn.setEnabled(False)
        self._use_btn.clicked.connect(self._import_to_layer)
        btn_row.addWidget(self._use_btn)

        self._log_btn = QToolButton()
        self._log_btn.setCheckable(True)
        self._log_btn.setToolTip("Show generation logs")
        self._log_btn.setIcon(Krita.instance().icon("view-list-text"))
        self._log_btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self._log_btn.toggled.connect(self._on_log_toggled)
        btn_row.addWidget(self._log_btn)

        # --- Progress bar + cancel button ---
        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 0, 0, 0)
        progress_row.setSpacing(4)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("")
        self._progress.setVisible(False)
        progress_row.addWidget(self._progress)
        self._cancel_btn = QToolButton()
        self._cancel_btn.setIcon(Krita.instance().icon("dialog-cancel"))
        self._cancel_btn.setToolTip("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel)
        progress_row.addWidget(self._cancel_btn)
        outer.addLayout(progress_row)

        # --- Log section ---
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        self._log.setMaximumHeight(240)
        self._log.setVisible(False)
        self._log.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))

        self._log_btns = QWidget()
        self._log_btns.setVisible(False)
        log_btns_layout = QHBoxLayout(self._log_btns)
        log_btns_layout.setContentsMargins(0, 0, 0, 0)
        log_btns_layout.setSpacing(4)
        self._copy_log_btn = QPushButton("Copy")
        self._copy_log_btn.clicked.connect(self._copy_log)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_log)
        log_btns_layout.addWidget(self._copy_log_btn)
        log_btns_layout.addWidget(self._clear_btn)
        log_btns_layout.addStretch()

        # --- Preview image — placed directly under progress, above log ---
        self._preview = PreviewLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._preview.setStyleSheet("border-radius: 4px;")
        self._preview.setMaximumHeight(240)
        self._preview.setVisible(False)
        outer.addWidget(self._log)
        outer.addWidget(self._log_btns)

        outer.addWidget(self._preview)

        self._time_label = QLabel()
        self._time_label.setAlignment(Qt.AlignLeft)
        self._time_label.setStyleSheet("color: #888; font-size: 11px;")
        self._time_label.setVisible(False)
        outer.addWidget(self._time_label)

        outer.addStretch()

        self._update_generate_btn()

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _tab_description(self, text: str) -> QLabel:
        """A muted, wrapping caption shown at the top of a tab."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #888; font-size: 11px;")
        return label

    def _build_generate_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._tab_description(
            "Generates a new image from your prompt, using the current canvas "
            "as the starting point."
        ))

        # --- Model selector ---
        self._gen_model = QComboBox()
        model_tooltips = {
            "flux2-klein-4b":      "Fast distilled 4B model. Good default for quick iterations. No guidance.",
            "flux2-klein-9b":      "Distilled 9B model. Higher quality than 4B but slower. No guidance.",
            "flux2-klein-base-4b": "Non-distilled 4B base model. Supports guidance. Needs more steps.",
            "flux2-klein-base-9b": "Non-distilled 9B base model. Best quality in the FLUX.2 family.",
        }
        for m in GENERATE_MODELS:
            self._gen_model.addItem(m)
            idx = self._gen_model.count() - 1
            self._gen_model.setItemData(idx, model_tooltips.get(m, ""), Qt.ToolTipRole)
        self._gen_model.setCurrentIndex(0)
        self._gen_model.currentIndexChanged.connect(self._update_generate_tab_ui)

        # --- Prompt ---
        self._gen_prompt = QPlainTextEdit()
        self._gen_prompt.setPlaceholderText("Prompt...")
        self._gen_prompt.setToolTip("Describe what you want the image to look like.")
        self._gen_prompt.setFixedHeight(60)


        # --- Settings form ---
        settings_widget = QWidget()
        form = QFormLayout(settings_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(20)

        form.addRow("Model", self._gen_model)

        self._gen_quantize = self._make_quantize_combo()
        form.addRow("Quantize", self._gen_quantize)

        self._gen_steps = QSpinBox()
        self._gen_steps.setRange(1, 100)
        self._gen_steps.setValue(8)
        self._gen_steps.setToolTip(
            "Number of denoising steps. More steps = higher quality but slower.\n"
            "Distilled models (klein) work well with 4–8 steps.\n"
            "Base models need 20–50 steps."
        )
        form.addRow("Steps", self._gen_steps)

        self._gen_guidance = QDoubleSpinBox()
        self._gen_guidance.setRange(0.0, 30.0)
        self._gen_guidance.setSingleStep(0.5)
        self._gen_guidance.setValue(3.5)
        self._gen_guidance.setToolTip(
            "How closely the result follows your prompt.\n"
            "Higher = more literal, lower = more creative.\n"
            "Not supported by distilled models."
        )
        form.addRow("Guidance", self._gen_guidance)
        self._gen_guidance_label = form.labelForField(self._gen_guidance)

        strength_row, self._gen_strength, self._gen_strength_spin = self._make_slider_row(
            0, 100, 75, "How much the canvas influences the result.\n"
            "0.0 = output ignores your painting entirely.\n"
            "1.0 = output stays very close to the canvas.\n"
            "0.5–0.7 is a good starting range."
        )
        form.addRow("Strength", strength_row)
        self._gen_strength_row = strength_row
        self._gen_strength_label = form.labelForField(strength_row)

        scale_row, self._gen_scale, self._gen_scale_spin = self._make_slider_row(
            0, 100, 50, "Scale of the canvas sent to mflux relative to its original size.\n"
            "0.5 = half resolution (faster, less VRAM).\n"
            "1.0 = full resolution."
        )
        form.addRow("Scale", scale_row)

        seed_row, self._gen_seed, self._gen_random_seed = self._make_seed_row()
        form.addRow("Seed", seed_row)

        layout.addWidget(self._gen_prompt)
        layout.addWidget(settings_widget)

        # --- LoRA section ---
        gen_lora_section = CollapsibleSection("LoRAs")
        gen_lora_content = QVBoxLayout()
        gen_lora_content.setSpacing(4)
        self._gen_lora_list = QVBoxLayout()
        self._gen_lora_list.setSpacing(4)
        gen_lora_content.addLayout(self._gen_lora_list)
        self._gen_lora_entries: list[tuple] = []

        add_lora_btn = QToolButton()
        add_lora_btn.setIcon(Krita.instance().icon("list-add"))
        add_lora_btn.setToolTip("Add LoRA")
        add_lora_btn.clicked.connect(lambda: self._add_lora_row(self._gen_lora_entries, self._gen_lora_list))
        gen_lora_content.addWidget(add_lora_btn)
        gen_lora_section.setContentLayout(gen_lora_content)
        layout.addWidget(gen_lora_section)

        layout.addStretch()
        content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._update_generate_tab_ui()
        return content

    def _build_edit_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._tab_description(
            "Edits the current canvas following your prompt. Optionally add "
            "reference images to steer the result."
        ))

        # --- Prompt ---
        self._edit_prompt = QPlainTextEdit()
        self._edit_prompt.setPlaceholderText("Prompt...")
        self._edit_prompt.setFixedHeight(60)

        # --- Settings form ---
        settings_widget = QWidget()
        form = QFormLayout(settings_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(20)

        self._edit_model = QComboBox()
        edit_model_tooltips = {
            "flux2-klein-4b":      "Distilled 4B model. Fast, no guidance.",
            "flux2-klein-9b":      "Distilled 9B model. Higher quality, no guidance.",
            "flux2-klein-base-4b": "Base 4B model. Slower, supports guidance.",
            "flux2-klein-base-9b": "Base 9B model. Best quality, supports guidance.",
        }
        for m in ANGLE_MODELS:
            self._edit_model.addItem(m)
            idx = self._edit_model.count() - 1
            self._edit_model.setItemData(idx, edit_model_tooltips.get(m, ""), Qt.ToolTipRole)
        self._edit_model.setCurrentIndex(2)  # default to base-4b for edit quality
        self._edit_model.currentIndexChanged.connect(self._update_edit_tab_ui)
        form.addRow("Model", self._edit_model)

        self._edit_quantize = self._make_quantize_combo()
        form.addRow("Quantize", self._edit_quantize)

        self._edit_steps = QSpinBox()
        self._edit_steps.setRange(1, 100)
        self._edit_steps.setValue(8)
        self._edit_steps.setToolTip("Number of denoising steps.")
        form.addRow("Steps", self._edit_steps)

        self._edit_guidance = QDoubleSpinBox()
        self._edit_guidance.setRange(0.0, 30.0)
        self._edit_guidance.setSingleStep(0.5)
        self._edit_guidance.setValue(3.5)
        self._edit_guidance.setToolTip("How closely the result follows your prompt.")
        form.addRow("Guidance", self._edit_guidance)
        self._edit_guidance_label = form.labelForField(self._edit_guidance)

        strength_row, self._edit_strength, self._edit_strength_spin = self._make_slider_row(
            0, 100, 75, "How much the canvas influences the result."
        )
        form.addRow("Strength", strength_row)
        self._edit_strength_row = strength_row
        self._edit_strength_label = form.labelForField(strength_row)

        scale_row, self._edit_scale, self._edit_scale_spin = self._make_slider_row(
            0, 100, 50, "Scale of the canvas sent to mflux relative to its original size."
        )
        form.addRow("Scale", scale_row)

        seed_row, self._edit_seed, self._edit_random_seed = self._make_seed_row()
        form.addRow("Seed", seed_row)

        layout.addWidget(self._edit_prompt)
        layout.addWidget(settings_widget)

        # --- Reference images (flux2-edit only) ---
        self._edit_ref_section = CollapsibleSection("Reference Images")
        ref_content = QVBoxLayout()
        ref_content.setSpacing(4)
        self._edit_ref_list = QVBoxLayout()
        self._edit_ref_list.setSpacing(4)
        ref_content.addLayout(self._edit_ref_list)
        self._edit_ref_entries: list[tuple] = []

        add_ref_btn = QToolButton()
        add_ref_btn.setIcon(Krita.instance().icon("list-add"))
        add_ref_btn.setToolTip("Add reference image")
        add_ref_btn.clicked.connect(lambda: self._add_ref_row(self._edit_ref_entries, self._edit_ref_list))
        ref_content.addWidget(add_ref_btn)
        self._edit_ref_section.setContentLayout(ref_content)
        self._edit_ref_section.setVisible(False)
        layout.addWidget(self._edit_ref_section)

        # --- LoRA section ---
        edit_lora_section = CollapsibleSection("LoRAs")
        edit_lora_content = QVBoxLayout()
        edit_lora_content.setSpacing(4)
        self._edit_lora_list = QVBoxLayout()
        self._edit_lora_list.setSpacing(4)
        edit_lora_content.addLayout(self._edit_lora_list)
        self._edit_lora_entries: list[tuple] = []

        add_lora_btn = QToolButton()
        add_lora_btn.setIcon(Krita.instance().icon("list-add"))
        add_lora_btn.setToolTip("Add LoRA")
        add_lora_btn.clicked.connect(lambda: self._add_lora_row(self._edit_lora_entries, self._edit_lora_list))
        edit_lora_content.addWidget(add_lora_btn)
        edit_lora_section.setContentLayout(edit_lora_content)
        layout.addWidget(edit_lora_section)

        layout.addStretch()
        content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._update_edit_tab_ui()
        return content

    def _build_angle_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._tab_description(
            "Re-renders the subject from a new camera angle while keeping its "
            "style, lighting, and background."
        ))

        # --- 3-D orbit widget + angle controls side by side ---
        orbit_row = QHBoxLayout()
        orbit_row.setSpacing(8)

        self._orbit = CameraOrbitWidget()
        orbit_row.addWidget(self._orbit)

        # Numeric readouts (synced to orbit widget).
        cam_form = QFormLayout()
        cam_form.setContentsMargins(0, 0, 0, 0)
        cam_form.setSpacing(4)
        cam_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        cam_form.setHorizontalSpacing(8)
        # On macOS the default is ExpandingFieldsGrow (only Expanding-policy
        # widgets grow).  Force AllNonFixedFieldsGrow so that our Preferred-
        # policy row widgets also fill the available right-column width.
        cam_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        def _int_slider_row(lo, hi, default, suffix, tooltip):
            row = QWidget()
            # Expanding policy is required for AllNonFixedFieldsGrow to stretch
            # the row to fill the right column (Preferred alone is not enough on
            # macOS even with AllNonFixedFieldsGrow).
            row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            hl  = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(4)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(default)
            slider.setToolTip(tooltip)
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(default)
            spin.setSuffix(suffix)
            # Keep the spinbox compact so the slider has room in narrow dockers.
            spin.setFixedWidth(50)
            spin.setToolTip(tooltip)
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            hl.addWidget(slider)
            hl.addWidget(spin)
            return row, slider, spin

        az_row, self._angle_azimuth, self._angle_azimuth_spin = _int_slider_row(
            -180, 180, 0, "°", "Horizontal rotation around the subject (0° = front).")
        self._angle_azimuth.valueChanged.connect(self._orbit.setAzimuth)
        self._orbit.azimuth_changed.connect(self._angle_azimuth.setValue)
        cam_form.addRow("Azimuth", az_row)

        el_row, self._angle_elevation, self._angle_elevation_spin = _int_slider_row(
            -90, 90, 0, "°", "Vertical angle (−90° = bottom, 0° = eye-level, 90° = top).")
        self._angle_elevation.valueChanged.connect(self._orbit.setElevation)
        self._orbit.elevation_changed.connect(self._angle_elevation.setValue)
        cam_form.addRow("Elevation", el_row)

        dist_row, self._angle_distance, self._angle_distance_spin = _int_slider_row(
            60, 180, 100, "", "Camera distance (60 = close-up, 100 = medium, 180 = wide).")
        self._angle_distance.valueChanged.connect(self._orbit.setDistance)
        self._orbit.distance_changed.connect(self._angle_distance.setValue)
        cam_form.addRow("Distance", dist_row)

        cam_form_widget = QWidget()
        cam_form_widget.setLayout(cam_form)
        cam_form_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Right column: sliders only — wrapped in a widget so the HBoxLayout
        # can stretch it properly.
        right_col_widget = QWidget()
        right_col_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        right_col = QVBoxLayout(right_col_widget)
        right_col.setSpacing(4)
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.addWidget(cam_form_widget)
        right_col.addStretch()

        orbit_row.addWidget(right_col_widget, stretch=1)
        orbit_row.setAlignment(Qt.AlignTop)
        layout.addLayout(orbit_row)

        # Preset buttons span the full content width so they never force the
        # right column (and therefore the orbit row) to be too wide.
        preset_row = QHBoxLayout()
        preset_row.setSpacing(3)
        preset_row.setContentsMargins(0, 0, 0, 0)
        for name, (az, el, dist) in ANGLE_PRESETS.items():
            btn = QPushButton(name)
            btn.setToolTip(f"Azimuth {az}°, Elevation {el}°, Distance {dist}")
            btn.clicked.connect(lambda checked=False, a=az, e=el, d=dist: self._apply_angle_preset(a, e, d))
            preset_row.addWidget(btn, stretch=1)
        layout.addLayout(preset_row)

        # --- Settings form ---
        settings_widget = QWidget()
        form = QFormLayout(settings_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(20)

        self._angle_model = QComboBox()
        angle_model_tooltips = {
            "flux2-klein-4b":      "Distilled 4B model. Fast, no guidance.",
            "flux2-klein-9b":      "Distilled 9B model. Higher quality, no guidance.",
            "flux2-klein-base-4b": "Base 4B model. Slower, supports guidance.",
            "flux2-klein-base-9b": "Base 9B model. Best quality, supports guidance.",
        }
        for m in ANGLE_MODELS:
            self._angle_model.addItem(m)
            idx = self._angle_model.count() - 1
            self._angle_model.setItemData(idx, angle_model_tooltips.get(m, ""), Qt.ToolTipRole)
        self._angle_model.setCurrentIndex(0)
        self._angle_model.currentIndexChanged.connect(self._update_angle_tab_ui)
        form.addRow("Model", self._angle_model)

        self._angle_quantize = self._make_quantize_combo()
        form.addRow("Quantize", self._angle_quantize)

        self._angle_steps = QSpinBox()
        self._angle_steps.setRange(1, 100)
        self._angle_steps.setValue(8)
        self._angle_steps.setToolTip("Number of denoising steps.")
        form.addRow("Steps", self._angle_steps)

        self._angle_guidance = QDoubleSpinBox()
        self._angle_guidance.setRange(0.0, 20.0)
        self._angle_guidance.setSingleStep(0.5)
        self._angle_guidance.setValue(3.5)
        self._angle_guidance.setToolTip("Guidance scale (classifier-free guidance strength).")
        form.addRow("Guidance", self._angle_guidance)
        self._angle_guidance_label = form.labelForField(self._angle_guidance)

        scale_row, self._angle_scale, self._angle_scale_spin = self._make_slider_row(
            0, 100, 50, "Scale of the canvas sent to mflux relative to its original size."
        )
        form.addRow("Scale", scale_row)

        seed_row, self._angle_seed, self._angle_random_seed = self._make_seed_row()
        form.addRow("Seed", seed_row)

        layout.addWidget(settings_widget)
        layout.addStretch()
        content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._update_angle_tab_ui()
        return content

    # ------------------------------------------------------------------
    # Widget factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_slider_row(min_val: int, max_val: int, default: int, tooltip: str) -> tuple:
        """Create a slider+spinbox row. Returns (row_widget, slider, spinbox)."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        spin = QDoubleSpinBox()
        spin.setRange(min_val / 100, max_val / 100)
        spin.setSingleStep(0.01)
        spin.setDecimals(2)
        spin.setValue(default / 100)
        spin.setFixedWidth(60)
        slider.valueChanged.connect(lambda v: spin.setValue(v / 100))
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * 100)))
        layout.addWidget(slider)
        layout.addWidget(spin)
        row.setToolTip(tooltip)
        return row, slider, spin

    @staticmethod
    def _make_seed_row() -> tuple:
        """Create a seed+random checkbox row. Returns (row_widget, seed_spin, random_cb)."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        seed = QSpinBox()
        seed.setRange(0, 2_000_000_000)
        seed.setValue(0)
        random_cb = QCheckBox("Random")
        random_cb.setChecked(False)
        random_cb.toggled.connect(seed.setDisabled)
        seed.setDisabled(False)
        layout.addWidget(seed)
        layout.addWidget(random_cb)
        row.setToolTip(
            "Fixed seed produces the same result every time given the same inputs.\n"
            "Random seed gives a different result on each generation."
        )
        return row, seed, random_cb

    # ------------------------------------------------------------------
    # Tab-specific UI updates
    # ------------------------------------------------------------------

    def _build_mask_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._tab_description(
            "Removes the background from the current canvas (or just the active "
            "selection, if there is one) and previews the result as a transparent "
            "image, ready to import as a new layer. Model weights download on "
            "first use."
        ))

        self._mask_model = QComboBox()
        model_tooltips = {
            "birefnet-general":  "Highest edge quality. Larger download, slower first run.",
            "isnet-general-use": "Balanced quality and speed. Good default.",
            "u2net":             "Fastest. Solid for clear, simple subjects.",
        }
        for m in MASK_MODELS:
            self._mask_model.addItem(m)
            idx = self._mask_model.count() - 1
            self._mask_model.setItemData(idx, model_tooltips.get(m, ""), Qt.ToolTipRole)
        self._mask_model.setCurrentIndex(0)

        self._mask_alpha_matting = QCheckBox("Refine Edges (Alpha Matting)")
        self._mask_alpha_matting.setToolTip(
            "Post-process the cutout for cleaner, softer edges (hair, fur). Slower."
        )

        settings_widget = QWidget()
        form = QFormLayout(settings_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setHorizontalSpacing(20)
        form.addRow("Model", self._mask_model)
        form.addRow("", self._mask_alpha_matting)
        layout.addWidget(settings_widget)

        layout.addStretch()
        content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        return content

    def _on_tab_changed(self, index: int) -> None:
        self._save_settings()
        self._update_generate_btn()
        # Tell the parent layout that our preferred size has changed so it
        # re-flows without stretching the tab widget to fill leftover space.
        self._tabs.updateGeometry()

    def _update_generate_tab_ui(self) -> None:
        model_name = self._gen_model.currentText()
        _, _, supports_strength, supports_guidance, *_ = MODEL_CLI.get(
            model_name, (None, None, True, True, False)
        )
        self._gen_guidance.setVisible(supports_guidance)
        if self._gen_guidance_label:
            self._gen_guidance_label.setVisible(supports_guidance)
        self._gen_strength_row.setVisible(supports_strength)
        if self._gen_strength_label:
            self._gen_strength_label.setVisible(supports_strength)

    def _update_edit_tab_ui(self) -> None:
        is_base = "base" in self._edit_model.currentText()
        self._edit_guidance.setVisible(is_base)
        if self._edit_guidance_label:
            self._edit_guidance_label.setVisible(is_base)
        self._edit_strength_row.setVisible(False)
        if self._edit_strength_label:
            self._edit_strength_label.setVisible(False)
        self._edit_ref_section.setVisible(True)
        self._edit_prompt.setPlaceholderText("Describe what you want to change...")
        self._edit_prompt.setToolTip("Describe what you want to change in the canvas.")

    def _update_angle_tab_ui(self) -> None:
        is_base = "base" in self._angle_model.currentText()
        self._angle_guidance.setVisible(is_base)
        if self._angle_guidance_label:
            self._angle_guidance_label.setVisible(is_base)

    # ------------------------------------------------------------------
    # Angle helpers
    # ------------------------------------------------------------------

    def _build_angle_prompt(self) -> str:
        az_desc   = _snap_to_nearest_wrap(self._angle_azimuth.value(), AZIMUTH_MAP)
        el_desc   = _snap_to_nearest(self._angle_elevation.value(), ELEVATION_MAP)
        dist_desc = _snap_to_nearest(self._angle_distance.value(), DISTANCE_MAP)
        return (
            f"Show the subject from a {az_desc}, {el_desc}, {dist_desc}. "
            f"Keep the same subject, style, lighting, and background."
        )

    def _apply_angle_preset(self, azimuth: int, elevation: int, distance: int) -> None:
        self._angle_azimuth.setValue(azimuth)
        self._angle_elevation.setValue(elevation)
        self._angle_distance.setValue(distance)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    ANNOTATION_TYPE = "kritai_settings"

    def _connect_settings_signals(self) -> None:
        # Generate tab signals.
        for signal in [
            self._gen_prompt.textChanged,
            self._gen_model.currentIndexChanged,
            self._gen_quantize.currentIndexChanged,
            self._gen_steps.valueChanged,
            self._gen_guidance.valueChanged,
            self._gen_strength.valueChanged,
            self._gen_scale.valueChanged,
            self._gen_seed.valueChanged,
            self._gen_random_seed.toggled,
        ]:
            signal.connect(self._save_settings)

        # Edit tab signals.
        for signal in [
            self._edit_prompt.textChanged,
            self._edit_quantize.currentIndexChanged,
            self._edit_steps.valueChanged,
            self._edit_guidance.valueChanged,
            self._edit_strength.valueChanged,
            self._edit_scale.valueChanged,
            self._edit_seed.valueChanged,
            self._edit_random_seed.toggled,
        ]:
            signal.connect(self._save_settings)

        # Angle tab signals.
        for signal in [
            self._angle_model.currentIndexChanged,
            self._angle_azimuth.valueChanged,
            self._angle_elevation.valueChanged,
            self._angle_distance.valueChanged,
            self._angle_quantize.currentIndexChanged,
            self._angle_steps.valueChanged,
            self._angle_guidance.valueChanged,
            self._angle_scale.valueChanged,
            self._angle_seed.valueChanged,
            self._angle_random_seed.toggled,
        ]:
            signal.connect(self._save_settings)

        # Mask tab signals.
        for signal in [
            self._mask_model.currentIndexChanged,
            self._mask_alpha_matting.toggled,
        ]:
            signal.connect(self._save_settings)

        # Auto-refresh triggers — text fields on focus-out, others immediately.
        gen_prompt_filter = _FocusOutSignal(self._gen_prompt)
        gen_prompt_filter.focusLost.connect(self._on_setting_changed)

        edit_prompt_filter = _FocusOutSignal(self._edit_prompt)
        edit_prompt_filter.focusLost.connect(self._on_setting_changed)

        for signal in [
            self._gen_model.currentIndexChanged,
            self._gen_quantize.currentIndexChanged,
            self._gen_steps.valueChanged,
            self._gen_guidance.valueChanged,
            self._gen_strength.valueChanged,
            self._gen_scale.valueChanged,
            self._gen_seed.valueChanged,
            self._gen_random_seed.toggled,
            self._edit_quantize.currentIndexChanged,
            self._edit_steps.valueChanged,
            self._edit_guidance.valueChanged,
            self._edit_strength.valueChanged,
            self._edit_scale.valueChanged,
            self._edit_seed.valueChanged,
            self._edit_random_seed.toggled,
            self._angle_model.currentIndexChanged,
            self._angle_azimuth.valueChanged,
            self._angle_elevation.valueChanged,
            self._angle_distance.valueChanged,
            self._angle_quantize.currentIndexChanged,
            self._angle_steps.valueChanged,
            self._angle_guidance.valueChanged,
            self._angle_scale.valueChanged,
            self._angle_seed.valueChanged,
            self._angle_random_seed.toggled,
            self._mask_model.currentIndexChanged,
            self._mask_alpha_matting.toggled,
        ]:
            signal.connect(self._on_setting_changed)

        # Generate button update on prompt change.
        self._gen_prompt.textChanged.connect(self._update_generate_btn)
        self._edit_prompt.textChanged.connect(self._update_generate_btn)

    def _on_setting_changed(self) -> None:
        if self._auto_btn.isChecked():
            self._debounce_timer.start()

    def _save_settings(self) -> None:
        doc = self._current_doc
        if not doc:
            return
        uid = doc.fileName() or str(id(doc))
        new = {
            "active_tab": self._tabs.currentIndex(),
            "generate": {
                "model": self._gen_model.currentText(),
                "prompt": self._gen_prompt.toPlainText(),
                "quantize": self._quantize_value(self._gen_quantize),
                "steps": self._gen_steps.value(),
                "guidance": self._gen_guidance.value(),
                "strength": self._gen_strength.value() / 100,
                "scale": self._gen_scale.value() / 100,
                "seed": self._gen_seed.value(),
                "random_seed": self._gen_random_seed.isChecked(),
                "loras": [
                    {"path": p.text().strip(), "scale": s.value(), "enabled": e.isChecked()}
                    for e, p, s, _ in self._gen_lora_entries if p.text().strip()
                ],
            },
            "edit": {
                "model": self._edit_model.currentText(),
                "prompt": self._edit_prompt.toPlainText(),
                "quantize": self._quantize_value(self._edit_quantize),
                "steps": self._edit_steps.value(),
                "guidance": self._edit_guidance.value(),
                "strength": self._edit_strength.value() / 100,
                "scale": self._edit_scale.value() / 100,
                "seed": self._edit_seed.value(),
                "random_seed": self._edit_random_seed.isChecked(),
                "reference_images": [
                    {"path": t.imagePath() or "", "enabled": e.isChecked()}
                    for e, t, _ in self._edit_ref_entries
                    if (t.imagePath() or "").strip()
                ],
                "loras": [
                    {"path": p.text().strip(), "scale": s.value(), "enabled": e.isChecked()}
                    for e, p, s, _ in self._edit_lora_entries if p.text().strip()
                ],
            },
            "angle": {
                "model": self._angle_model.currentText(),
                "azimuth": self._angle_azimuth.value(),
                "elevation": self._angle_elevation.value(),
                "distance": self._angle_distance.value(),
                "quantize": self._quantize_value(self._angle_quantize),
                "steps": self._angle_steps.value(),
                "guidance": self._angle_guidance.value(),
                "scale": self._angle_scale.value() / 100,
                "seed": self._angle_seed.value(),
                "random_seed": self._angle_random_seed.isChecked(),
            },
            "mask": {
                "model": self._mask_model.currentText(),
                "alpha_matting": self._mask_alpha_matting.isChecked(),
            },
            "upscale": self._upscale_settings.get(uid, {}),
            "preview_path": self._tmp_output or "",
        }
        prev = self._doc_settings.get(uid)
        self._doc_settings[uid] = new
        if prev != new:
            # Write annotation immediately so it's included in any subsequent
            # Krita save — imageSaved fires *after* the file is written, so
            # deferring to _flush_settings_to_doc would miss the current save.
            if doc.fileName():
                raw = json.dumps(new).encode("utf-8")
                doc.setAnnotation(self.ANNOTATION_TYPE, "Kritai settings", QByteArray(raw))
            doc.setModified(True)

    def _flush_settings_to_doc(self, doc: object) -> None:
        """Write in-memory settings for *doc* into its KRA annotation."""
        filename = doc.fileName()
        if not filename:
            return
        uid = filename
        # Migrate settings stored under the old id-based key (before first save).
        old_uid = str(id(doc))
        if old_uid != uid and old_uid in self._doc_settings:
            self._doc_settings[uid] = self._doc_settings.pop(old_uid)
        data = self._doc_settings.get(uid)
        if data:
            raw = json.dumps(data).encode("utf-8")
            doc.setAnnotation(self.ANNOTATION_TYPE, "Kritai settings", QByteArray(raw))
            doc.setModified(False)

    def _on_image_saved(self, filename: str) -> None:
        """Flush in-memory settings to the document annotation on save."""
        self._save_settings()
        for doc in Krita.instance().documents():
            if doc.fileName() == filename:
                self._flush_settings_to_doc(doc)
                break

    def _on_application_closing(self) -> None:
        """Flush settings for all open documents before Krita quits."""
        self._save_settings()
        for doc in Krita.instance().documents():
            if doc.fileName():
                self._flush_settings_to_doc(doc)

    def _load_settings(self, doc: object) -> None:
        uid = doc.fileName() or str(id(doc))
        if uid in self._doc_settings:
            data = self._doc_settings[uid]
        else:
            # First time seeing this doc this session — load from saved annotation.
            raw = doc.annotation(self.ANNOTATION_TYPE)
            if not raw:
                return
            try:
                data = json.loads(bytes(raw).decode("utf-8"))
            except Exception:
                return
            self._doc_settings[uid] = data

        # Migrate old flat format.
        if "active_tab" not in data:
            data = self._migrate_old_settings(data, uid)

        # Block signals while restoring.
        all_widgets = [
            self._gen_prompt, self._gen_model,
            self._gen_quantize, self._gen_steps, self._gen_guidance,
            self._gen_strength, self._gen_scale, self._gen_seed, self._gen_random_seed,
            self._edit_prompt,
            self._edit_quantize, self._edit_steps, self._edit_guidance,
            self._edit_strength, self._edit_scale, self._edit_seed, self._edit_random_seed,
            self._angle_model,
            self._angle_azimuth, self._angle_azimuth_spin,
            self._angle_elevation, self._angle_elevation_spin,
            self._angle_distance, self._angle_distance_spin,
            self._orbit, self._angle_quantize, self._angle_steps, self._angle_guidance,
            self._angle_scale, self._angle_seed, self._angle_random_seed,
            self._mask_model, self._mask_alpha_matting,
            self._tabs,
        ]
        for w in all_widgets:
            w.blockSignals(True)

        # --- Restore Generate tab ---
        gen = data.get("generate", {})
        self._gen_prompt.setPlainText(gen.get("prompt", ""))
        idx = self._gen_model.findText(gen.get("model", "flux2-klein-4b"))
        if idx >= 0:
            self._gen_model.setCurrentIndex(idx)
        self._set_quantize_combo(self._gen_quantize, gen.get("quantize", 4))
        self._gen_steps.setValue(gen.get("steps", 8))
        self._gen_guidance.setValue(gen.get("guidance", 3.5))
        sv = int(gen.get("strength", 0.75) * 100)
        self._gen_strength.setValue(sv)
        self._gen_strength_spin.setValue(sv / 100)
        scv = int(gen.get("scale", 0.5) * 100)
        self._gen_scale.setValue(scv)
        self._gen_scale_spin.setValue(scv / 100)
        self._gen_seed.setValue(gen.get("seed", 0))
        self._gen_random_seed.setChecked(gen.get("random_seed", False))
        self._gen_seed.setDisabled(self._gen_random_seed.isChecked())
        # Restore generate LoRAs.
        for *_, row in list(self._gen_lora_entries):
            self._gen_lora_list.removeWidget(row)
            row.deleteLater()
        self._gen_lora_entries.clear()
        for lora in gen.get("loras", []):
            self._add_lora_row(self._gen_lora_entries, self._gen_lora_list,
                               lora.get("path", ""), lora.get("scale", 1.0), lora.get("enabled", True))

        # --- Restore Edit tab ---
        edit = data.get("edit", {})
        idx = self._edit_model.findText(edit.get("model", "flux2-klein-base-4b"))
        if idx >= 0:
            self._edit_model.setCurrentIndex(idx)
        self._edit_prompt.setPlainText(edit.get("prompt", ""))
        self._set_quantize_combo(self._edit_quantize, edit.get("quantize", 4))
        self._edit_steps.setValue(edit.get("steps", 8))
        self._edit_guidance.setValue(edit.get("guidance", 3.5))
        sv = int(edit.get("strength", 0.75) * 100)
        self._edit_strength.setValue(sv)
        self._edit_strength_spin.setValue(sv / 100)
        scv = int(edit.get("scale", 0.5) * 100)
        self._edit_scale.setValue(scv)
        self._edit_scale_spin.setValue(scv / 100)
        self._edit_seed.setValue(edit.get("seed", 0))
        self._edit_random_seed.setChecked(edit.get("random_seed", False))
        self._edit_seed.setDisabled(self._edit_random_seed.isChecked())
        # Restore edit reference images.
        for *_, row in list(self._edit_ref_entries):
            self._edit_ref_list.removeWidget(row)
            row.deleteLater()
        self._edit_ref_entries.clear()
        for ref in edit.get("reference_images", []):
            self._add_ref_row(self._edit_ref_entries, self._edit_ref_list,
                              ref.get("path", ""), ref.get("enabled", True))
        # Restore edit LoRAs.
        for *_, row in list(self._edit_lora_entries):
            self._edit_lora_list.removeWidget(row)
            row.deleteLater()
        self._edit_lora_entries.clear()
        for lora in edit.get("loras", []):
            self._add_lora_row(self._edit_lora_entries, self._edit_lora_list,
                               lora.get("path", ""), lora.get("scale", 1.0), lora.get("enabled", True))

        # --- Restore Angle tab ---
        angle = data.get("angle", {})
        idx = self._angle_model.findText(angle.get("model", "flux2-klein-4b"))
        if idx >= 0:
            self._angle_model.setCurrentIndex(idx)
        az = angle.get("azimuth", 0)
        el = angle.get("elevation", 0)
        dv = angle.get("distance", 100)
        self._angle_azimuth.setValue(az)
        self._angle_azimuth_spin.setValue(az)
        self._angle_elevation.setValue(el)
        self._angle_elevation_spin.setValue(el)
        self._angle_distance.setValue(dv)
        self._angle_distance_spin.setValue(dv)
        self._orbit.setAzimuth(az)
        self._orbit.setElevation(el)
        self._orbit.setDistance(dv)
        self._set_quantize_combo(self._angle_quantize, angle.get("quantize", 4))
        self._angle_steps.setValue(angle.get("steps", 8))
        self._angle_guidance.setValue(angle.get("guidance", 3.5))
        ascv = int(angle.get("scale", 0.5) * 100)
        self._angle_scale.setValue(ascv)
        self._angle_scale_spin.setValue(ascv / 100)
        self._angle_seed.setValue(angle.get("seed", 0))
        self._angle_random_seed.setChecked(angle.get("random_seed", False))
        self._angle_seed.setDisabled(self._angle_random_seed.isChecked())

        # --- Restore Mask tab ---
        mask = data.get("mask", {})
        midx = self._mask_model.findText(mask.get("model", MASK_MODELS[0]))
        if midx >= 0:
            self._mask_model.setCurrentIndex(midx)
        self._mask_alpha_matting.setChecked(mask.get("alpha_matting", False))

        # --- Restore active tab ---
        self._tabs.setCurrentIndex(data.get("active_tab", 0))

        # --- Restore upscale settings ---
        upscale = data.get("upscale")
        if upscale:
            self._upscale_settings[uid] = upscale

        # --- Restore preview image if the temp file still exists ---
        preview_path = data.get("preview_path", "")
        if preview_path and os.path.exists(preview_path):
            self._tmp_output = preview_path
            pixmap = QPixmap(preview_path)
            if not pixmap.isNull():
                self._preview.setPixmap(pixmap)
                self._doc_previews[uid] = pixmap

        for w in all_widgets:
            w.blockSignals(False)

        self._update_generate_tab_ui()
        self._update_edit_tab_ui()
        self._update_angle_tab_ui()

    def _migrate_old_settings(self, data: dict, uid: str) -> dict:
        """Migrate old flat settings format to new namespaced format."""
        old_model = data.get("model", "flux2-klein-4b")
        common = {
            "prompt": data.get("prompt", ""),
            "quantize": data.get("quantize", 4),
            "steps": data.get("steps", 8),
            "guidance": data.get("guidance", 3.5),
            "strength": data.get("strength", 0.75),
            "scale": data.get("scale", data.get("downscale", data.get("resolution_scale", 0.5))),
            "seed": data.get("seed", 0),
            "random_seed": data.get("random_seed", False),
            "loras": data.get("loras", []),
        }
        if old_model in EDIT_MODELS:
            edit_data = {**common, "model": old_model}
            # Migrate reference images.
            refs = data.get("reference_images", [])
            if not refs and data.get("reference_image"):
                refs = [{"path": data["reference_image"], "prompt": "", "enabled": True}]
            edit_data["reference_images"] = refs
            migrated = {"active_tab": 1, "generate": {}, "edit": edit_data, "angle": {}}
        else:
            gen_data = {**common, "model": old_model}
            migrated = {"active_tab": 0, "generate": gen_data, "edit": {}, "angle": {}}
        migrated["upscale"] = data.get("upscale", {})
        migrated["preview_path"] = data.get("preview_path", "")
        self._doc_settings[uid] = migrated
        return migrated

    # ------------------------------------------------------------------
    # Auto-mode
    # ------------------------------------------------------------------

    def _on_auto_toggled(self, checked: bool) -> None:
        if checked:
            self._last_canvas_hash = self._canvas_hash()
            self._poll_timer.start()
        else:
            self._poll_timer.stop()
            self._debounce_timer.stop()

    def _export_canvas(self, doc: object, path: str) -> None:
        """Export the canvas to *path* at full resolution."""
        doc.setBatchmode(True)
        doc.exportImage(path, InfoObject())
        doc.setBatchmode(False)

    def _canvas_hash(self) -> Optional[str]:
        """Return a hash of a small in-memory thumbnail — no disk I/O."""
        from krita import Krita

        doc = Krita.instance().activeDocument()
        if not doc:
            return None
        try:
            thumb = doc.thumbnail(128, 128)
            if thumb.isNull():
                return None
            ptr = thumb.bits()
            ptr.setsize(thumb.byteCount())
            return hashlib.sha1(bytes(ptr)).hexdigest()
        except Exception:
            return None

    def _update_generate_btn(self) -> None:
        """Enable/disable generate button based on document and prompt state."""
        if not self._current_doc:
            self._generate_btn.setEnabled(False)
            self._generate_btn.setToolTip("No active document.")
            return
        tab = self._tabs.currentIndex()
        if tab == 3:
            self._generate_btn.setEnabled(True)
            self._generate_btn.setToolTip("Remove the background from the current canvas.")
            return
        if tab == 0:
            has_prompt = bool(self._gen_prompt.toPlainText().strip())
        elif tab == 1:
            has_prompt = bool(self._edit_prompt.toPlainText().strip())
        else:
            has_prompt = True  # Angle tab always has auto-generated prompt.
        if not has_prompt:
            self._generate_btn.setEnabled(False)
            self._generate_btn.setToolTip("Enter a prompt to generate.")
            return
        self._generate_btn.setEnabled(True)
        self._generate_btn.setToolTip("Generate an image from the current canvas and prompt.")

    def _poll_canvas(self) -> None:
        if self._thread and self._thread.isRunning():
            return
        current = self._canvas_hash()
        if current and current != self._last_canvas_hash:
            self._last_canvas_hash = current
            self._debounce_timer.start()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._progress.setFormat(text)

    def _set_busy(self, busy: bool) -> None:
        self._progress.setValue(0)
        self._progress.setVisible(busy)
        self._cancel_btn.setEnabled(busy)
        self._cancel_btn.setVisible(busy)
        self._set_status("Initializing..." if busy else "")

    def _cancel(self) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.terminate()
            self._thread.wait()
        self._set_busy(False)

    def _ensure_dependency(self, dep: _Dependency) -> bool:
        """Return True if *dep* is available, prompting to install it if not."""
        if all(find_executable(e) for e in dep.executables):
            return True
        dlg = DependencyDialog(self, dep, log_fn=self._on_log_message)
        dlg.exec_()
        return dlg.installed

    def _generate(self) -> None:
        from krita import Krita

        app = Krita.instance()
        doc = app.activeDocument()
        if not doc:
            return

        tab = self._tabs.currentIndex()
        is_mask = tab == 3
        if is_mask:
            prompt = ""
        elif tab == 0:
            prompt = self._gen_prompt.toPlainText().strip()
        elif tab == 1:
            prompt = self._build_edit_prompt()
        else:
            prompt = self._build_angle_prompt()

        if not is_mask and not prompt:
            return

        # Make sure the CLI this action needs is installed before doing any work.
        if not self._ensure_dependency(DEP_REMBG if is_mask else DEP_MFLUX):
            return

        if self._thread and self._thread.isRunning():
            self._thread.terminate()
            self._thread.wait()

        # Clean up previous temp files
        for path in (self._tmp_input, self._tmp_output):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

        tmp_in = tempfile.NamedTemporaryFile(suffix="_kf_input.png", delete=False)
        tmp_in.close()
        self._tmp_input = tmp_in.name

        # Don't pre-create the output file — mflux won't overwrite an existing path.
        self._tmp_output = os.path.join(
            tempfile.gettempdir(), f"kf_output_{os.getpid()}.png"
        )

        self._edit_selection_bounds = None
        self._active_result_bounds = None
        if is_mask:
            # Honor an active selection: process only that region and remember
            # where to drop the cutout back on import. No selection → whole canvas.
            sel = doc.selection()
            bounds = None
            if sel and sel.width() > 0 and sel.height() > 0:
                bounds = _export_selection_crop(doc, self._tmp_input)
            if bounds:
                self._active_result_bounds = bounds
                _sx, _sy, _sw, _sh = bounds
                if _sw > 0:
                    self._preview.setRatio(_sh / _sw)
            else:
                self._export_canvas(doc, self._tmp_input)
                self._update_preview_ratio(doc)
            self._start_mask()
            return

        if tab == 1:
            sel = doc.selection()
            if sel and sel.width() > 0 and sel.height() > 0:
                bounds = _export_selection_crop(doc, self._tmp_input)
                if bounds:
                    self._edit_selection_bounds = bounds
                    _sx, _sy, _sw, _sh = bounds
                    if _sw > 0:
                        self._preview.setRatio(_sh / _sw)
                else:
                    self._export_canvas(doc, self._tmp_input)
            else:
                self._export_canvas(doc, self._tmp_input)
                self._update_preview_ratio(doc)
        else:
            self._export_canvas(doc, self._tmp_input)

        if tab == 0:
            cmd = self._build_generate_cmd(prompt, doc)
        elif tab == 1:
            cmd = self._build_edit_cmd(prompt, doc)
        else:
            cmd = self._build_angle_cmd(prompt, doc)

        self._on_log_message("Running: " + " ".join(f'"{t}"' if " " in t else t for t in cmd))
        self._gen_start_time = time.monotonic()
        self._set_busy(True)
        self._thread = GenerateThread(cmd, self._tmp_output)
        self._thread.finished.connect(self._on_finished)
        self._thread.errored.connect(self._on_error)
        self._thread.logged.connect(self._on_log_message)
        self._thread.progress.connect(self._on_progress)
        self._thread.start()

    def _start_mask(self) -> None:
        """Run rembg on the exported canvas, landing a transparent PNG."""
        rembg = find_executable(REMBG_CLI)  # guaranteed by _ensure_dependency
        model = self._mask_model.currentText()
        cmd = [rembg, "i", "-m", model]
        if self._mask_alpha_matting.isChecked():
            cmd.append("-a")
        cmd += [self._tmp_input, self._tmp_output]

        self._on_log_message(
            "Running: " + " ".join(f'"{t}"' if " " in t else t for t in cmd)
        )
        self._gen_start_time = time.monotonic()
        self._set_busy(True)
        self._thread = GenerateThread(cmd, self._tmp_output)
        self._thread.finished.connect(self._on_finished)
        self._thread.errored.connect(self._on_error)
        self._thread.logged.connect(self._on_log_message)
        self._thread.progress.connect(self._on_progress)
        self._thread.start()

    def _build_generate_cmd(self, prompt: str, doc: object) -> list[str]:
        model_name = self._gen_model.currentText()
        cli_name, model_flag, supports_strength, supports_guidance, *_ = MODEL_CLI.get(
            model_name, ("mflux-generate-flux2", model_name, True, True, False)
        )
        scale = self._gen_scale.value() / 100
        target_w = max(1, int(doc.width() * scale))
        target_h = max(1, int(doc.height() * scale))
        cli_path = os.path.join(MFLUX_DIR, cli_name)

        cmd = [cli_path, "--prompt", prompt]
        if model_flag:
            cmd += ["--model", model_flag]
        cmd += ["--image-path", self._tmp_input]
        cmd += ["--steps", str(self._gen_steps.value()), "--output", self._tmp_output]
        if abs(scale - 1.0) >= 0.001:
            cmd += ["--width", str(target_w), "--height", str(target_h)]
        if supports_guidance:
            cmd += ["--guidance", str(self._gen_guidance.value())]
        if supports_strength:
            cmd += ["--image-strength", str(self._gen_strength.value() / 100)]
        if (q := self._quantize_value(self._gen_quantize)) is not None:
            cmd += ["--quantize", str(q)]
        if not self._gen_random_seed.isChecked():
            cmd += ["--seed", str(self._gen_seed.value())]
        cmd += self._get_lora_args(self._gen_lora_entries)
        return cmd

    def _build_edit_cmd(self, prompt: str, doc: object) -> list[str]:
        model_name = self._edit_model.currentText()
        is_base = "base" in model_name
        scale = self._edit_scale.value() / 100
        target_w = max(1, int(doc.width() * scale))
        target_h = max(1, int(doc.height() * scale))

        # flux2-edit supports --width/--height but dimensions default to the first image
        # when set to "auto". Pre-scale the canvas so the output matches the document size.
        if abs(scale - 1.0) >= 0.001:
            img = QImage(self._tmp_input)
            if not img.isNull():
                img.scaled(target_w, target_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation).save(self._tmp_input, "PNG")

        cli_path = os.path.join(MFLUX_DIR, "mflux-generate-flux2-edit")
        cmd = [cli_path, "--prompt", prompt, "--model", model_name]
        cmd += ["--image-paths", self._tmp_input]
        for enabled_cb, thumb, _ in self._edit_ref_entries:
            if not enabled_cb.isChecked():
                continue
            ref_path = (thumb.imagePath() or "").strip()
            if ref_path:
                cmd.append(ref_path)

        cmd += ["--steps", str(self._edit_steps.value()), "--output", self._tmp_output]
        if is_base:
            cmd += ["--guidance", str(self._edit_guidance.value())]
        if (q := self._quantize_value(self._edit_quantize)) is not None:
            cmd += ["--quantize", str(q)]
        if not self._edit_random_seed.isChecked():
            cmd += ["--seed", str(self._edit_seed.value())]
        cmd += self._get_lora_args(self._edit_lora_entries)
        return cmd

    def _build_angle_cmd(self, prompt: str, doc: object) -> list[str]:
        model_name = self._angle_model.currentText()
        is_base = "base" in model_name
        scale = self._angle_scale.value() / 100
        target_w = max(1, int(doc.width() * scale))
        target_h = max(1, int(doc.height() * scale))

        # flux2-edit uses --image-paths and doesn't support --width/--height,
        # so pre-scale the input image when needed.
        if abs(scale - 1.0) >= 0.001:
            img = QImage(self._tmp_input)
            if not img.isNull():
                img.scaled(target_w, target_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation).save(self._tmp_input, "PNG")

        cli_path = os.path.join(MFLUX_DIR, "mflux-generate-flux2-edit")
        cmd = [
            cli_path,
            "--prompt", prompt,
            "--model", model_name,
            "--image-paths", self._tmp_input,
            "--steps", str(self._angle_steps.value()),
            "--guidance", str(self._angle_guidance.value()) if is_base else "1.0",
            "--output", self._tmp_output,
        ]
        if (q := self._quantize_value(self._angle_quantize)) is not None:
            cmd += ["--quantize", str(q)]
        if not self._angle_random_seed.isChecked():
            cmd += ["--seed", str(self._angle_seed.value())]
        return cmd

    def _build_edit_prompt(self) -> str:
        return self._edit_prompt.toPlainText().strip()

    def _on_finished(self, output_path: str) -> None:
        self._set_busy(False)
        exists = os.path.exists(output_path)
        size = os.path.getsize(output_path) if exists else 0
        self._on_log_message(
            f"Output path: {output_path}\n"
            f"File exists: {exists}, size: {size} bytes"
        )
        self._show_preview(output_path)
        # Remember where this result should land on import (a selection-scoped
        # cutout goes back at its bounds; everything else is full-canvas).
        if self._current_doc:
            uid = self._current_doc.fileName() or str(id(self._current_doc))
            self._result_bounds[uid] = self._active_result_bounds
        elapsed = time.monotonic() - self._gen_start_time
        minutes, seconds = divmod(int(elapsed), 60)
        self._time_label.setText(f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s")
        self._time_label.setVisible(True)
        self._progress.setVisible(False)

        if self._edit_selection_bounds:
            self._edit_selection_bounds = None

    def _on_error(self, message: str) -> None:
        self._set_busy(False)
        self._set_status("Error — see Logs.")
        # Auto-expand the log panel on error so the user notices it.
        self._log_btn.setChecked(True)

    def _on_log_message(self, text: str) -> None:
        if text:
            self._log.appendPlainText(text)
            self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
            if any(kw in text for kw in ("Downloading", "Fetching", "fetching")):
                self._set_status("Downloading…")

    def _on_progress(self, value: int) -> None:
        self._progress.setValue(value)
        if value > 0:
            self._set_status(f"Generating… {value}%")

    def _on_log_toggled(self, checked: bool) -> None:
        self._log.setVisible(checked)
        self._log_btns.setVisible(checked)

    def _copy_log(self) -> None:
        from PyQt5.QtWidgets import QApplication
        QApplication.clipboard().setText(self._log.toPlainText())

    def _clear_log(self) -> None:
        self._log.clear()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _show_preview(self, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._set_status("Could not load result image.")
            return
        self._preview.setPixmap(pixmap)
        self._preview.setVisible(True)
        self._use_btn.setEnabled(True)
        self._clear_preview_btn.setEnabled(True)
        if self._current_doc:
            self._doc_previews[self._current_doc.fileName() or str(id(self._current_doc))] = pixmap

    def _clear_preview(self) -> None:
        reply = QMessageBox.question(
            self, "Clear Preview",
            "Are you sure you want to clear the preview?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._preview.clearPixmap()
        self._time_label.setVisible(False)
        self._use_btn.setEnabled(False)
        self._clear_preview_btn.setEnabled(False)
        if self._current_doc:
            uid = self._current_doc.fileName() or str(id(self._current_doc))
            self._doc_previews.pop(uid, None)
            self._result_bounds.pop(uid, None)

    def _import_to_layer(self) -> None:
        doc = self._current_doc
        if not doc:
            return
        pixmap = self._doc_previews.get(doc.fileName() or str(id(doc)))
        if not pixmap:
            return

        uid = doc.fileName() or str(id(doc))

        # Determine scale from the active tab.
        tab = self._tabs.currentIndex()
        if tab == 0:
            scale = self._gen_scale.value() / 100
        elif tab == 1:
            scale = self._edit_scale.value() / 100
        elif tab == 2:
            scale = self._angle_scale.value() / 100
        else:
            scale = 1.0  # Mask output matches the canvas; nothing to upscale.

        can_upscale = (
            scale < 1.0
            and self._tmp_output
            and os.path.exists(self._tmp_output)
            and not (self._thread and self._thread.isRunning())
        )

        # Restore previous upscale settings for this document.
        saved = self._upscale_settings.get(uid, {})

        dlg = QDialog(self)
        dlg.setWindowTitle("Import as Layer")
        layout = QVBoxLayout(dlg)

        # --- Checkable upscale group ---
        upscale_group = QGroupBox("Upscale")
        upscale_group.setCheckable(True)
        upscale_group.setChecked(can_upscale)
        upscale_group.setEnabled(can_upscale)
        upscale_group.setToolTip("If needed, the image will be upscaled to fit the canvas size.")
        upscale_form = QFormLayout(upscale_group)

        softness_row = QWidget()
        softness_layout = QHBoxLayout(softness_row)
        softness_layout.setContentsMargins(0, 0, 0, 0)
        softness_layout.setSpacing(6)
        softness = QSlider(Qt.Horizontal)
        softness.setRange(0, 100)
        softness.setValue(saved.get("softness", 0))
        softness_spin = QDoubleSpinBox()
        softness_spin.setRange(0.0, 1.0)
        softness_spin.setSingleStep(0.01)
        softness_spin.setDecimals(2)
        softness_spin.setValue(softness.value() / 100)
        softness_spin.setFixedWidth(60)
        softness.valueChanged.connect(lambda v: softness_spin.setValue(v / 100))
        softness_spin.valueChanged.connect(lambda v: softness.setValue(int(v * 100)))
        softness_layout.addWidget(softness)
        softness_layout.addWidget(softness_spin)
        softness_row.setToolTip("0.0 = off, 1.0 = maximum softness.")
        upscale_form.addRow("Softness", softness_row)

        # Use active tab's quantize as default.
        default_q = self._quantize_value(self._gen_quantize) if tab == 0 else (
            self._quantize_value(self._edit_quantize) if tab == 1 else self._quantize_value(self._angle_quantize)
        )
        quantize = self._make_quantize_combo(default=default_q or 4)
        saved_q = saved.get("quantize")
        self._set_quantize_combo(quantize, saved_q if saved_q is not None else default_q)
        upscale_form.addRow("Quantize", quantize)

        seed_row = QWidget()
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        dlg_seed = QSpinBox()
        dlg_seed.setRange(0, 2_000_000_000)
        dlg_seed.setValue(saved.get("seed", 0))
        dlg_random_seed = QCheckBox("Random")
        dlg_random_seed.setChecked(saved.get("random_seed", True))
        dlg_random_seed.toggled.connect(dlg_seed.setDisabled)
        dlg_seed.setDisabled(dlg_random_seed.isChecked())
        seed_layout.addWidget(dlg_seed)
        seed_layout.addWidget(dlg_random_seed)
        upscale_form.addRow("Seed", seed_row)

        layout.addWidget(upscale_group)

        # --- Progress bar (hidden until upscale starts) ---
        dlg_progress = QProgressBar()
        dlg_progress.setRange(0, 100)
        dlg_progress.setValue(0)
        dlg_progress.setTextVisible(True)
        dlg_progress.setFormat("")
        dlg_progress.setVisible(False)
        layout.addWidget(dlg_progress)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Import")
        layout.addWidget(buttons)

        def save_upscale_settings():
            self._upscale_settings[uid] = {
                "softness": softness.value(),
                "quantize": self._quantize_value(quantize),
                "seed": dlg_seed.value(),
                "random_seed": dlg_random_seed.isChecked(),
            }

        def on_import():
            save_upscale_settings()

            if not upscale_group.isChecked():
                self._commit_to_layer(pixmap)
                dlg.accept()
                return

            # Upscaling shells out to mflux — make sure it's installed first.
            if not self._ensure_dependency(DEP_MFLUX):
                return

            # Disable controls and show progress.
            upscale_group.setEnabled(False)
            buttons.button(QDialogButtonBox.Ok).setEnabled(False)
            dlg_progress.setVisible(True)
            dlg_progress.setValue(0)
            dlg_progress.setFormat("Initializing...")

            upscale_factor = 1.0 / scale
            upscaled_path = os.path.join(
                tempfile.gettempdir(), f"kf_upscaled_{os.getpid()}.png"
            )
            # Remove stale file — mflux won't overwrite an existing path.
            if os.path.exists(upscaled_path):
                os.remove(upscaled_path)

            cli_path = os.path.join(MFLUX_DIR, "mflux-upscale-seedvr2")
            cmd = [
                cli_path,
                "--image-path", self._tmp_output,
                "--resolution", f"{upscale_factor}x",
                "--output", upscaled_path,
            ]
            if softness.value() > 0:
                cmd += ["--softness", str(softness.value() / 100)]

            if (q := self._quantize_value(quantize)) is not None:
                cmd += ["--quantize", str(q)]
            if not dlg_random_seed.isChecked():
                cmd += ["--seed", str(dlg_seed.value())]

            self._on_log_message("Running: " + " ".join(f'"{t}"' if " " in t else t for t in cmd))

            thread = GenerateThread(cmd, upscaled_path)
            # Keep a reference so it isn't garbage-collected.
            dlg._thread = thread

            def on_upscale_progress(value):
                dlg_progress.setValue(value)
                if value > 0:
                    dlg_progress.setFormat(f"Generating… {value}%")

            def on_upscale_log(text):
                self._on_log_message(text)
                if any(kw in text for kw in ("Downloading", "Fetching", "fetching")):
                    dlg_progress.setFormat("Downloading…")

            thread.progress.connect(on_upscale_progress)
            thread.logged.connect(on_upscale_log)

            def on_finished(path):
                pix = QPixmap(path)
                if not pix.isNull():
                    self._commit_to_layer(pix)
                dlg.accept()

            def on_error(msg):
                self._on_error(msg)
                dlg.reject()

            thread.finished.connect(on_finished)
            thread.errored.connect(on_error)
            thread.start()

        buttons.accepted.connect(on_import)
        buttons.rejected.connect(lambda: (
            dlg._thread.terminate() if hasattr(dlg, '_thread') and dlg._thread.isRunning() else None,
            dlg.reject()
        ))

        dlg.exec_()

    def _commit_to_layer(self, pixmap: QPixmap) -> None:
        doc = self._current_doc
        if not doc:
            return

        img = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        if img.isNull():
            return

        uid = doc.fileName() or str(id(doc))
        bounds = self._result_bounds.get(uid)

        layer = doc.createNode("Kritai Result", "paintlayer")
        doc.rootNode().addChildNode(layer, None)

        if bounds:
            # Selection-scoped cutout: drop it back at its original position,
            # native size, on an otherwise-transparent full-canvas layer.
            x, y, w, h = bounds
            if img.width() != w or img.height() != h:
                img = img.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            layer.setPixelData(bytes(ptr), x, y, w, h)
        else:
            if img.width() != doc.width() or img.height() != doc.height():
                img = img.scaled(doc.width(), doc.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            layer.setPixelData(bytes(ptr), 0, 0, img.width(), img.height())

        doc.refreshProjection()

        # Stop auto mode now that the result is committed to the document.
        self._auto_btn.setChecked(False)

    # ------------------------------------------------------------------
    # LoRA / Reference helpers (parameterized)
    # ------------------------------------------------------------------

    def _add_lora_row(self, entries_list: list, lora_layout: QVBoxLayout,
                      path: str = "", scale: float = 1.0, enabled: bool = True) -> None:
        """Add a LoRA entry row with enable checkbox, path, scale, and remove button."""
        from PyQt5.QtWidgets import QFileDialog

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        enabled_cb = QCheckBox()
        enabled_cb.setChecked(enabled)
        enabled_cb.setToolTip("Enable/disable this LoRA")

        path_edit = QLineEdit()
        path_edit.setPlaceholderText("LoRA path or HuggingFace repo...")
        path_edit.setText(path)
        path_edit.setToolTip(
            "Local .safetensors file, HuggingFace repo (org/model),\n"
            "or collection format (repo:filename.safetensors)."
        )

        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        def browse(checked: bool = False, pe: QLineEdit = path_edit) -> None:
            p, _ = QFileDialog.getOpenFileName(
                self, "Select LoRA file", "",
                "LoRA files (*.safetensors *.bin);;All files (*)"
            )
            if p:
                pe.setText(p)
        browse_btn.clicked.connect(browse)

        scale_spin = QDoubleSpinBox()
        scale_spin.setRange(0.0, 2.0)
        scale_spin.setSingleStep(0.1)
        scale_spin.setValue(scale)
        scale_spin.setFixedWidth(60)
        scale_spin.setToolTip("Scale factor for this LoRA (1.0 = full strength)")

        remove_btn = QPushButton("×")
        remove_btn.setFixedWidth(22)
        remove_btn.clicked.connect(
            lambda checked=False, r=row: self._remove_lora_row(entries_list, lora_layout, r)
        )

        row_layout.addWidget(enabled_cb)
        row_layout.addWidget(path_edit, 1)
        row_layout.addWidget(browse_btn)
        row_layout.addWidget(scale_spin)
        row_layout.addWidget(remove_btn)

        lora_layout.addWidget(row)
        entries_list.append((enabled_cb, path_edit, scale_spin, row))

    def _remove_lora_row(self, entries_list: list, lora_layout: QVBoxLayout, row: QWidget) -> None:
        entries_list[:] = [e for e in entries_list if e[-1] is not row]
        lora_layout.removeWidget(row)
        row.deleteLater()

    @staticmethod
    def _get_lora_args(entries_list: list) -> list[str]:
        """Return the --lora-paths and --lora-scales command fragments."""
        paths = []
        scales = []
        for enabled_cb, path_edit, scale_spin, _ in entries_list:
            if not enabled_cb.isChecked():
                continue
            p = path_edit.text().strip()
            if p:
                paths.append(p)
                scales.append(str(scale_spin.value()))
        if not paths:
            return []
        return ["--lora-paths"] + paths + ["--lora-scales"] + scales

    _QUANTIZE_CHOICES = [None, 3, 4, 5, 6, 8]

    def _make_quantize_combo(self, default: int = 4) -> QComboBox:
        combo = QComboBox()
        combo.setToolTip(
            "Reduces model weight precision to save memory and speed up generation.\n"
            "4 is a good balance of quality and speed. None = full precision."
        )
        for v in self._QUANTIZE_CHOICES:
            combo.addItem("None" if v is None else str(v))
        self._set_quantize_combo(combo, default)
        return combo

    @staticmethod
    def _quantize_value(combo: QComboBox) -> int | None:
        text = combo.currentText()
        return None if text == "None" else int(text)

    @staticmethod
    def _set_quantize_combo(combo: QComboBox, value: int | None) -> None:
        target = "None" if value is None else str(value)
        idx = combo.findText(target)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _add_ref_row(self, entries_list: list, ref_layout: QVBoxLayout,
                     path: str = "", enabled: bool = True) -> None:
        """Add a reference image entry with enable checkbox, thumbnail, and remove button."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        enabled_cb = QCheckBox()
        enabled_cb.setChecked(enabled)
        enabled_cb.setToolTip("Enable/disable this reference image")

        thumb = DropThumbnail()
        if path:
            thumb.setImagePath(path)

        remove_btn = QPushButton("×")
        remove_btn.setFixedWidth(22)
        remove_btn.clicked.connect(
            lambda checked=False, r=row: self._remove_ref_row(entries_list, ref_layout, r)
        )

        row_layout.addWidget(enabled_cb)
        row_layout.addWidget(thumb, 1)
        row_layout.addWidget(remove_btn)

        enabled_cb.toggled.connect(self._update_generate_btn)
        thumb.pathChanged.connect(self._update_generate_btn)

        ref_layout.addWidget(row)
        entries_list.append((enabled_cb, thumb, row))
        self._update_generate_btn()

    def _remove_ref_row(self, entries_list: list, ref_layout: QVBoxLayout, row: QWidget) -> None:
        entries_list[:] = [e for e in entries_list if e[-1] is not row]
        ref_layout.removeWidget(row)
        row.deleteLater()
        self._update_generate_btn()

    # ------------------------------------------------------------------

    def _update_preview_ratio(self, doc: object) -> None:
        if doc and doc.width() > 0:
            self._preview.setRatio(doc.height() / doc.width())

    def canvasChanged(self, canvas: object) -> None:
        # Disable auto mode when switching documents.
        self._auto_btn.setChecked(False)

        from krita import Krita
        self._current_doc = Krita.instance().activeDocument() if canvas is not None else None
        if self._current_doc:
            self._update_preview_ratio(self._current_doc)
            self._load_settings(self._current_doc)
            cached = self._doc_previews.get(self._current_doc.fileName() or str(id(self._current_doc)))
            if cached:
                self._preview.setPixmap(cached)
                self._preview.setVisible(True)
                self._use_btn.setEnabled(True)
                self._clear_preview_btn.setEnabled(True)
            else:
                self._preview.clearPixmap()
                self._time_label.setVisible(False)
                self._use_btn.setEnabled(False)
                self._clear_preview_btn.setEnabled(False)
        else:
            self._preview.clearPixmap()
            self._time_label.setVisible(False)
            self._use_btn.setEnabled(False)
            self._clear_preview_btn.setEnabled(False)
        self._update_generate_btn()


def _export_selection_crop(doc: object, path: str) -> Optional[tuple[int, int, int, int]]:
    """Export the canvas region covered by the current selection to *path*.

    Returns ``(x, y, w, h)`` of the selection bounds, or ``None`` if there is
    no selection.
    """
    sel = doc.selection()
    if sel is None:
        return None
    sx, sy, sw, sh = sel.x(), sel.y(), sel.width(), sel.height()
    if sw <= 0 or sh <= 0:
        return None

    tmp_full = tempfile.NamedTemporaryFile(suffix="_kf_full.png", delete=False)
    tmp_full.close()
    try:
        doc.setBatchmode(True)
        doc.exportImage(tmp_full.name, InfoObject())
        doc.setBatchmode(False)
        full_img = QImage(tmp_full.name)
    finally:
        try:
            os.unlink(tmp_full.name)
        except OSError:
            pass
    if full_img.isNull():
        return None

    cropped = full_img.copy(sx, sy, sw, sh)
    cropped.save(path, "PNG")
    return (sx, sy, sw, sh)
