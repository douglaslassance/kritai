from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
from typing import Optional

from krita import DockWidget, InfoObject
from PyQt5.QtCore import (
    QByteArray,
    QEvent,
    QObject,
    QSize,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
)
from PyQt5.QtGui import QFontDatabase, QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QSpinBox,
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
    # FLUX.2 edit — canvas + optional reference image via --image-paths
    "flux2-edit":          ("mflux-generate-flux2-edit", None,                  False, False, True),
    # Kontext (image editing via instruction)
    "kontext-dev":         ("mflux-generate-kontext",     "dev",                True,  True,  False),
    "kontext-schnell":     ("mflux-generate-kontext",     "schnell",            True,  False, False),
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

    def setRatio(self, ratio: float) -> None:
        if ratio > 0 and ratio != self._ratio:
            self._ratio = ratio
            self._update_fixed_height()

    def resizeEvent(self, event: QEvent) -> None:
        super().resizeEvent(event)
        self._update_fixed_height()

    def _update_fixed_height(self) -> None:
        self.setFixedHeight(max(1, self.heightForWidth(self.width())))

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
                self.setPixmap(pix.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                self.setStyleSheet("border: 1px solid #666;")
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
            self.setStyleSheet("border: 1px solid #666;")

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

        # Flush settings to annotation only when Krita saves the file.
        Krita.instance().notifier().imageSaved.connect(self._on_image_saved)

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
        root = QWidget()
        self.setWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # --- Model selector ---
        self._model = QComboBox()
        # Groups mirror the model families in the mflux README.
        # SeedVR2 (upscaling) and Depth Pro (depth estimation) are omitted
        # as they are not image-to-image generation models.
        model_tooltips = {
            "flux2-klein-4b":      "Fast distilled 4B model. Good default for quick iterations. No guidance.",
            "flux2-klein-9b":      "Distilled 9B model. Higher quality than 4B but slower. No guidance.",
            "flux2-klein-base-4b": "Non-distilled 4B base model. Supports guidance. Needs more steps.",
            "flux2-klein-base-9b": "Non-distilled 9B base model. Best quality in the FLUX.2 family.",
            "flux2-edit":          "Edit the canvas with a prompt and an optional reference image.",
            "kontext-dev":         "High-quality instruction-based image editing.",
            "kontext-schnell":     "Faster distilled variant. No guidance.",
        }
        model_groups = {
            "FLUX.2 (recommended)": [
                "flux2-klein-4b",
                "flux2-klein-9b",
                "flux2-klein-base-4b",
                "flux2-klein-base-9b",
                "flux2-edit",
            ],
            "Kontext": [
                "kontext-dev",
                "kontext-schnell",
            ],
        }
        for group, models in model_groups.items():
            self._model.insertSeparator(self._model.count())
            # Qt doesn't have native group headers; use a disabled item as label.
            self._model.addItem(group)
            self._model.model().item(self._model.count() - 1).setEnabled(False)
            for m in models:
                self._model.addItem(m)
                idx = self._model.count() - 1
                self._model.setItemData(idx, model_tooltips.get(m, ""), Qt.ToolTipRole)
        self._model.setCurrentIndex(self._model.findText("flux2-klein-4b"))
        self._model.setToolTip(
            "Which mflux model family to use for generation.\n"
            "FLUX.2 is the fastest and highest quality.\n"
            "Kontext offers instruction-based editing."
        )
        self._model.currentIndexChanged.connect(self._update_model_ui)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model"))
        model_row.addWidget(self._model, 1)
        outer.addLayout(model_row)

        # --- Prompt ---
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Prompt...")
        self._prompt.setToolTip("Describe what you want the image to look like.")
        self._prompt.setFixedHeight(60)
        outer.addWidget(self._prompt)

        # --- Negative prompt ---
        self._negative_prompt = QLineEdit()
        self._negative_prompt.setPlaceholderText("Optional negative prompt…")
        self._negative_prompt.setToolTip(
            "Describe what you want to avoid in the result.\n"
            "Example: blurry, low quality, extra limbs."
        )
        self._negative_prompt.setVisible(False)
        outer.addWidget(self._negative_prompt)

        # --- Settings ---
        settings_widget = QWidget()
        form = QFormLayout(settings_widget)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        outer.addWidget(settings_widget)

        self._quantize = QSpinBox()
        self._quantize.setSpecialValueText("None")
        self._quantize.setRange(0, 8)
        self._quantize.setValue(4)
        self._quantize.setToolTip(
            "Reduces model weight precision to save memory and speed up generation.\n"
            "4 is a good balance of quality and speed.\n"
            "Higher = better quality, more RAM. 0 = no quantization (full precision)."
        )
        form.addRow("Quantize", self._quantize)

        self._steps = QSpinBox()
        self._steps.setRange(1, 100)
        self._steps.setValue(20)
        self._steps.setToolTip(
            "Number of denoising steps. More steps = higher quality but slower.\n"
            "Distilled models (schnell, klein, turbo) work well with 4–8 steps.\n"
            "Base models (dev, base) typically need 20–50 steps."
        )
        form.addRow("Steps", self._steps)

        self._guidance = QDoubleSpinBox()
        self._guidance.setRange(0.0, 30.0)
        self._guidance.setSingleStep(0.5)
        self._guidance.setValue(3.5)
        self._guidance.setToolTip(
            "How closely the result follows your prompt.\n"
            "Higher = more literal, lower = more creative.\n"
            "Not supported by distilled models (they ignore this setting)."
        )
        form.addRow("Guidance", self._guidance)
        self._guidance_label = form.labelForField(self._guidance)

        strength_row = QWidget()
        strength_layout = QHBoxLayout(strength_row)
        strength_layout.setContentsMargins(0, 0, 0, 0)
        strength_layout.setSpacing(6)
        self._strength = QSlider(Qt.Horizontal)
        self._strength.setRange(0, 100)
        self._strength.setValue(75)
        self._strength_label = QLabel("0.60")
        self._strength_label.setFixedWidth(30)
        self._strength.valueChanged.connect(
            lambda v: self._strength_label.setText(f"{v / 100:.2f}")
        )
        strength_layout.addWidget(self._strength)
        strength_layout.addWidget(self._strength_label)
        strength_row.setToolTip(
            "How much the canvas influences the result.\n"
            "0.0 = output ignores your painting entirely.\n"
            "1.0 = output stays very close to the canvas.\n"
            "0.5–0.7 is a good starting range."
        )
        form.addRow("Strength", strength_row)
        self._strength_row = strength_row
        self._strength_label_form = form.labelForField(strength_row)

        scale_row = QWidget()
        scale_layout = QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setSpacing(6)
        self._scale = QSlider(Qt.Horizontal)
        self._scale.setRange(0, 100)
        self._scale.setValue(50)
        self._scale_label = QLabel("0.50")
        self._scale_label.setFixedWidth(30)
        self._scale.valueChanged.connect(
            lambda v: self._scale_label.setText(f"{v / 100:.2f}")
        )
        scale_layout.addWidget(self._scale)
        scale_layout.addWidget(self._scale_label)
        scale_row.setToolTip(
            "Scale of the canvas sent to mflux relative to its original size.\n"
            "0.5 = half resolution (faster, less VRAM).\n"
            "1.0 = full resolution."
        )
        form.addRow("Scale", scale_row)

        seed_row = QWidget()
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        self._seed = QSpinBox()
        self._seed.setRange(0, 2_000_000_000)
        self._seed.setValue(0)
        self._random_seed = QCheckBox("Random")
        self._random_seed.setChecked(False)
        self._random_seed.toggled.connect(self._seed.setDisabled)
        self._seed.setDisabled(False)
        seed_layout.addWidget(self._seed)
        seed_layout.addWidget(self._random_seed)
        seed_row.setToolTip(
            "Fixed seed produces the same result every time given the same inputs.\n"
            "Random seed gives a different result on each generation."
        )
        form.addRow("Seed", seed_row)

        # --- Reference images (edit models only) ---
        self._ref_section = CollapsibleSection("Reference Images")
        ref_content = QVBoxLayout()
        ref_content.setSpacing(4)

        self._ref_list = QVBoxLayout()
        self._ref_list.setSpacing(4)
        ref_content.addLayout(self._ref_list)
        self._ref_entries = []  # list of (thumbnail, prompt_edit, row_widget)

        add_ref_btn = QPushButton("+ Add Reference")
        add_ref_btn.clicked.connect(lambda: self._add_ref_row())
        ref_content.addWidget(add_ref_btn)

        self._ref_section.setContentLayout(ref_content)
        self._ref_section.setVisible(False)
        outer.addWidget(self._ref_section)

        # --- LoRA list ---
        lora_section = CollapsibleSection("LoRAs")
        lora_content = QVBoxLayout()
        lora_content.setSpacing(4)

        self._lora_list = QVBoxLayout()
        self._lora_list.setSpacing(4)
        lora_content.addLayout(self._lora_list)
        self._lora_entries = []  # list of (path_widget, scale_widget, row_widget)

        add_lora_btn = QPushButton("+ Add LoRA")
        add_lora_btn.clicked.connect(lambda: self._add_lora_row())
        lora_content.addWidget(add_lora_btn)

        lora_section.setContentLayout(lora_content)
        outer.addWidget(lora_section)

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
        btn_row.addWidget(self._generate_btn)
        btn_row.addWidget(self._auto_btn)

        self._fill_btn = QToolButton()
        self._fill_btn.setToolTip("Modify Selected Region")
        self._fill_btn.setIcon(Krita.instance().icon("tool_rect_selection"))
        self._fill_btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self._fill_btn.setEnabled(False)
        self._fill_btn.clicked.connect(self._fill_selection)
        self._fill_opacity = QGraphicsOpacityEffect(self._fill_btn)
        self._fill_opacity.setOpacity(0.3)
        self._fill_btn.setGraphicsEffect(self._fill_opacity)
        btn_row.addWidget(self._fill_btn)

        # Poll selection state to enable/disable fill button.
        self._sel_poll_timer = QTimer()
        self._sel_poll_timer.setInterval(500)
        self._sel_poll_timer.timeout.connect(self._poll_selection)
        self._sel_poll_timer.start()

        self._use_btn = QToolButton()
        self._use_btn.setToolTip("Add result as a new layer in the document")
        self._use_btn.setIcon(Krita.instance().icon("addlayer"))
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

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setVisible(False)
        self._clear_btn.clicked.connect(self._clear_log)

        outer.addWidget(self._log)
        outer.addWidget(self._clear_btn)

        # --- Preview image ---
        self._preview = PreviewLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._preview.setStyleSheet("border-radius: 4px;")
        self._preview.setVisible(False)
        outer.addWidget(self._preview)
        outer.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self._update_model_ui()

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    ANNOTATION_TYPE = "kritai_settings"

    def _connect_settings_signals(self) -> None:
        # All changes persist settings immediately.
        for signal in [
            self._prompt.textChanged,
            self._negative_prompt.textChanged,
            self._model.currentIndexChanged,
            self._quantize.valueChanged,
            self._steps.valueChanged,
            self._guidance.valueChanged,
            self._strength.valueChanged,
            self._scale.valueChanged,
            self._seed.valueChanged,
            self._random_seed.toggled,
        ]:
            signal.connect(self._save_settings)

        # Auto-refresh triggers only on focus-out for text fields,
        # immediately for all other controls.
        self._negative_prompt.editingFinished.connect(self._on_setting_changed)

        prompt_filter = _FocusOutSignal(self._prompt)
        prompt_filter.focusLost.connect(self._on_setting_changed)

        for signal in [
            self._model.currentIndexChanged,
            self._quantize.valueChanged,
            self._steps.valueChanged,
            self._guidance.valueChanged,
            self._strength.valueChanged,
            self._scale.valueChanged,
            self._seed.valueChanged,
            self._random_seed.toggled,
        ]:
            signal.connect(self._on_setting_changed)

    def _on_setting_changed(self) -> None:
        if self._auto_btn.isChecked():
            self._debounce_timer.start()

    def _save_settings(self) -> None:
        doc = self._current_doc
        if not doc:
            return
        uid = doc.fileName() or str(id(doc))
        new = {
            "prompt": self._prompt.toPlainText(),
            "negative_prompt": self._negative_prompt.text(),
            "reference_images": [
                {"path": t.imagePath() or "", "prompt": p.text().strip(), "enabled": e.isChecked()}
                for e, t, p, _ in self._ref_entries
                if (t.imagePath() or "").strip() or p.text().strip()
            ],
            "model": self._model.currentText(),
            "quantize": self._quantize.value(),
            "steps": self._steps.value(),
            "guidance": self._guidance.value(),
            "strength": self._strength.value() / 100,
            "scale": self._scale.value() / 100,
            "seed": self._seed.value(),
            "random_seed": self._random_seed.isChecked(),
            "loras": [
                {"path": p.text().strip(), "scale": s.value(), "enabled": e.isChecked()}
                for e, p, s, _ in self._lora_entries if p.text().strip()
            ],
            "upscale": self._upscale_settings.get(uid, {}),
        }
        prev = self._doc_settings.get(uid)
        self._doc_settings[uid] = new
        if prev != new:
            doc.setModified(True)

    def _on_image_saved(self, filename: str) -> None:
        """Flush in-memory settings to the document annotation on save."""
        for doc in Krita.instance().documents():
            if doc.fileName() == filename:
                data = self._doc_settings.get(doc.fileName() or str(id(doc)))
                if data:
                    raw = json.dumps(data).encode("utf-8")
                    doc.setAnnotation(self.ANNOTATION_TYPE, "Kritai settings", QByteArray(raw))
                    # setAnnotation marks the doc modified; clear it since we
                    # just saved.
                    doc.setModified(False)
                break

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

        # Block signals while restoring to avoid triggering _save_settings
        # for each individual widget change.
        widgets = [
            self._prompt, self._negative_prompt, self._model,
            self._quantize, self._steps, self._guidance,
            self._strength, self._scale, self._seed, self._random_seed,
        ]
        for w in widgets:
            w.blockSignals(True)

        self._prompt.setPlainText(data.get("prompt", ""))
        self._negative_prompt.setText(data.get("negative_prompt", ""))
        # Restore reference images.
        for *_, row in list(self._ref_entries):
            self._ref_list.removeWidget(row)
            row.deleteLater()
        self._ref_entries.clear()
        for ref in data.get("reference_images", []):
            self._add_ref_row(ref.get("path", ""), ref.get("prompt", ""), ref.get("enabled", True))
        # Backward compat: migrate old single reference_image field.
        if not data.get("reference_images") and data.get("reference_image"):
            self._add_ref_row(data["reference_image"], "", True)
        model = data.get("model", "dev")
        idx = self._model.findText(model)
        if idx >= 0:
            self._model.setCurrentIndex(idx)
        self._quantize.setValue(data.get("quantize", 4))
        self._steps.setValue(data.get("steps", 20))
        self._guidance.setValue(data.get("guidance", 3.5))
        strength_val = int(data.get("strength", 0.75) * 100)
        self._strength.setValue(strength_val)
        self._strength_label.setText(f"{strength_val / 100:.2f}")
        scale_val = int(data.get("scale", data.get("downscale", data.get("resolution_scale", 0.5))) * 100)
        self._scale.setValue(scale_val)
        self._scale_label.setText(f"{scale_val / 100:.2f}")
        self._seed.setValue(data.get("seed", 0))
        self._random_seed.setChecked(data.get("random_seed", False))

        # Restore LoRAs.
        # Clear existing rows.
        for *_, row in list(self._lora_entries):
            self._lora_list.removeWidget(row)
            row.deleteLater()
        self._lora_entries.clear()
        for lora in data.get("loras", []):
            self._add_lora_row(lora.get("path", ""), lora.get("scale", 1.0), lora.get("enabled", True))

        # Restore upscale settings.
        upscale = data.get("upscale")
        if upscale:
            self._upscale_settings[uid] = upscale
        self._seed.setDisabled(self._random_seed.isChecked())

        for w in widgets:
            w.blockSignals(False)

        self._update_model_ui()

    # ------------------------------------------------------------------
    # Auto-mode
    # ------------------------------------------------------------------

    def _on_auto_toggled(self, checked: bool) -> None:
        if checked:
            self._last_canvas_hash = self._canvas_hash()
            self._poll_timer.start()
            self._generate()
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

    def _poll_selection(self) -> None:
        """Enable/disable fill button based on whether a selection is active."""
        has_sel = False
        if self._current_doc:
            sel = self._current_doc.selection()
            has_sel = sel is not None and sel.width() > 0 and sel.height() > 0
        self._fill_btn.setEnabled(has_sel)
        self._fill_opacity.setOpacity(1.0 if has_sel else 0.3)

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

    def _generate(self) -> None:
        from krita import Krita

        app = Krita.instance()
        doc = app.activeDocument()
        if not doc:
            self._set_status("No active document.")
            return

        raw_prompt = self._prompt.toPlainText().strip()
        if not raw_prompt:
            self._set_status("Enter a prompt first.")
            return
        prompt = self._build_prompt()

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

        self._export_canvas(doc, self._tmp_input)

        scale = self._scale.value() / 100
        target_w = max(1, int(doc.width() * scale))
        target_h = max(1, int(doc.height() * scale))

        model_name = self._model.currentText()
        cli_name, model_flag, supports_strength, supports_guidance, *rest = MODEL_CLI.get(
            model_name, ("mflux-generate", model_name, True, True, False)
        )
        needs_reference = rest[0] if rest else False

        # Edit models don't accept --width/--height; pre-scale the input instead.
        if needs_reference and abs(scale - 1.0) >= 0.001:
            img = QImage(self._tmp_input)
            if not img.isNull():
                img.scaled(target_w, target_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation).save(self._tmp_input, "PNG")
        cli_path = os.path.join(MFLUX_DIR, cli_name)

        cmd = [cli_path, "--prompt", prompt]

        if model_flag:
            cmd += ["--model", model_flag]

        if needs_reference:
            cmd += ["--image-paths", self._tmp_input]
            for enabled_cb, thumb, _, _ in self._ref_entries:
                if not enabled_cb.isChecked():
                    continue
                ref_path = (thumb.imagePath() or "").strip()
                if ref_path:
                    cmd.append(ref_path)
        else:
            cmd += ["--image-path", self._tmp_input]

        cmd += [
            "--steps", str(self._steps.value()),
            "--output", self._tmp_output,
        ]

        if abs(scale - 1.0) >= 0.001 and not needs_reference:
            cmd += ["--width", str(target_w), "--height", str(target_h)]

        if supports_guidance:
            cmd += ["--guidance", str(self._guidance.value())]

        if supports_strength:
            cmd += ["--image-strength", str(self._strength.value() / 100)]

        if self._quantize.value() > 0:
            cmd += ["--quantize", str(self._quantize.value())]

        neg = self._negative_prompt.text().strip()
        if neg:
            cmd += ["--negative-prompt", neg]

        if not self._random_seed.isChecked():
            cmd += ["--seed", str(self._seed.value())]

        cmd += self._get_lora_args()

        self._on_log_message("Running: " + " ".join(f'"{t}"' if " " in t else t for t in cmd))
        self._set_busy(True)
        self._thread = GenerateThread(cmd, self._tmp_output)
        self._thread.finished.connect(self._on_finished)
        self._thread.errored.connect(self._on_error)
        self._thread.logged.connect(self._on_log_message)
        self._thread.progress.connect(self._on_progress)
        self._thread.start()

    def _on_finished(self, output_path: str) -> None:
        self._set_busy(False)
        exists = os.path.exists(output_path)
        size = os.path.getsize(output_path) if exists else 0
        self._on_log_message(
            f"Output path: {output_path}\n"
            f"File exists: {exists}, size: {size} bytes"
        )
        self._show_preview(output_path)
        self._progress.setVisible(False)

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
        self._clear_btn.setVisible(checked)

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
        if self._current_doc:
            self._doc_previews[self._current_doc.fileName() or str(id(self._current_doc))] = pixmap

    @staticmethod
    def _load_fill_settings() -> dict:
        from PyQt5.QtCore import QSettings
        s = QSettings("kritai", "fill")
        return {
            "strength": int(s.value("strength", 90)),
            "steps": int(s.value("steps", 20)),
            "guidance": float(s.value("guidance", 3.5)),
            "quantize": int(s.value("quantize", 4)),
            "seed": int(s.value("seed", 0)),
            "random_seed": s.value("random_seed", "true") == "true",
            "prompt": str(s.value("prompt", "")),
        }

    @staticmethod
    def _save_fill_settings(data: dict) -> None:
        from PyQt5.QtCore import QSettings
        s = QSettings("kritai", "fill")
        for k, v in data.items():
            s.setValue(k, str(v).lower() if isinstance(v, bool) else v)

    def _fill_selection(self) -> None:
        from krita import Krita

        doc = Krita.instance().activeDocument()
        if not doc:
            return
        sel = doc.selection()
        if sel is None or sel.width() <= 0 or sel.height() <= 0:
            return

        sel_x, sel_y, sel_w, sel_h = sel.x(), sel.y(), sel.width(), sel.height()

        saved = self._load_fill_settings()

        dlg = QDialog(self)
        dlg.setWindowTitle("Modify Selected Region")
        layout = QVBoxLayout(dlg)

        # --- Prompt ---
        prompt_edit = QPlainTextEdit()
        prompt_edit.setPlaceholderText("Prompt...")
        prompt_edit.setPlainText(saved["prompt"])
        prompt_edit.setFixedHeight(60)
        layout.addWidget(prompt_edit)

        # --- Settings ---
        form = QFormLayout()

        # Strength
        strength_row = QWidget()
        strength_layout = QHBoxLayout(strength_row)
        strength_layout.setContentsMargins(0, 0, 0, 0)
        strength_layout.setSpacing(6)
        fill_strength = QSlider(Qt.Horizontal)
        fill_strength.setRange(0, 100)
        fill_strength.setValue(saved["strength"])
        fill_strength_label = QLabel(f"{fill_strength.value() / 100:.2f}")
        fill_strength_label.setFixedWidth(30)
        fill_strength.valueChanged.connect(
            lambda v: fill_strength_label.setText(f"{v / 100:.2f}")
        )
        strength_layout.addWidget(fill_strength)
        strength_layout.addWidget(fill_strength_label)
        strength_row.setToolTip(
            "How much the selection content influences the result.\n"
            "0.0 = ignore existing content entirely.\n"
            "1.0 = stay very close to what is already there.\n"
            "Default: 0.90"
        )
        form.addRow("Strength", strength_row)

        steps = QSpinBox()
        steps.setRange(1, 100)
        steps.setValue(saved["steps"])
        form.addRow("Steps", steps)

        guidance = QDoubleSpinBox()
        guidance.setRange(0.0, 100.0)
        guidance.setSingleStep(0.5)
        guidance.setValue(saved["guidance"])
        form.addRow("Guidance", guidance)

        quantize = QSpinBox()
        quantize.setSpecialValueText("None")
        quantize.setRange(0, 8)
        quantize.setValue(saved["quantize"])
        form.addRow("Quantize", quantize)

        seed_row = QWidget()
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        dlg_seed = QSpinBox()
        dlg_seed.setRange(0, 2_000_000_000)
        dlg_seed.setValue(saved["seed"])
        dlg_random_seed = QCheckBox("Random")
        dlg_random_seed.setChecked(saved["random_seed"])
        dlg_random_seed.toggled.connect(dlg_seed.setDisabled)
        dlg_seed.setDisabled(dlg_random_seed.isChecked())
        seed_layout.addWidget(dlg_seed)
        seed_layout.addWidget(dlg_random_seed)
        form.addRow("Seed", seed_row)

        layout.addLayout(form)

        # --- Progress ---
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setTextVisible(True)
        progress.setFormat("")
        progress.setVisible(False)
        layout.addWidget(progress)

        # --- Buttons ---
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        generate_btn = buttons.addButton("Generate", QDialogButtonBox.AcceptRole)
        layout.addWidget(buttons)
        buttons.rejected.connect(dlg.reject)

        dlg._thread = None

        def on_generate():
            prompt = prompt_edit.toPlainText().strip()
            if not prompt:
                return

            # Save settings at Krita level.
            self._save_fill_settings({
                "strength": fill_strength.value(),
                "steps": steps.value(),
                "guidance": guidance.value(),
                "quantize": quantize.value(),
                "seed": dlg_seed.value(),
                "random_seed": dlg_random_seed.isChecked(),
                "prompt": prompt,
            })

            generate_btn.setEnabled(False)
            progress.setVisible(True)
            progress.setValue(0)
            progress.setFormat("Initializing...")

            # Export cropped selection.
            tmp_in = tempfile.NamedTemporaryFile(suffix="_kf_sel_in.png", delete=False)
            tmp_in.close()
            tmp_in_path = tmp_in.name

            bounds = _export_selection_crop(doc, tmp_in_path)
            if bounds is None:
                generate_btn.setEnabled(True)
                progress.setVisible(False)
                return

            tmp_out = os.path.join(
                tempfile.gettempdir(), f"kf_sel_out_{os.getpid()}.png"
            )
            if os.path.exists(tmp_out):
                try:
                    os.remove(tmp_out)
                except OSError:
                    pass

            # Build command — always Kontext dev.
            cli_path = os.path.join(MFLUX_DIR, "mflux-generate-kontext")

            cmd = [
                cli_path,
                "--prompt", prompt,
                "--model", "dev",
                "--image-path", tmp_in_path,
                "--image-strength", str(fill_strength.value() / 100),
                "--steps", str(steps.value()),
                "--guidance", str(guidance.value()),
                "--output", tmp_out,
            ]

            if quantize.value() > 0:
                cmd += ["--quantize", str(quantize.value())]

            if not dlg_random_seed.isChecked():
                cmd += ["--seed", str(dlg_seed.value())]

            cmd += self._get_lora_args()

            self._on_log_message("Running: " + " ".join(
                f'"{t}"' if " " in t else t for t in cmd
            ))

            thread = GenerateThread(cmd, tmp_out)
            dlg._thread = thread

            def on_progress(value):
                progress.setValue(value)
                if value > 0:
                    progress.setFormat(f"Generating... {value}%")

            def on_log(text):
                self._on_log_message(text)
                if any(kw in text for kw in ("Downloading", "Fetching", "fetching")):
                    progress.setFormat("Downloading...")

            def on_finished(output_path):
                result_img = QImage(output_path)
                if result_img.isNull():
                    generate_btn.setEnabled(True)
                    progress.setVisible(False)
                    return

                result_img = result_img.convertToFormat(QImage.Format_ARGB32)
                if result_img.width() != sel_w or result_img.height() != sel_h:
                    result_img = result_img.scaled(
                        sel_w, sel_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                    )

                layer = doc.createNode("Kritai Fill", "paintlayer")
                doc.rootNode().addChildNode(layer, None)
                ptr = result_img.bits()
                ptr.setsize(result_img.byteCount())
                layer.setPixelData(bytes(ptr), sel_x, sel_y, sel_w, sel_h)
                doc.refreshProjection()

                for p in (tmp_in_path, tmp_out):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

                dlg.accept()

            def on_error(msg):
                self._on_error(msg)
                generate_btn.setEnabled(True)
                progress.setVisible(False)

            thread.progress.connect(on_progress)
            thread.logged.connect(on_log)
            thread.finished.connect(on_finished)
            thread.errored.connect(on_error)
            thread.start()

        generate_btn.clicked.connect(on_generate)
        dlg.exec_()

    def _import_to_layer(self) -> None:
        doc = self._current_doc
        if not doc:
            return
        pixmap = self._doc_previews.get(doc.fileName() or str(id(doc)))
        if not pixmap:
            return

        uid = doc.fileName() or str(id(doc))
        scale = self._scale.value() / 100
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
        softness_value_label = QLabel(f"{softness.value() / 100:.2f}")
        softness_value_label.setFixedWidth(30)
        softness.valueChanged.connect(lambda v: softness_value_label.setText(f"{v / 100:.2f}"))
        softness_layout.addWidget(softness)
        softness_layout.addWidget(softness_value_label)
        softness_row.setToolTip("0.0 = off, 1.0 = maximum softness.")
        upscale_form.addRow("Softness", softness_row)

        quantize = QSpinBox()
        quantize.setSpecialValueText("None")
        quantize.setRange(0, 8)
        quantize.setValue(saved.get("quantize", self._quantize.value()))
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
                "quantize": quantize.value(),
                "seed": dlg_seed.value(),
                "random_seed": dlg_random_seed.isChecked(),
            }

        def on_import():
            save_upscale_settings()

            if not upscale_group.isChecked():
                self._commit_to_layer(pixmap)
                dlg.accept()
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

            if quantize.value() > 0:
                cmd += ["--quantize", str(quantize.value())]
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
        buttons.rejected.connect(dlg.reject)

        dlg.exec_()

    def _commit_to_layer(self, pixmap: QPixmap) -> None:
        doc = self._current_doc
        if not doc:
            return

        img = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        if img.isNull():
            return
        if img.width() != doc.width() or img.height() != doc.height():
            img = img.scaled(doc.width(), doc.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        layer = doc.createNode("Kritai Result", "paintlayer")
        doc.rootNode().addChildNode(layer, None)
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        layer.setPixelData(bytes(ptr), 0, 0, img.width(), img.height())
        doc.refreshProjection()

        # Stop auto mode now that the result is committed to the document.
        self._auto_btn.setChecked(False)

    # ------------------------------------------------------------------

    def _update_model_ui(self) -> None:
        model_name = self._model.currentText()
        _, _, supports_strength, supports_guidance, *rest = MODEL_CLI.get(
            model_name, (None, None, True, True, False)
        )
        needs_ref = rest[0] if rest else False

        self._guidance.setVisible(supports_guidance)
        if self._guidance_label:
            self._guidance_label.setVisible(supports_guidance)

        self._strength_row.setVisible(supports_strength)
        if self._strength_label_form:
            self._strength_label_form.setVisible(supports_strength)

        self._ref_section.setVisible(needs_ref)

    def _add_lora_row(self, path: str = "", scale: float = 1.0, enabled: bool = True) -> None:
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
        remove_btn.clicked.connect(lambda checked=False, r=row: self._remove_lora_row(r))

        row_layout.addWidget(enabled_cb)
        row_layout.addWidget(path_edit, 1)
        row_layout.addWidget(browse_btn)
        row_layout.addWidget(scale_spin)
        row_layout.addWidget(remove_btn)

        self._lora_list.addWidget(row)
        self._lora_entries.append((enabled_cb, path_edit, scale_spin, row))

    def _remove_lora_row(self, row: QWidget) -> None:
        self._lora_entries = [
            e for e in self._lora_entries if e[-1] is not row
        ]
        self._lora_list.removeWidget(row)
        row.deleteLater()

    def _get_lora_args(self) -> list[str]:
        """Return the --lora-paths and --lora-scales command fragments."""
        paths = []
        scales = []
        for enabled_cb, path_edit, scale_spin, _ in self._lora_entries:
            if not enabled_cb.isChecked():
                continue
            p = path_edit.text().strip()
            if p:
                paths.append(p)
                scales.append(str(scale_spin.value()))
        if not paths:
            return []
        return ["--lora-paths"] + paths + ["--lora-scales"] + scales

    def _add_ref_row(self, path: str = "", prompt: str = "", enabled: bool = True) -> None:
        """Add a reference image entry with enable checkbox, thumbnail, prompt, and remove button."""
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

        prompt_edit = QLineEdit()
        prompt_edit.setPlaceholderText("Describe this reference...")
        prompt_edit.setText(prompt)

        remove_btn = QPushButton("×")
        remove_btn.setFixedWidth(22)
        remove_btn.clicked.connect(lambda checked=False, r=row: self._remove_ref_row(r))

        row_layout.addWidget(enabled_cb)
        row_layout.addWidget(thumb)
        row_layout.addWidget(prompt_edit, 1)
        row_layout.addWidget(remove_btn)

        self._ref_list.addWidget(row)
        self._ref_entries.append((enabled_cb, thumb, prompt_edit, row))

    def _remove_ref_row(self, row: QWidget) -> None:
        self._ref_entries = [
            e for e in self._ref_entries if e[-1] is not row
        ]
        self._ref_list.removeWidget(row)
        row.deleteLater()

    def _build_prompt(self) -> str:
        """Assemble the main prompt with per-reference-image descriptions."""
        main = self._prompt.toPlainText().strip()
        if not main:
            return ""
        parts = [main]
        idx = 0
        for enabled_cb, thumb, prompt_edit, _ in self._ref_entries:
            if not enabled_cb.isChecked():
                continue
            ref_path = (thumb.imagePath() or "").strip()
            ref_prompt = prompt_edit.text().strip()
            if ref_path and ref_prompt:
                idx += 1
                parts.append(f"Reference image {idx}: {ref_prompt}")
        return "\n".join(parts)

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
            else:
                self._preview.setVisible(False)
                self._use_btn.setEnabled(False)
        else:
            self._preview.setVisible(False)
            self._use_btn.setEnabled(False)


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
