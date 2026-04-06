import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading

from krita import DockWidget, InfoObject
from PyQt5.QtCore import QByteArray
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtCore import QEvent, QObject, QSize
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpacerItem,
    QToolButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

MFLUX_DIR = os.path.expanduser("~/.local/bin")

# Maps model name → (cli_binary, model_flag_or_None, supports_strength, supports_guidance, needs_reference_image)
# Distilled models (klein, turbo, schnell) don't accept a variable guidance scale.
# Each model family has its own CLI; FLUX.2 and FIBO share a CLI across variants.
# Models with needs_reference_image=True use --image-paths [canvas, ref] instead of --image-path canvas.
MODEL_CLI = {
    # FLUX.2 — distilled variants: no guidance; base variants: guidance ok
    "flux2-klein-4b":      ("mflux-generate-flux2",      "flux2-klein-4b",      True,  False, False),
    "flux2-klein-9b":      ("mflux-generate-flux2",      "flux2-klein-9b",      True,  False, False),
    "flux2-klein-base-4b": ("mflux-generate-flux2",      "flux2-klein-base-4b", True,  True,  False),
    "flux2-klein-base-9b": ("mflux-generate-flux2",      "flux2-klein-base-9b", True,  True,  False),
    # FLUX.2 edit — canvas + optional reference image via --image-paths
    "flux2-edit":          ("mflux-generate-flux2-edit", None,                  False, True,  True),
    # Z-Image
    "z-image-turbo":       ("mflux-generate-z-image-turbo", None,               True,  False, False),
    "z-image":             ("mflux-generate-z-image",     None,                 True,  True,  False),
    # FIBO
    "fibo":                ("mflux-generate-fibo",        "fibo",               True,  True,  False),
    "fibo-lite":           ("mflux-generate-fibo",        "fibo-lite",          True,  True,  False),
    # Qwen
    "qwen":                ("mflux-generate-qwen",        None,                 True,  True,  False),
    # Qwen edit — canvas + optional reference image via --image-paths
    "qwen-edit":           ("mflux-generate-qwen-edit",   None,                 False, True,  True),
    # Kontext (image editing via instruction)
    "kontext-dev":         ("mflux-generate-kontext",     "dev",                True,  True,  False),
    "kontext-schnell":     ("mflux-generate-kontext",     "schnell",            True,  False, False),
    # FLUX.1 (legacy)
    "dev":                 ("mflux-generate",             "dev",                True,  True,  False),
    "schnell":             ("mflux-generate",             "schnell",            True,  False, False),
}

# How often (ms) to poll canvas for changes when auto-mode is on.
POLL_INTERVAL_MS = 1500
# How long (ms) to wait after the last detected change before generating.
DEBOUNCE_MS = 2000


class PreviewLabel(QLabel):
    """QLabel that paints its pixmap centered with correct aspect ratio."""

    def __init__(self):
        super().__init__()
        self._ratio = 1.0
        self._source = None

    def setRatio(self, ratio):
        if ratio > 0 and ratio != self._ratio:
            self._ratio = ratio
            self._update_fixed_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_fixed_height()

    def _update_fixed_height(self):
        self.setFixedHeight(max(1, self.heightForWidth(self.width())))

    def setPixmap(self, pixmap):
        self._source = pixmap
        self.update()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return int(width * self._ratio)

    def sizeHint(self):
        return QSize(1, self.heightForWidth(1))

    def minimumSizeHint(self):
        return QSize(1, 1)

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter
        painter = QPainter(self)
        if self._source:
            scaled = self._source.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)


class CollapsibleSection(QWidget):
    """A full-width accordion-style collapsible section."""

    def __init__(self, title, parent=None):
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

    def addHeaderWidget(self, widget):
        """Add a widget to the right side of the header (e.g. a Clear button)."""
        widget.setVisible(False)
        self._toggle.toggled.connect(widget.setVisible)
        self._header_row.addWidget(widget)

    def setContentLayout(self, layout):
        self._content.setLayout(layout)

    def setExpanded(self, expanded):
        self._toggle.setChecked(expanded)

    def _on_toggled(self, checked):
        self._content.setVisible(checked)
        self._toggle.setText(("▼" if checked else "▶") + f"  {self._title}")


class GenerateThread(QThread):
    finished = pyqtSignal(str)        # output path
    errored = pyqtSignal(str)         # error message
    logged = pyqtSignal(str)          # line of stdout/stderr for the log panel
    progress = pyqtSignal(int)        # 0–100

    def __init__(self, cmd, output_path):
        super().__init__()
        self.cmd = cmd
        self.output_path = output_path

    def run(self):
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

    def __init__(self, widget):
        super().__init__(widget)
        widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusOut:
            self.focusLost.emit()
        return False


class KritaiDocker(DockWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kritai")
        self._thread = None
        self._tmp_input = None
        self._tmp_output = None
        self._last_canvas_hash = None
        self._current_doc = None
        self._doc_previews = {}   # uid → QPixmap, session only
        self._doc_settings = {}   # uid → settings dict, session only

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

    def _build_ui(self):
        root = QWidget()
        self.setWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # --- Preview image ---
        self._preview = PreviewLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._preview.setStyleSheet("border-radius: 4px;")
        self._preview.setVisible(False)
        outer.addWidget(self._preview)

        # --- Progress bar (doubles as status display) ---
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("")
        self._progress.setFixedHeight(18)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

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
            "z-image-turbo":       "Fast distilled variant. No guidance.",
            "z-image":             "Full model. Supports guidance.",
            "fibo":                "Generation model with strong style capabilities.",
            "fibo-lite":           "Lighter and faster variant.",
            "qwen":                "Qwen-based generation. Supports guidance.",
            "qwen-edit":           "Instruction-based canvas editing with an optional reference image.",
            "kontext-dev":         "High-quality instruction-based image editing.",
            "kontext-schnell":     "Faster distilled variant. No guidance.",
            "dev":                 "Original full-quality model. Supports guidance.",
            "schnell":             "Fast distilled variant. No guidance.",
        }
        model_groups = {
            "FLUX.2 (recommended)": [
                "flux2-klein-4b",
                "flux2-klein-9b",
                "flux2-klein-base-4b",
                "flux2-klein-base-9b",
                "flux2-edit",
            ],
            "Z-Image": [
                "z-image-turbo",
                "z-image",
            ],
            "FIBO": [
                "fibo",
                "fibo-lite",
            ],
            "Qwen": [
                "qwen",
                "qwen-edit",
            ],
            "Kontext": [
                "kontext-dev",
                "kontext-schnell",
            ],
            "FLUX.1 (legacy)": [
                "dev",
                "schnell",
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
        self._model.setCurrentIndex(self._model.findText("dev"))
        self._model.setToolTip(
            "Which mflux model family to use for generation.\n"
            "FLUX.2 and Z-Image are the fastest and highest quality.\n"
            "FLUX.1 (dev/schnell) are older but widely supported."
        )
        self._model.currentIndexChanged.connect(self._update_model_ui)
        outer.addWidget(self._model)

        # --- Prompt ---
        self._prompt = QPlainTextEdit()
        self._prompt.setPlaceholderText("Prompt…")
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

        ref_row = QWidget()
        ref_layout = QHBoxLayout(ref_row)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        ref_layout.setSpacing(4)
        self._ref_image = QLineEdit()
        self._ref_image.setPlaceholderText("Optional reference image…")
        self._ref_image.setToolTip(
            "Optional second image passed alongside the canvas to the model.\n"
            "Leave empty to use the canvas alone."
        )
        ref_browse = QPushButton("…")
        ref_browse.setFixedWidth(28)
        ref_browse.clicked.connect(self._browse_reference_image)
        ref_layout.addWidget(self._ref_image)
        ref_layout.addWidget(ref_browse)
        form.addRow("Reference", ref_row)
        self._ref_row = ref_row
        self._ref_label = form.labelForField(ref_row)

        self._resolution_scale = QDoubleSpinBox()
        self._resolution_scale.setRange(0.1, 4.0)
        self._resolution_scale.setSingleStep(0.25)
        self._resolution_scale.setValue(0.5)
        self._resolution_scale.setSuffix("")
        self._resolution_scale.setToolTip(
            "Scale of the canvas sent to mflux relative to its original size.\n"
            "0.5× = half resolution (faster, less VRAM).\n"
            "1.0× = full resolution."
        )
        form.addRow("Scale", self._resolution_scale)

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

        self._upscale_btn = QPushButton()
        self._upscale_btn.setIcon(Krita.instance().icon("zoom-in"))
        self._upscale_btn.setToolTip("Upscale preview using SeedVR2")
        self._upscale_btn.setEnabled(False)
        self._upscale_btn.clicked.connect(self._upscale)
        btn_row.addWidget(self._upscale_btn)

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

        # --- Log section ---
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        self._log.setMaximumHeight(240)
        self._log.setVisible(False)
        from PyQt5.QtGui import QFontDatabase
        self._log.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setVisible(False)
        self._clear_btn.clicked.connect(self._clear_log)

        outer.addWidget(self._log)
        outer.addWidget(self._clear_btn)
        outer.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self._update_model_ui()

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    ANNOTATION_TYPE = "kritai_settings"

    def _connect_settings_signals(self):
        # All changes persist settings immediately.
        for signal in [
            self._prompt.textChanged,
            self._negative_prompt.textChanged,
            self._ref_image.textChanged,
            self._model.currentIndexChanged,
            self._quantize.valueChanged,
            self._steps.valueChanged,
            self._guidance.valueChanged,
            self._strength.valueChanged,
            self._resolution_scale.valueChanged,
            self._seed.valueChanged,
            self._random_seed.toggled,
        ]:
            signal.connect(self._save_settings)

        # Auto-refresh triggers only on focus-out for text fields,
        # immediately for all other controls.
        self._negative_prompt.editingFinished.connect(self._on_setting_changed)
        self._ref_image.editingFinished.connect(self._on_setting_changed)

        prompt_filter = _FocusOutSignal(self._prompt)
        prompt_filter.focusLost.connect(self._on_setting_changed)

        for signal in [
            self._model.currentIndexChanged,
            self._quantize.valueChanged,
            self._steps.valueChanged,
            self._guidance.valueChanged,
            self._strength.valueChanged,
            self._resolution_scale.valueChanged,
            self._seed.valueChanged,
            self._random_seed.toggled,
        ]:
            signal.connect(self._on_setting_changed)

    def _on_setting_changed(self):
        if self._auto_btn.isChecked():
            self._debounce_timer.start()

    def _save_settings(self):
        doc = self._current_doc
        if not doc:
            return
        self._doc_settings[doc.fileName() or str(id(doc))] = {
            "prompt": self._prompt.toPlainText(),
            "negative_prompt": self._negative_prompt.text(),
            "reference_image": self._ref_image.text(),
            "model": self._model.currentText(),
            "quantize": self._quantize.value(),
            "steps": self._steps.value(),
            "guidance": self._guidance.value(),
            "strength": self._strength.value() / 100,
            "resolution_scale": self._resolution_scale.value(),
            "seed": self._seed.value(),
            "random_seed": self._random_seed.isChecked(),
        }

    def _on_image_saved(self, filename):
        """Flush in-memory settings to the document annotation on save."""
        for doc in Krita.instance().documents():
            if doc.fileName() == filename:
                data = self._doc_settings.get(doc.fileName() or str(id(doc)))
                if data:
                    raw = json.dumps(data).encode("utf-8")
                    doc.setAnnotation(self.ANNOTATION_TYPE, "Kritai settings", QByteArray(raw))
                break

    def _load_settings(self, doc):
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
            self._prompt, self._negative_prompt, self._ref_image, self._model,
            self._quantize, self._steps, self._guidance,
            self._strength, self._resolution_scale, self._seed, self._random_seed,
        ]
        for w in widgets:
            w.blockSignals(True)

        self._prompt.setPlainText(data.get("prompt", ""))
        self._negative_prompt.setText(data.get("negative_prompt", ""))
        self._ref_image.setText(data.get("reference_image", ""))
        model = data.get("model", "dev")
        idx = self._model.findText(model)
        if idx >= 0:
            self._model.setCurrentIndex(idx)
        self._quantize.setValue(data.get("quantize", 4))
        self._steps.setValue(data.get("steps", 20))
        self._guidance.setValue(data.get("guidance", 3.5))
        self._strength.setValue(int(data.get("strength", 0.75) * 100))
        self._resolution_scale.setValue(data.get("resolution_scale", 0.5))
        self._seed.setValue(data.get("seed", 0))
        self._random_seed.setChecked(data.get("random_seed", False))
        self._seed.setDisabled(self._random_seed.isChecked())

        for w in widgets:
            w.blockSignals(False)

        self._update_model_ui()

    # ------------------------------------------------------------------
    # Auto-mode
    # ------------------------------------------------------------------

    def _on_auto_toggled(self, checked):
        if checked:
            self._last_canvas_hash = self._canvas_hash()
            self._poll_timer.start()
            self._generate()
        else:
            self._poll_timer.stop()
            self._debounce_timer.stop()

    def _export_scaled(self, doc, path):
        """Export the canvas to path, scaled by the resolution scale factor."""
        scale = self._resolution_scale.value()
        doc.setBatchmode(True)
        doc.exportImage(path, InfoObject())
        doc.setBatchmode(False)

        if abs(scale - 1.0) < 0.001:
            return
        img = QImage(path)
        if img.isNull():
            return
        new_w = max(1, int(img.width() * scale))
        new_h = max(1, int(img.height() * scale))
        img.scaled(new_w, new_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation).save(path, "PNG")

    def _canvas_hash(self):
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

    def _poll_canvas(self):
        if self._thread and self._thread.isRunning():
            return
        current = self._canvas_hash()
        if current and current != self._last_canvas_hash:
            self._last_canvas_hash = current
            self._debounce_timer.start()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _set_status(self, text):
        self._progress.setFormat(text)

    def _set_busy(self, busy):
        self._progress.setValue(0)
        self._progress.setVisible(busy)
        self._set_status("Generating…" if busy else "")

    def _generate(self):
        from krita import Krita

        app = Krita.instance()
        doc = app.activeDocument()
        if not doc:
            self._set_status("No active document.")
            return

        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            self._set_status("Enter a prompt first.")
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

        self._export_scaled(doc, self._tmp_input)

        model_name = self._model.currentText()
        cli_name, model_flag, supports_strength, supports_guidance, *rest = MODEL_CLI.get(
            model_name, ("mflux-generate", model_name, True, True, False)
        )
        needs_reference = rest[0] if rest else False
        cli_path = os.path.join(MFLUX_DIR, cli_name)

        cmd = [cli_path, "--prompt", prompt]

        if model_flag:
            cmd += ["--model", model_flag]

        if needs_reference:
            cmd += ["--image-paths", self._tmp_input]
            ref_path = self._ref_image.text().strip()
            if ref_path:
                cmd.append(ref_path)
        else:
            cmd += ["--image-path", self._tmp_input]

        cmd += [
            "--steps", str(self._steps.value()),
            "--output", self._tmp_output,
        ]

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

        self._on_log_message("Running: " + " ".join(cmd))
        self._set_busy(True)
        self._thread = GenerateThread(cmd, self._tmp_output)
        self._thread.finished.connect(self._on_finished)
        self._thread.errored.connect(self._on_error)
        self._thread.logged.connect(self._on_log_message)
        self._thread.progress.connect(self._on_progress)
        self._thread.start()

    def _on_finished(self, output_path):
        self._set_busy(False)
        exists = os.path.exists(output_path)
        size = os.path.getsize(output_path) if exists else 0
        self._on_log_message(
            f"Output path: {output_path}\n"
            f"File exists: {exists}, size: {size} bytes"
        )
        self._show_preview(output_path)
        self._progress.setVisible(False)

    def _on_error(self, message):
        self._set_busy(False)
        self._set_status("Error — see Logs.")
        # Auto-expand the log panel on error so the user notices it.
        self._log_btn.setChecked(True)

    def _on_log_message(self, text):
        if text:
            self._log.appendPlainText(text)
            self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
            if any(kw in text for kw in ("Downloading", "Fetching", "fetching")):
                self._set_status("Downloading…")

    def _on_progress(self, value):
        self._progress.setValue(value)
        if value > 0:
            self._set_status(f"Generating… {value}%")

    def _on_log_toggled(self, checked):
        self._log.setVisible(checked)
        self._clear_btn.setVisible(checked)

    def _clear_log(self):
        self._log.clear()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _show_preview(self, path):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._set_status("Could not load result image.")
            return
        self._preview.setPixmap(pixmap)
        self._preview.setVisible(True)
        self._use_btn.setEnabled(True)
        self._upscale_btn.setEnabled(True)
        if self._current_doc:
            self._doc_previews[self._current_doc.fileName() or str(id(self._current_doc))] = pixmap

    def _import_to_layer(self):
        doc = self._current_doc
        if not doc:
            return
        pixmap = self._doc_previews.get(doc.fileName() or str(id(doc)))
        if not pixmap:
            return

        # Convert the cached pixmap and scale it to the document dimensions.
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

    def _upscale(self):
        if not self._tmp_output or not os.path.exists(self._tmp_output):
            self._set_status("No preview to upscale.")
            return
        if self._thread and self._thread.isRunning():
            return

        # --- Modal options dialog ---
        dlg = QDialog(self)
        dlg.setWindowTitle("Upscale Options")
        layout = QVBoxLayout(dlg)
        form = QFormLayout()
        layout.addLayout(form)

        resolution = QDoubleSpinBox()
        resolution.setRange(1.0, 8.0)
        resolution.setSingleStep(0.5)
        resolution.setValue(2.0)
        resolution.setToolTip("Upscale factor relative to the current preview size.")
        form.addRow("Resolution", resolution)

        softness_row = QWidget()
        softness_layout = QHBoxLayout(softness_row)
        softness_layout.setContentsMargins(0, 0, 0, 0)
        softness_layout.setSpacing(6)
        softness = QSlider(Qt.Horizontal)
        softness.setRange(0, 100)
        softness.setValue(0)
        softness_value_label = QLabel("0.00")
        softness_value_label.setFixedWidth(30)
        softness.valueChanged.connect(lambda v: softness_value_label.setText(f"{v / 100:.2f}"))
        softness_layout.addWidget(softness)
        softness_layout.addWidget(softness_value_label)
        softness_row.setToolTip("0.0 = off, 1.0 = maximum softness.")
        form.addRow("Softness", softness_row)

        quantize = QSpinBox()
        quantize.setSpecialValueText("None")
        quantize.setRange(0, 8)
        quantize.setValue(self._quantize.value())
        form.addRow("Quantize", quantize)

        seed_row = QWidget()
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        dlg_seed = QSpinBox()
        dlg_seed.setRange(0, 2_000_000_000)
        dlg_seed.setValue(0)
        dlg_random_seed = QCheckBox("Random")
        dlg_random_seed.setChecked(True)
        dlg_random_seed.toggled.connect(dlg_seed.setDisabled)
        dlg_seed.setDisabled(True)
        seed_layout.addWidget(dlg_seed)
        seed_layout.addWidget(dlg_random_seed)
        form.addRow("Seed", seed_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec_() != QDialog.Accepted:
            return

        # --- Run upscale ---
        upscaled_path = os.path.join(
            tempfile.gettempdir(), f"kf_upscaled_{os.getpid()}.png"
        )
        cli_path = os.path.join(MFLUX_DIR, "mflux-upscale-seedvr2")
        cmd = [
            cli_path,
            "--image-path", self._tmp_output,
            "--resolution", f"{resolution.value()}x",
            "--output", upscaled_path,
        ]

        if softness.value() > 0:
            cmd += ["--softness", str(softness.value() / 100)]

        if quantize.value() > 0:
            cmd += ["--quantize", str(quantize.value())]

        if not dlg_random_seed.isChecked():
            cmd += ["--seed", str(dlg_seed.value())]

        self._on_log_message("Running: " + " ".join(cmd))
        self._set_busy(True)
        self._thread = GenerateThread(cmd, upscaled_path)
        self._thread.finished.connect(self._on_finished)
        self._thread.errored.connect(self._on_error)
        self._thread.logged.connect(self._on_log_message)
        self._thread.progress.connect(self._on_progress)
        self._thread.start()

    # ------------------------------------------------------------------

    def _update_model_ui(self):
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

        self._ref_row.setVisible(needs_ref)
        if self._ref_label:
            self._ref_label.setVisible(needs_ref)

        self._update_prompt_placeholder()

    def _update_prompt_placeholder(self):
        model_name = self._model.currentText().strip()
        edit_models = {"kontext-dev", "kontext-schnell", "flux2-edit", "qwen-edit"}
        if model_name in edit_models:
            self._prompt.setPlaceholderText("Describe your changes…")
        else:
            self._prompt.setPlaceholderText("Describe your image…")

    def _browse_reference_image(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select reference image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.tiff)"
        )
        if path:
            self._ref_image.setText(path)

    def _update_preview_ratio(self, doc):
        if doc and doc.width() > 0:
            self._preview.setRatio(doc.height() / doc.width())


    def canvasChanged(self, canvas):
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
                self._upscale_btn.setEnabled(True)
            else:
                self._preview.setVisible(False)
                self._use_btn.setEnabled(False)
                self._upscale_btn.setEnabled(False)
        else:
            self._preview.setVisible(False)
            self._use_btn.setEnabled(False)
            self._upscale_btn.setEnabled(False)
