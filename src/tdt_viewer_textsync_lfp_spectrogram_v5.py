import sys
import os
import pickle
from fractions import Fraction
import numpy as np

# Keep pyqtgraph and the application on the same Qt binding. Mixing PyQt and
# PySide can make the macOS Cocoa platform plugin fail during startup.
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from camera_sync import CameraDock

APP_NAME = "TDT Viewer TextSync LFP Spectrogram v5"

try:
    from scipy import signal as scipy_signal
except ImportError as e:
    raise SystemExit("The 'scipy' package is required. Install it with: pip install scipy") from e

try:
    import tdt
except ImportError as e:
    raise SystemExit("The 'tdt' package is required. Install it with: pip install tdt") from e


class CachedStream:
    def __init__(self, data, fs, start_time=0.0):
        self.data = data
        self.fs = fs
        self.start_time = start_time


class CachedData:
    def __init__(self):
        self.streams = {}
        self.epocs = None


class CursorPlotWidget(pg.PlotWidget):
    doubleClicked = QtCore.Signal(float)

    def mouseDoubleClickEvent(self, event):
        vb = self.getPlotItem().vb
        pos = vb.mapSceneToView(event.position())
        self.doubleClicked.emit(float(pos.x()))
        super().mouseDoubleClickEvent(event)


class ChannelLabelAxis(pg.AxisItem):
    def __init__(self, orientation="left", **kwargs):
        super().__init__(orientation=orientation, **kwargs)
        self._manual_label_pairs = []

    def set_manual_labels(self, ticks):
        self._manual_label_pairs = [(float(pos), str(label)) for pos, label in ticks]
        self.setTicks([self._manual_label_pairs])

    def clear_manual_labels(self):
        self._manual_label_pairs = []
        self.setTicks([[]])

    def tickStrings(self, values, scale, spacing):
        if not self._manual_label_pairs:
            return [""] * len(values)

        tolerance = max(1e-6, abs(float(spacing or 0.0)) * 1e-3)
        labels = []
        for value in values:
            label = ""
            for pos, text in self._manual_label_pairs:
                if abs(float(value) - pos) <= tolerance:
                    label = text
                    break
            labels.append(label)
        return labels


class DoubleClickToolButton(QtWidgets.QToolButton):
    doubleClicked = QtCore.Signal()

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


class CollapsibleSection(QtWidgets.QWidget):
    toggled = QtCore.Signal(bool)

    def __init__(self, title="", expanded=False, parent=None):
        super().__init__(parent)

        self.toggle_btn = DoubleClickToolButton()
        self.toggle_btn.setText(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.toggle_btn.clicked.connect(self._on_toggled)

        self.content = QtWidgets.QWidget()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.content)
        self.content.setVisible(expanded)

    def _on_toggled(self, checked):
        self.toggle_btn.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self.content.setVisible(checked)
        self.toggled.emit(checked)

    def set_expanded(self, expanded: bool):
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.content.setVisible(expanded)

    def is_expanded(self):
        return self.toggle_btn.isChecked()


class ScrollChannelGroup(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, title="", height=110, expanded=False, text_entry=False, parent=None):
        super().__init__(parent)

        self.section = CollapsibleSection(title, expanded=expanded)
        # All channel selectors support direct entry such as 1,2,5-8, all, none.
        self.section.toggle_btn.doubleClicked.connect(self._edit_channels_by_text)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(height)
        self.scroll.setMaximumHeight(height)

        self.content = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.content)
        self.grid.setContentsMargins(4, 4, 4, 4)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(4)
        self.scroll.setWidget(self.content)

        inner_layout = QtWidgets.QVBoxLayout(self.section.content)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(self.scroll)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.section)

        self.checkboxes = []

    def clear_checks(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.checkboxes = []

    def set_channels(self, prefix: str, n_channels: int, checked_n: int = 8, columns: int = 2):
        self.clear_checks()
        for ch in range(n_channels):
            cb = QtWidgets.QCheckBox(f"{prefix} {ch + 1}")
            cb.setChecked(ch < min(checked_n, n_channels))
            cb.toggled.connect(lambda _checked=False: self.changed.emit())
            row = ch // columns
            col = ch % columns
            self.grid.addWidget(cb, row, col)
            self.checkboxes.append(cb)

    def selected_channels(self):
        return [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]

    def set_selected_channels(self, channels):
        channels = set(channels or [])
        for i, cb in enumerate(self.checkboxes):
            cb.blockSignals(True)
            cb.setChecked(i in channels)
            cb.blockSignals(False)

    def set_expanded(self, expanded: bool):
        self.section.set_expanded(expanded)

    def is_expanded(self):
        return self.section.is_expanded()

    def _selected_channels_text(self):
        return ",".join(str(i + 1) for i in self.selected_channels())

    def _parse_channel_text(self, text):
        text = (text or "").strip().lower()
        if not text or text == "none":
            return []
        if text == "all":
            return list(range(len(self.checkboxes)))

        selected = set()
        for part in (part.strip() for part in text.split(",") if part.strip()):
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start = int(start_text.strip())
                end = int(end_text.strip())
                if start > end:
                    start, end = end, start
                selected.update(range(start, end + 1))
            else:
                selected.add(int(part))

        max_channel = len(self.checkboxes)
        invalid = sorted(ch for ch in selected if ch < 1 or ch > max_channel)
        if invalid:
            raise ValueError(f"Channel must be between 1 and {max_channel}.")
        return [ch - 1 for ch in sorted(selected)]

    def _edit_channels_by_text(self):
        if not self.checkboxes:
            return

        max_channel = len(self.checkboxes)
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Select Channels",
            f"Channels 1-{max_channel}\nExamples: 1,2,5-8 or all / none",
            QtWidgets.QLineEdit.Normal,
            self._selected_channels_text(),
        )
        if not ok:
            return

        try:
            channels = self._parse_channel_text(text)
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Channels", str(exc))
            return

        self.set_selected_channels(channels)
        self.changed.emit()


class ScrollCheckGroup(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, title="", height=100, expanded=False, parent=None):
        super().__init__(parent)

        self.section = CollapsibleSection(title, expanded=expanded)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumHeight(height)
        self.scroll.setMaximumHeight(height)

        self.content = QtWidgets.QWidget()
        self.layout_inner = QtWidgets.QVBoxLayout(self.content)
        self.layout_inner.setContentsMargins(4, 4, 4, 4)
        self.layout_inner.setSpacing(2)
        self.layout_inner.addStretch()
        self.scroll.setWidget(self.content)

        inner_layout = QtWidgets.QVBoxLayout(self.section.content)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(self.scroll)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.section)

        self.checkboxes = []

    def clear_checks(self):
        for cb in self.checkboxes:
            cb.deleteLater()
        self.checkboxes = []
        while self.layout_inner.count() > 1:
            item = self.layout_inner.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def set_items(self, names, checked_first_n=None, checked_all=False):
        self.clear_checks()
        for i, name in enumerate(names):
            cb = QtWidgets.QCheckBox(str(name))
            if checked_all:
                cb.setChecked(True)
            elif checked_first_n is not None:
                cb.setChecked(i < checked_first_n)
            cb.toggled.connect(lambda _checked=False: self.changed.emit())
            self.layout_inner.insertWidget(self.layout_inner.count() - 1, cb)
            self.checkboxes.append(cb)

    def selected_items(self):
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]

    def set_selected_items(self, names):
        names = set(names or [])
        for cb in self.checkboxes:
            cb.blockSignals(True)
            cb.setChecked(cb.text() in names)
            cb.blockSignals(False)

    def set_expanded(self, expanded: bool):
        self.section.set_expanded(expanded)

    def is_expanded(self):
        return self.section.is_expanded()


class TdtViewerWindow(QtWidgets.QMainWindow):
    TXT_POSITION_TOP = "Top"
    TXT_POSITION_MIDDLE = "Middle"
    TXT_POSITION_BOTTOM = "Bottom"

    XAXIS_BOTTOM_ONLY = "Bottom only"
    XAXIS_ALL = "All panels"

    def __init__(self, controller, block_path=""):
        super().__init__()
        self.controller = controller
        self.settings = QtCore.QSettings("PingChou", "TDTTextSyncGainNormalizedV4DualSpectrogram")

        self.block_path = ""
        self.data = None
        self.stream_names = []
        self.epoc_names = []
        self.epoc_onsets = {}
        self.epoc_values = {}

        self.default_window_sec = 5.0
        self.default_channel_spacing = 1.0
        self.default_white_mode = True
        self.default_show_grid = True
        self.default_show_scale_bar = True
        self.default_lfp_amp = 1.0
        self.default_mu_amp = 1.0
        self.default_txt_amp = 1.0
        self.default_signal_gain = 1.0
        self.default_lfp_bar = 500.0
        self.default_mu_bar = 100.0
        self.default_txt_bar = 1.0
        self.default_line_width = 1.0
        self.default_text_trace_fs = 50.0
        self.default_txt_position = self.TXT_POSITION_MIDDLE
        self.default_epoc_name = "PC3_"
        self.default_xaxis_mode = self.XAXIS_BOTTOM_ONLY
        self.default_show_spectrogram = True
        self.default_spectrogram_fmin = 0.5
        self.default_spectrogram_fmax = 300.0
        self.default_spectrogram_resolution_hz = 0.5
        self.default_spectrogram_window_sec = 2.0
        self.default_spectrogram_time_step_sec = 0.5
        self.default_spectrogram_overlap = 75
        self.default_spectrogram_db_min = -20.0
        self.default_spectrogram_db_max = 40.0
        self.default_spectrogram_colormap = "viridis"
        self.default_spectrogram_power_mode = "Real power"
        self.default_spectrogram_percentage_min = 0.0
        self.default_spectrogram_percentage_max = 5.0
        self.default_spectrogram_detrend = "Constant"
        self.default_spectrogram_gaussian_smoothing = False
        self.default_spectrogram_gaussian_width_bins = 3.0
        self.default_show_txt_spectrogram = True
        self.default_txt_spectrogram_fmin = 0.5
        self.default_txt_spectrogram_fmax = 20.0
        self.default_txt_spectrogram_resolution_hz = 0.5
        self.default_txt_spectrogram_window_sec = 2.0
        self.default_txt_spectrogram_time_step_sec = 0.5
        self.default_txt_spectrogram_overlap = 75
        self.default_txt_spectrogram_db_min = -20.0
        self.default_txt_spectrogram_db_max = 40.0
        self.default_txt_spectrogram_colormap = "viridis"
        self.default_txt_spectrogram_power_mode = "Real power"
        self.default_txt_spectrogram_percentage_min = 0.0
        self.default_txt_spectrogram_percentage_max = 5.0
        self.default_txt_spectrogram_detrend = "Constant"
        self.default_txt_spectrogram_gaussian_smoothing = False
        self.default_txt_spectrogram_gaussian_width_bins = 3.0

        self.current_time = 0.0
        self.window_sec = self.default_window_sec
        self.channel_spacing = self.default_channel_spacing
        self.white_mode = self.default_white_mode
        self.show_grid = self.default_show_grid
        self.show_scale_bar = self.default_show_scale_bar
        self.export_include_settings = False
        self.line_width = self.default_line_width
        self.txt_position = self.default_txt_position
        self.xaxis_mode = self.default_xaxis_mode
        self.show_spectrogram = self.default_show_spectrogram
        self.show_txt_spectrogram = self.default_show_txt_spectrogram

        self.stream_amp = {}
        self.signal_gain = self.default_signal_gain
        self.scale_bar_value_lfp = self.default_lfp_bar
        self.scale_bar_value_mus = self.default_mu_bar
        self.scale_bar_value_txt = self.default_txt_bar

        self.total_duration = 1.0
        self.target_points_lfp = 4000
        self.target_points_mus = None

        self.text_trace_path = ""
        self.text_trace_data = None
        self.text_trace_fs = self.default_text_trace_fs
        self.text_trace_start_time = 0.0

        self.cursor_time = 0.0
        self.show_cursor_on_traces = False
        self._cursor_syncing = False
        self.cursor_lines = []
        self.cursor_labels = {}
        self.cursor_line_pen = pg.mkPen((200, 0, 0), width=1.5)

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._do_refresh_plot)

        self.setWindowTitle(APP_NAME)
        self.resize(1900, 1100)
        self.setMinimumSize(1100, 700)

        self._build_ui()
        self.camera_dock = CameraDock(self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.camera_dock)
        self.camera_dock.timeRequested.connect(self._camera_time_requested)
        self.camera_dock.visibilityChanged.connect(self._camera_visibility_changed)
        self.camera_dock.hide()
        self._apply_plot_theme()

        if block_path:
            self.load_block(block_path)
        else:
            self.load_settings()

        if not self.block_path and not self.text_trace_path:
            self.statusBar().showMessage("Open a TDT block or a text trace.")

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        top_btn_col = QtWidgets.QVBoxLayout()

        self.open_btn = QtWidgets.QPushButton("Open Block")
        self.open_btn.clicked.connect(self.choose_block)
        self.open_btn.setMinimumHeight(32)
        self.open_btn.setFixedWidth(120)
        top_btn_col.addWidget(self.open_btn)

        self.open_text_btn = QtWidgets.QPushButton("Open Text")
        self.open_text_btn.clicked.connect(self.choose_text_trace)
        self.open_text_btn.setMinimumHeight(32)
        self.open_text_btn.setFixedWidth(120)
        top_btn_col.addWidget(self.open_text_btn)

        self.open_session_btn = QtWidgets.QPushButton("Open Session")
        self.open_session_btn.clicked.connect(self.open_session_dialog)
        self.open_session_btn.setMinimumHeight(32)
        self.open_session_btn.setFixedWidth(120)
        top_btn_col.addWidget(self.open_session_btn)

        self.save_session_btn = QtWidgets.QPushButton("Save Session")
        self.save_session_btn.clicked.connect(self.save_session_dialog)
        self.save_session_btn.setMinimumHeight(32)
        self.save_session_btn.setFixedWidth(120)
        top_btn_col.addWidget(self.save_session_btn)

        self.new_viewer_btn = QtWidgets.QPushButton("New Viewer")
        self.new_viewer_btn.clicked.connect(self.controller.open_new_viewer_dialog)
        self.new_viewer_btn.setMinimumHeight(32)
        self.new_viewer_btn.setFixedWidth(120)
        top_btn_col.addWidget(self.new_viewer_btn)

        self.credits_btn = QtWidgets.QPushButton("Credits")
        self.credits_btn.clicked.connect(self.show_credits)
        self.credits_btn.setMinimumHeight(32)
        self.credits_btn.setFixedWidth(120)
        top_btn_col.addWidget(self.credits_btn)

        self.camera_btn = QtWidgets.QPushButton("Camera")
        self.camera_btn.setCheckable(True)
        self.camera_btn.setEnabled(False)
        self.camera_btn.setMinimumHeight(32)
        self.camera_btn.setFixedWidth(120)
        self.camera_btn.toggled.connect(self._toggle_camera)
        top_btn_col.addWidget(self.camera_btn)

        left_layout.addLayout(top_btn_col)

        self.block_section = CollapsibleSection("Block", expanded=False)
        self.block_label = QtWidgets.QPlainTextEdit()
        self.block_label.setReadOnly(True)
        self.block_label.setMaximumHeight(90)
        self.block_label.setMinimumHeight(90)
        self.block_label.setPlainText("Block:\n(not selected)")
        self.block_label.setStyleSheet("font-size: 11px;")
        block_layout = QtWidgets.QVBoxLayout(self.block_section.content)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.addWidget(self.block_label)
        left_layout.addWidget(self.block_section)

        self.text_section = CollapsibleSection("Text Trace", expanded=False)
        self.text_trace_label = QtWidgets.QPlainTextEdit()
        self.text_trace_label.setReadOnly(True)
        self.text_trace_label.setMaximumHeight(70)
        self.text_trace_label.setMinimumHeight(70)
        self.text_trace_label.setPlainText("Text Trace:\n(not selected)")
        self.text_trace_label.setStyleSheet("font-size: 11px;")
        text_layout = QtWidgets.QVBoxLayout(self.text_section.content)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.addWidget(self.text_trace_label)
        left_layout.addWidget(self.text_section)

        left_layout.addWidget(QtWidgets.QLabel("TXT position"))
        self.txt_position_combo = QtWidgets.QComboBox()
        self.txt_position_combo.addItems([
            self.TXT_POSITION_TOP,
            self.TXT_POSITION_MIDDLE,
            self.TXT_POSITION_BOTTOM,
        ])
        self.txt_position_combo.setCurrentText(self.txt_position)
        self.txt_position_combo.currentTextChanged.connect(self._txt_position_changed)
        self.txt_position_combo.setMaximumWidth(150)
        left_layout.addWidget(self.txt_position_combo)

        left_layout.addWidget(QtWidgets.QLabel("X-axis display"))
        self.xaxis_mode_combo = QtWidgets.QComboBox()
        self.xaxis_mode_combo.addItems([self.XAXIS_BOTTOM_ONLY, self.XAXIS_ALL])
        self.xaxis_mode_combo.setCurrentText(self.xaxis_mode)
        self.xaxis_mode_combo.currentTextChanged.connect(self._xaxis_mode_changed)
        self.xaxis_mode_combo.setMaximumWidth(150)
        left_layout.addWidget(self.xaxis_mode_combo)

        self.stream_section = CollapsibleSection("Streams", expanded=False)
        self.stream_list = QtWidgets.QListWidget()
        self.stream_list.setMaximumHeight(90)
        self.stream_list.itemChanged.connect(self.refresh_plot)
        stream_layout = QtWidgets.QVBoxLayout(self.stream_section.content)
        stream_layout.setContentsMargins(0, 0, 0, 0)
        stream_layout.addWidget(self.stream_list)
        left_layout.addWidget(self.stream_section)

        self.lfp_channels = ScrollChannelGroup("LFP ch", height=90, expanded=False, text_entry=True)
        self.lfp_channels.changed.connect(self.refresh_plot)
        left_layout.addWidget(self.lfp_channels)

        self.mus_channels = ScrollChannelGroup("MU ch", height=90, expanded=False, text_entry=True)
        self.mus_channels.changed.connect(self.refresh_plot)
        left_layout.addWidget(self.mus_channels)

        self.txt_channels = ScrollChannelGroup("TXT ch", height=90, expanded=False)
        self.txt_channels.changed.connect(self.refresh_plot)
        left_layout.addWidget(self.txt_channels)

        self.spectrogram_section = CollapsibleSection("LFP spectrogram", expanded=True)
        spec_form = QtWidgets.QFormLayout(self.spectrogram_section.content)
        spec_form.setContentsMargins(0, 0, 0, 0)

        self.spectrogram_show_check = QtWidgets.QCheckBox("Show")
        self.spectrogram_show_check.setChecked(self.show_spectrogram)
        self.spectrogram_show_check.toggled.connect(self._spectrogram_settings_changed)
        spec_form.addRow(self.spectrogram_show_check)

        self.spectrogram_power_mode_combo = QtWidgets.QComboBox()
        self.spectrogram_power_mode_combo.addItems(["Real power", "Percentage power"])
        self.spectrogram_power_mode_combo.setCurrentText(self.default_spectrogram_power_mode)
        self.spectrogram_power_mode_combo.currentTextChanged.connect(
            self._spectrogram_power_mode_changed
        )
        spec_form.addRow("Power display", self.spectrogram_power_mode_combo)

        self.spectrogram_detrend_combo = QtWidgets.QComboBox()
        self.spectrogram_detrend_combo.addItems(["None", "Constant", "Linear"])
        self.spectrogram_detrend_combo.setCurrentText(self.default_spectrogram_detrend)
        self.spectrogram_detrend_combo.currentTextChanged.connect(
            self._spectrogram_settings_changed
        )
        spec_form.addRow("Detrend", self.spectrogram_detrend_combo)

        self.spectrogram_fmin_spin = QtWidgets.QDoubleSpinBox()
        self.spectrogram_fmin_spin.setRange(0.0, 10000.0)
        self.spectrogram_fmin_spin.setValue(self.default_spectrogram_fmin)
        self.spectrogram_fmin_spin.setSuffix(" Hz")
        self.spectrogram_fmin_spin.valueChanged.connect(self._spectrogram_settings_changed)
        spec_form.addRow("Min freq", self.spectrogram_fmin_spin)

        self.spectrogram_fmax_spin = QtWidgets.QDoubleSpinBox()
        self.spectrogram_fmax_spin.setRange(1.0, 10000.0)
        self.spectrogram_fmax_spin.setValue(self.default_spectrogram_fmax)
        self.spectrogram_fmax_spin.setSuffix(" Hz")
        self.spectrogram_fmax_spin.valueChanged.connect(self._spectrogram_settings_changed)
        spec_form.addRow("Max freq", self.spectrogram_fmax_spin)

        self.spectrogram_resolution_spin = QtWidgets.QDoubleSpinBox()
        self.spectrogram_resolution_spin.setRange(0.05, 100.0)
        self.spectrogram_resolution_spin.setDecimals(3)
        self.spectrogram_resolution_spin.setValue(self.default_spectrogram_resolution_hz)
        self.spectrogram_resolution_spin.setSuffix(" Hz")
        self.spectrogram_resolution_spin.valueChanged.connect(self._spectrogram_settings_changed)
        spec_form.addRow("Freq resolution", self.spectrogram_resolution_spin)

        self.spectrogram_window_spin = QtWidgets.QDoubleSpinBox()
        self.spectrogram_window_spin.setRange(0.02, 10.0)
        self.spectrogram_window_spin.setDecimals(3)
        self.spectrogram_window_spin.setValue(self.default_spectrogram_window_sec)
        self.spectrogram_window_spin.setSuffix(" s")
        self.spectrogram_window_spin.valueChanged.connect(self._spectrogram_settings_changed)
        spec_form.addRow("STFT window", self.spectrogram_window_spin)

        self.spectrogram_time_step_spin = QtWidgets.QDoubleSpinBox()
        self.spectrogram_time_step_spin.setRange(0.01, 10.0)
        self.spectrogram_time_step_spin.setDecimals(3)
        self.spectrogram_time_step_spin.setValue(self.default_spectrogram_time_step_sec)
        self.spectrogram_time_step_spin.setSuffix(" s")
        self.spectrogram_time_step_spin.valueChanged.connect(self._spectrogram_settings_changed)
        spec_form.addRow("Max time step", self.spectrogram_time_step_spin)

        self.spectrogram_overlap_spin = QtWidgets.QSpinBox()
        self.spectrogram_overlap_spin.setRange(0, 95)
        self.spectrogram_overlap_spin.setValue(self.default_spectrogram_overlap)
        self.spectrogram_overlap_spin.setSuffix(" %")
        self.spectrogram_overlap_spin.valueChanged.connect(self._spectrogram_settings_changed)
        spec_form.addRow("Overlap", self.spectrogram_overlap_spin)

        self.spectrogram_gaussian_check = QtWidgets.QCheckBox("Enable")
        self.spectrogram_gaussian_check.setChecked(self.default_spectrogram_gaussian_smoothing)
        self.spectrogram_gaussian_check.toggled.connect(self._spectrogram_gaussian_toggled)
        spec_form.addRow("Gaussian smoothing", self.spectrogram_gaussian_check)

        self.spectrogram_gaussian_width_spin = QtWidgets.QDoubleSpinBox()
        self.spectrogram_gaussian_width_spin.setRange(0.1, 1000.0)
        self.spectrogram_gaussian_width_spin.setDecimals(3)
        self.spectrogram_gaussian_width_spin.setValue(self.default_spectrogram_gaussian_width_bins)
        self.spectrogram_gaussian_width_spin.setSuffix(" bins")
        self.spectrogram_gaussian_width_spin.setToolTip(
            "NeuroExplorer-compatible Gaussian full width at half maximum (FWHM)."
        )
        self.spectrogram_gaussian_width_spin.valueChanged.connect(self._spectrogram_settings_changed)
        spec_form.addRow("Filter width", self.spectrogram_gaussian_width_spin)
        self._update_gaussian_controls()

        self.spectrogram_db_min_spin = QtWidgets.QDoubleSpinBox()
        self.spectrogram_db_min_spin.setRange(-200.0, 200.0)
        self.spectrogram_db_min_spin.setValue(self.default_spectrogram_db_min)
        self.spectrogram_db_min_spin.setSuffix(" dB")
        self.spectrogram_db_min_spin.valueChanged.connect(self._spectrogram_settings_changed)
        spec_form.addRow("Color min", self.spectrogram_db_min_spin)

        self.spectrogram_db_max_spin = QtWidgets.QDoubleSpinBox()
        self.spectrogram_db_max_spin.setRange(-200.0, 200.0)
        self.spectrogram_db_max_spin.setValue(self.default_spectrogram_db_max)
        self.spectrogram_db_max_spin.setSuffix(" dB")
        self.spectrogram_db_max_spin.valueChanged.connect(self._spectrogram_settings_changed)
        spec_form.addRow("Color max", self.spectrogram_db_max_spin)

        self.spectrogram_colormap_combo = QtWidgets.QComboBox()
        self.spectrogram_colormap_combo.addItems(["viridis", "jet"])
        self.spectrogram_colormap_combo.setCurrentText(self.default_spectrogram_colormap)
        self.spectrogram_colormap_combo.currentTextChanged.connect(self._spectrogram_settings_changed)
        spec_form.addRow("Color map", self.spectrogram_colormap_combo)

        self.spectrogram_fs_label = QtWidgets.QLabel("Analysis fs: -")
        self.spectrogram_fs_label.setWordWrap(True)
        spec_form.addRow(self.spectrogram_fs_label)
        left_layout.addWidget(self.spectrogram_section)

        self.spectrogram_channels = ScrollChannelGroup("Spectrogram ch", height=90, expanded=True)
        self.spectrogram_channels.changed.connect(self.refresh_plot)
        left_layout.addWidget(self.spectrogram_channels)

        self.txt_spectrogram_section = CollapsibleSection("TXT spectrogram", expanded=True)
        txt_spec_form = QtWidgets.QFormLayout(self.txt_spectrogram_section.content)
        txt_spec_form.setContentsMargins(0, 0, 0, 0)

        self.txt_spectrogram_show_check = QtWidgets.QCheckBox("Show")
        self.txt_spectrogram_show_check.setChecked(self.show_txt_spectrogram)
        self.txt_spectrogram_show_check.toggled.connect(self._txt_spectrogram_settings_changed)
        txt_spec_form.addRow(self.txt_spectrogram_show_check)

        self.txt_spectrogram_power_mode_combo = QtWidgets.QComboBox()
        self.txt_spectrogram_power_mode_combo.addItems(["Real power", "Percentage power"])
        self.txt_spectrogram_power_mode_combo.setCurrentText(self.default_txt_spectrogram_power_mode)
        self.txt_spectrogram_power_mode_combo.currentTextChanged.connect(
            self._txt_spectrogram_power_mode_changed
        )
        txt_spec_form.addRow("Power display", self.txt_spectrogram_power_mode_combo)

        self.txt_spectrogram_detrend_combo = QtWidgets.QComboBox()
        self.txt_spectrogram_detrend_combo.addItems(["None", "Constant", "Linear"])
        self.txt_spectrogram_detrend_combo.setCurrentText(self.default_txt_spectrogram_detrend)
        self.txt_spectrogram_detrend_combo.currentTextChanged.connect(
            self._txt_spectrogram_settings_changed
        )
        txt_spec_form.addRow("Detrend", self.txt_spectrogram_detrend_combo)

        self.txt_spectrogram_fmin_spin = QtWidgets.QDoubleSpinBox()
        self.txt_spectrogram_fmin_spin.setRange(0.0, 10000.0)
        self.txt_spectrogram_fmin_spin.setValue(self.default_txt_spectrogram_fmin)
        self.txt_spectrogram_fmin_spin.setSuffix(" Hz")
        self.txt_spectrogram_fmin_spin.valueChanged.connect(self._txt_spectrogram_settings_changed)
        txt_spec_form.addRow("Min freq", self.txt_spectrogram_fmin_spin)

        self.txt_spectrogram_fmax_spin = QtWidgets.QDoubleSpinBox()
        self.txt_spectrogram_fmax_spin.setRange(0.001, 10000.0)
        self.txt_spectrogram_fmax_spin.setValue(self.default_txt_spectrogram_fmax)
        self.txt_spectrogram_fmax_spin.setSuffix(" Hz")
        self.txt_spectrogram_fmax_spin.valueChanged.connect(self._txt_spectrogram_settings_changed)
        txt_spec_form.addRow("Max freq", self.txt_spectrogram_fmax_spin)

        self.txt_spectrogram_resolution_spin = QtWidgets.QDoubleSpinBox()
        self.txt_spectrogram_resolution_spin.setRange(0.001, 100.0)
        self.txt_spectrogram_resolution_spin.setDecimals(3)
        self.txt_spectrogram_resolution_spin.setValue(self.default_txt_spectrogram_resolution_hz)
        self.txt_spectrogram_resolution_spin.setSuffix(" Hz")
        self.txt_spectrogram_resolution_spin.valueChanged.connect(self._txt_spectrogram_settings_changed)
        txt_spec_form.addRow("Freq resolution", self.txt_spectrogram_resolution_spin)

        self.txt_spectrogram_window_spin = QtWidgets.QDoubleSpinBox()
        self.txt_spectrogram_window_spin.setRange(0.02, 60.0)
        self.txt_spectrogram_window_spin.setDecimals(3)
        self.txt_spectrogram_window_spin.setValue(self.default_txt_spectrogram_window_sec)
        self.txt_spectrogram_window_spin.setSuffix(" s")
        self.txt_spectrogram_window_spin.valueChanged.connect(self._txt_spectrogram_settings_changed)
        txt_spec_form.addRow("STFT window", self.txt_spectrogram_window_spin)

        self.txt_spectrogram_time_step_spin = QtWidgets.QDoubleSpinBox()
        self.txt_spectrogram_time_step_spin.setRange(0.01, 10.0)
        self.txt_spectrogram_time_step_spin.setDecimals(3)
        self.txt_spectrogram_time_step_spin.setValue(self.default_txt_spectrogram_time_step_sec)
        self.txt_spectrogram_time_step_spin.setSuffix(" s")
        self.txt_spectrogram_time_step_spin.valueChanged.connect(self._txt_spectrogram_settings_changed)
        txt_spec_form.addRow("Max time step", self.txt_spectrogram_time_step_spin)

        self.txt_spectrogram_overlap_spin = QtWidgets.QSpinBox()
        self.txt_spectrogram_overlap_spin.setRange(0, 95)
        self.txt_spectrogram_overlap_spin.setValue(self.default_txt_spectrogram_overlap)
        self.txt_spectrogram_overlap_spin.setSuffix(" %")
        self.txt_spectrogram_overlap_spin.valueChanged.connect(self._txt_spectrogram_settings_changed)
        txt_spec_form.addRow("Overlap", self.txt_spectrogram_overlap_spin)

        self.txt_spectrogram_gaussian_check = QtWidgets.QCheckBox("Enable")
        self.txt_spectrogram_gaussian_check.setChecked(self.default_txt_spectrogram_gaussian_smoothing)
        self.txt_spectrogram_gaussian_check.toggled.connect(self._txt_spectrogram_gaussian_toggled)
        txt_spec_form.addRow("Gaussian smoothing", self.txt_spectrogram_gaussian_check)

        self.txt_spectrogram_gaussian_width_spin = QtWidgets.QDoubleSpinBox()
        self.txt_spectrogram_gaussian_width_spin.setRange(0.1, 1000.0)
        self.txt_spectrogram_gaussian_width_spin.setDecimals(3)
        self.txt_spectrogram_gaussian_width_spin.setValue(self.default_txt_spectrogram_gaussian_width_bins)
        self.txt_spectrogram_gaussian_width_spin.setSuffix(" bins")
        self.txt_spectrogram_gaussian_width_spin.setToolTip(
            "NeuroExplorer-compatible Gaussian full width at half maximum (FWHM)."
        )
        self.txt_spectrogram_gaussian_width_spin.valueChanged.connect(
            self._txt_spectrogram_settings_changed
        )
        txt_spec_form.addRow("Filter width", self.txt_spectrogram_gaussian_width_spin)
        self._update_txt_gaussian_controls()

        self.txt_spectrogram_db_min_spin = QtWidgets.QDoubleSpinBox()
        self.txt_spectrogram_db_min_spin.setRange(-200.0, 200.0)
        self.txt_spectrogram_db_min_spin.setValue(self.default_txt_spectrogram_db_min)
        self.txt_spectrogram_db_min_spin.setSuffix(" dB")
        self.txt_spectrogram_db_min_spin.valueChanged.connect(self._txt_spectrogram_settings_changed)
        txt_spec_form.addRow("Color min", self.txt_spectrogram_db_min_spin)

        self.txt_spectrogram_db_max_spin = QtWidgets.QDoubleSpinBox()
        self.txt_spectrogram_db_max_spin.setRange(-200.0, 200.0)
        self.txt_spectrogram_db_max_spin.setValue(self.default_txt_spectrogram_db_max)
        self.txt_spectrogram_db_max_spin.setSuffix(" dB")
        self.txt_spectrogram_db_max_spin.valueChanged.connect(self._txt_spectrogram_settings_changed)
        txt_spec_form.addRow("Color max", self.txt_spectrogram_db_max_spin)

        self.txt_spectrogram_colormap_combo = QtWidgets.QComboBox()
        self.txt_spectrogram_colormap_combo.addItems(["viridis", "jet"])
        self.txt_spectrogram_colormap_combo.setCurrentText(self.default_txt_spectrogram_colormap)
        self.txt_spectrogram_colormap_combo.currentTextChanged.connect(
            self._txt_spectrogram_settings_changed
        )
        txt_spec_form.addRow("Color map", self.txt_spectrogram_colormap_combo)

        self.txt_spectrogram_fs_label = QtWidgets.QLabel("Analysis fs: -")
        self.txt_spectrogram_fs_label.setWordWrap(True)
        txt_spec_form.addRow(self.txt_spectrogram_fs_label)
        left_layout.addWidget(self.txt_spectrogram_section)

        self.txt_spectrogram_channels = ScrollChannelGroup(
            "TXT spectrogram ch", height=90, expanded=True
        )
        self.txt_spectrogram_channels.changed.connect(self.refresh_plot)
        left_layout.addWidget(self.txt_spectrogram_channels)

        self.epoc_group = ScrollCheckGroup("Epochs", height=90, expanded=False)
        self.epoc_group.changed.connect(self.refresh_plot)
        left_layout.addWidget(self.epoc_group)

        self.epoch_jump_section = CollapsibleSection("Epoch jump", expanded=False)
        jump_layout = QtWidgets.QVBoxLayout(self.epoch_jump_section.content)
        jump_layout.setContentsMargins(0, 0, 0, 0)
        jump_layout.setSpacing(4)

        jump_layout.addWidget(QtWidgets.QLabel("Epoch name"))
        self.epoch_jump_combo = QtWidgets.QComboBox()
        self.epoch_jump_combo.setFixedWidth(110)
        jump_layout.addWidget(self.epoch_jump_combo)

        jump_layout.addWidget(QtWidgets.QLabel("Tick index"))
        self.epoch_tick_spin = QtWidgets.QSpinBox()
        self.epoch_tick_spin.setRange(1, 1)
        self.epoch_tick_spin.setValue(1)
        self.epoch_tick_spin.setFixedWidth(90)
        jump_layout.addWidget(self.epoch_tick_spin)

        btn_row = QtWidgets.QHBoxLayout()
        self.epoch_prev_btn = QtWidgets.QPushButton("Prev")
        self.epoch_prev_btn.clicked.connect(self.jump_to_previous_epoch_tick)
        btn_row.addWidget(self.epoch_prev_btn)

        self.epoch_jump_btn = QtWidgets.QPushButton("Jump")
        self.epoch_jump_btn.clicked.connect(self.jump_to_selected_epoch_tick)
        btn_row.addWidget(self.epoch_jump_btn)

        self.epoch_next_btn = QtWidgets.QPushButton("Next")
        self.epoch_next_btn.clicked.connect(self.jump_to_next_epoch_tick)
        btn_row.addWidget(self.epoch_next_btn)

        jump_layout.addLayout(btn_row)

        self.epoch_tick_label = QtWidgets.QLabel("Time: - | Tick - / -")
        jump_layout.addWidget(self.epoch_tick_label)

        self.epoch_jump_combo.currentTextChanged.connect(self._update_epoch_tick_controls)
        self.epoch_tick_spin.valueChanged.connect(self._update_epoch_tick_label)

        left_layout.addWidget(self.epoch_jump_section)

        self.cursor_section = CollapsibleSection("Cursor", expanded=False)
        cursor_layout = QtWidgets.QVBoxLayout(self.cursor_section.content)
        cursor_layout.setContentsMargins(0, 0, 0, 0)
        cursor_layout.setSpacing(4)

        self.cursor_time_label = QtWidgets.QLabel("Cursor time: 0.000 s")
        self.cursor_time_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.cursor_time_label.setCursor(QtCore.Qt.IBeamCursor)
        cursor_layout.addWidget(self.cursor_time_label)

        self.show_cursor_check = QtWidgets.QCheckBox("Show cursor on traces")
        self.show_cursor_check.setChecked(self.show_cursor_on_traces)
        self.show_cursor_check.toggled.connect(self._show_cursor_toggled)
        cursor_layout.addWidget(self.show_cursor_check)

        self.jump_cursor_btn = QtWidgets.QPushButton("Jump to Cursor")
        self.jump_cursor_btn.setFixedWidth(120)
        self.jump_cursor_btn.clicked.connect(self.jump_to_cursor_time)
        cursor_layout.addWidget(self.jump_cursor_btn)

        self.copy_cursor_btn = QtWidgets.QPushButton("Copy Cursor Time")
        self.copy_cursor_btn.setFixedWidth(140)
        self.copy_cursor_btn.clicked.connect(self.copy_cursor_time)
        cursor_layout.addWidget(self.copy_cursor_btn)

        left_layout.addWidget(self.cursor_section)

        left_layout.addWidget(QtWidgets.QLabel("Window length"))
        self.window_spin = QtWidgets.QDoubleSpinBox()
        self.window_spin.setRange(0.1, 600.0)
        self.window_spin.setDecimals(2)
        self.window_spin.setValue(self.window_sec)
        self.window_spin.setSuffix(" s")
        self.window_spin.setMaximumWidth(150)
        self.window_spin.valueChanged.connect(self._window_changed)
        left_layout.addWidget(self.window_spin)

        left_layout.addWidget(QtWidgets.QLabel("TXT sampling rate"))
        self.txt_fs_spin = QtWidgets.QDoubleSpinBox()
        self.txt_fs_spin.setRange(0.001, 1000000.0)
        self.txt_fs_spin.setDecimals(3)
        self.txt_fs_spin.setValue(self.text_trace_fs)
        self.txt_fs_spin.setSuffix(" Hz")
        self.txt_fs_spin.setMaximumWidth(150)
        self.txt_fs_spin.valueChanged.connect(self._txt_fs_changed)
        left_layout.addWidget(self.txt_fs_spin)

        left_layout.addWidget(QtWidgets.QLabel("LFP_ amplitude"))
        self.lfp_amp_spin = QtWidgets.QDoubleSpinBox()
        self.lfp_amp_spin.setRange(0.001, 100000.0)
        self.lfp_amp_spin.setDecimals(3)
        self.lfp_amp_spin.setValue(self.default_lfp_amp)
        self.lfp_amp_spin.setMaximumWidth(150)
        self.lfp_amp_spin.valueChanged.connect(self._lfp_amp_changed)
        left_layout.addWidget(self.lfp_amp_spin)

        left_layout.addWidget(QtWidgets.QLabel("MUs_ amplitude"))
        self.mus_amp_spin = QtWidgets.QDoubleSpinBox()
        self.mus_amp_spin.setRange(0.001, 100000.0)
        self.mus_amp_spin.setDecimals(3)
        self.mus_amp_spin.setValue(self.default_mu_amp)
        self.mus_amp_spin.setMaximumWidth(150)
        self.mus_amp_spin.valueChanged.connect(self._mus_amp_changed)
        left_layout.addWidget(self.mus_amp_spin)

        left_layout.addWidget(QtWidgets.QLabel("TXT amplitude"))
        self.txt_amp_spin = QtWidgets.QDoubleSpinBox()
        self.txt_amp_spin.setRange(0.001, 1000000.0)
        self.txt_amp_spin.setDecimals(6)
        self.txt_amp_spin.setValue(self.default_txt_amp)
        self.txt_amp_spin.setMaximumWidth(150)
        self.txt_amp_spin.valueChanged.connect(self._txt_amp_changed)
        left_layout.addWidget(self.txt_amp_spin)

        left_layout.addWidget(QtWidgets.QLabel("Signal gain (raw / gain)"))
        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setRange(0.000001, 1000000000.0)
        self.gain_spin.setDecimals(6)
        self.gain_spin.setValue(self.signal_gain)
        self.gain_spin.setMaximumWidth(150)
        self.gain_spin.setToolTip(
            "LFP and MU raw samples are divided by gain before amplitude and scale calculations."
        )
        self.gain_spin.valueChanged.connect(self._gain_changed)
        left_layout.addWidget(self.gain_spin)

        left_layout.addWidget(QtWidgets.QLabel("Channel spacing"))
        self.spacing_spin = QtWidgets.QDoubleSpinBox()
        self.spacing_spin.setRange(0.001, 100000.0)
        self.spacing_spin.setDecimals(3)
        self.spacing_spin.setValue(self.channel_spacing)
        self.spacing_spin.setMaximumWidth(150)
        self.spacing_spin.valueChanged.connect(self._spacing_changed)
        left_layout.addWidget(self.spacing_spin)

        self.theme_check = QtWidgets.QCheckBox("White/black")
        self.theme_check.setChecked(self.white_mode)
        self.theme_check.toggled.connect(self._theme_changed)
        left_layout.addWidget(self.theme_check)

        self.scale_bar_check = QtWidgets.QCheckBox("Show scale bar")
        self.scale_bar_check.setChecked(self.show_scale_bar)
        self.scale_bar_check.toggled.connect(self._scale_bar_toggled)
        left_layout.addWidget(self.scale_bar_check)

        left_layout.addWidget(QtWidgets.QLabel("LFP_ scale bar"))
        self.lfp_bar_spin = QtWidgets.QDoubleSpinBox()
        self.lfp_bar_spin.setRange(0.000001, 1000000.0)
        self.lfp_bar_spin.setDecimals(3)
        self.lfp_bar_spin.setValue(self.scale_bar_value_lfp)
        self.lfp_bar_spin.setSuffix(" µV")
        self.lfp_bar_spin.setMaximumWidth(150)
        self.lfp_bar_spin.valueChanged.connect(self._lfp_bar_changed)
        left_layout.addWidget(self.lfp_bar_spin)

        left_layout.addWidget(QtWidgets.QLabel("MUs_ scale bar"))
        self.mus_bar_spin = QtWidgets.QDoubleSpinBox()
        self.mus_bar_spin.setRange(0.000001, 1000000.0)
        self.mus_bar_spin.setDecimals(3)
        self.mus_bar_spin.setValue(self.scale_bar_value_mus)
        self.mus_bar_spin.setSuffix(" µV")
        self.mus_bar_spin.setMaximumWidth(150)
        self.mus_bar_spin.valueChanged.connect(self._mus_bar_changed)
        left_layout.addWidget(self.mus_bar_spin)

        left_layout.addWidget(QtWidgets.QLabel("TXT scale bar (a.u.)"))
        self.txt_bar_spin = QtWidgets.QDoubleSpinBox()
        self.txt_bar_spin.setRange(0.000001, 1000000.0)
        self.txt_bar_spin.setDecimals(3)
        self.txt_bar_spin.setValue(self.scale_bar_value_txt)
        self.txt_bar_spin.setSuffix("")
        self.txt_bar_spin.setMaximumWidth(150)
        self.txt_bar_spin.valueChanged.connect(self._txt_bar_changed)
        left_layout.addWidget(self.txt_bar_spin)

        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_defaults)
        self.reset_btn.setMinimumHeight(32)
        self.reset_btn.setFixedWidth(100)
        left_layout.addWidget(self.reset_btn)

        left_layout.addStretch()

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        left_scroll.setMinimumWidth(170)
        left_scroll.setMaximumWidth(320)

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.plot_splitter.setHandleWidth(10)
        self.plot_splitter.setChildrenCollapsible(True)
        right_layout.addWidget(self.plot_splitter, 1)

        self.epoch_widget = CursorPlotWidget(axisItems={"left": ChannelLabelAxis("left")})
        self.text_widget = CursorPlotWidget(axisItems={"left": ChannelLabelAxis("left")})
        self.txt_spectrogram_widget = CursorPlotWidget(axisItems={"left": ChannelLabelAxis("left")})
        self.lfp_widget = CursorPlotWidget(axisItems={"left": ChannelLabelAxis("left")})
        self.spectrogram_widget = CursorPlotWidget(axisItems={"left": ChannelLabelAxis("left")})
        self.mu_widget = CursorPlotWidget(axisItems={"left": ChannelLabelAxis("left")})

        # Keep spectrogram panels easy to resize even when many channels are shown.
        self.spectrogram_widget.setMinimumHeight(50)
        self.txt_spectrogram_widget.setMinimumHeight(50)

        self.epoch_plot = self.epoch_widget.getPlotItem()
        self.text_plot = self.text_widget.getPlotItem()
        self.txt_spectrogram_plot = self.txt_spectrogram_widget.getPlotItem()
        self.lfp_plot = self.lfp_widget.getPlotItem()
        self.spectrogram_plot = self.spectrogram_widget.getPlotItem()
        self.mu_plot = self.mu_widget.getPlotItem()

        self.spectrogram_colorbar = pg.ColorBarItem(
            values=(self.default_spectrogram_db_min, self.default_spectrogram_db_max),
            width=18,
            colorMap=self._spectrogram_colormap(),
            label="dB re 1 µV²/Hz",
            interactive=False,
        )
        self.spectrogram_colorbar.setImageItem([], insert_in=self.spectrogram_plot)

        self.txt_spectrogram_colorbar = pg.ColorBarItem(
            values=(self.default_txt_spectrogram_db_min, self.default_txt_spectrogram_db_max),
            width=18,
            colorMap=self._txt_spectrogram_colormap(),
            label="dB re 1 unit²/Hz",
            interactive=False,
        )
        self.txt_spectrogram_colorbar.setImageItem([], insert_in=self.txt_spectrogram_plot)

        for pw in (
            self.epoch_widget,
            self.text_widget,
            self.txt_spectrogram_widget,
            self.lfp_widget,
            self.spectrogram_widget,
            self.mu_widget,
        ):
            pw.setMouseEnabled(x=False, y=False)
            pw.hideButtons()
            pw.setMenuEnabled(False)
            pw.doubleClicked.connect(self._on_plot_double_clicked)

        self.right_panel = right_widget
        self.right_panel.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.right_panel.customContextMenuRequested.connect(self._show_plot_context_menu)

        slider_row = QtWidgets.QHBoxLayout()

        self.show_settings_btn = QtWidgets.QPushButton("Hide")
        self.show_settings_btn.setFixedWidth(70)
        self.show_settings_btn.clicked.connect(self.toggle_left_panel)
        slider_row.addWidget(self.show_settings_btn)

        self.bottom_left_btn = QtWidgets.QPushButton("← 1/4W")
        self.bottom_left_btn.clicked.connect(self._step_left)
        self.bottom_left_btn.setFixedWidth(70)
        slider_row.addWidget(self.bottom_left_btn)

        self.bottom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bottom_slider.setRange(0, 1000)
        self.bottom_slider.valueChanged.connect(self._bottom_slider_changed)
        slider_row.addWidget(self.bottom_slider, 1)

        self.bottom_right_btn = QtWidgets.QPushButton("1/4W →")
        self.bottom_right_btn.clicked.connect(self._step_right)
        self.bottom_right_btn.setFixedWidth(70)
        slider_row.addWidget(self.bottom_right_btn)

        self.time_spin = QtWidgets.QDoubleSpinBox()
        self.time_spin.setDecimals(3)
        self.time_spin.setRange(0.0, 1.0)
        self.time_spin.setSingleStep(0.1)
        self.time_spin.setSuffix(" s")
        self.time_spin.valueChanged.connect(self._time_spin_changed)
        self.time_spin.setFixedWidth(140)
        slider_row.addWidget(self.time_spin)

        right_layout.addLayout(slider_row)

        self.left_panel = left_scroll
        self.main_splitter.addWidget(self.left_panel)
        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([220, 1400])

        self._rebuild_plot_splitter()

    def _create_cursor_lines(self):
        self.cursor_lines = []
        self.cursor_labels = {}

        plot_map = {
            "epoch": self.epoch_plot,
            "text": self.text_plot,
            "txt_spectrogram": self.txt_spectrogram_plot,
            "lfp": self.lfp_plot,
            "spectrogram": self.spectrogram_plot,
            "mu": self.mu_plot,
        }

        for key, plot in plot_map.items():
            line = pg.InfiniteLine(pos=self.cursor_time, angle=90, movable=False, pen=self.cursor_line_pen)
            line.sigPositionChanged.connect(self._on_cursor_moved)
            plot.addItem(line)
            self.cursor_lines.append(line)

            self.cursor_labels[key] = None

    def _rebuild_cursor_items_after_clear(self):
        self.cursor_lines = []
        self.cursor_labels = {}

        if not self.show_cursor_on_traces:
            return

        self._create_cursor_lines()
        for item in self.cursor_lines:
            item.setZValue(1000)
        for item in self.cursor_labels.values():
            if item is not None:
                item.setZValue(1001)
        self._set_cursor_time(self.cursor_time, update_label=True)
        self._update_cursor_plot_labels()

    def _on_plot_double_clicked(self, x):
        self._set_cursor_time(x, update_label=True)
        self._update_cursor_plot_labels()
        self.camera_dock.seek(x)

    def _toggle_camera(self, checked):
        self.camera_dock.setVisible(bool(checked) and self.camera_dock.has_camera)

    def _camera_visibility_changed(self, visible):
        self.camera_btn.blockSignals(True)
        self.camera_btn.setChecked(bool(visible))
        self.camera_btn.blockSignals(False)

    def _configure_camera(self):
        found = self.camera_dock.configure(
            self.block_path, self.epoc_onsets, self.epoc_values
        )
        self.camera_btn.setEnabled(found)
        self.camera_btn.setChecked(found)
        self.camera_dock.seek(self.current_time)

    def _camera_time_requested(self, seconds):
        self._set_cursor_time(seconds, update_label=True)
        if not self.show_cursor_on_traces:
            self.show_cursor_check.setChecked(True)
        if not (self.current_time <= seconds < self.current_time + self.window_sec):
            self.current_time = max(0.0, seconds - self.window_sec / 2.0)
            self._sync_time_widgets()
            self.refresh_plot()

    def _set_cursor_time(self, t, update_label=True):
        t = float(np.clip(t, 0.0, max(self.total_duration, 0.001)))
        self.cursor_time = t

        self._cursor_syncing = True
        try:
            for line in self.cursor_lines:
                line.setValue(t)
        finally:
            self._cursor_syncing = False

        if update_label:
            self.cursor_time_label.setText(f"Cursor time: {t:.3f} s")

    def _on_cursor_moved(self, line):
        if self._cursor_syncing:
            return
        t = float(line.value())
        self._set_cursor_time(t, update_label=True)
        self._update_cursor_plot_labels()

    def jump_to_cursor_time(self):
        self.current_time = float(self.cursor_time)
        self._sync_time_widgets()
        self.refresh_plot()

    def copy_cursor_time(self):
        QtWidgets.QApplication.clipboard().setText(f"{self.cursor_time:.6f}")
        self.statusBar().showMessage(f"Copied cursor time: {self.cursor_time:.6f} s", 2000)

    def _update_cursor_plot_labels(self):
        return

    def _make_grid_pen(self):
        if self.white_mode:
            pen = pg.mkPen((180, 180, 180), width=1)
        else:
            pen = pg.mkPen((110, 110, 110), width=1)
        pen.setStyle(QtCore.Qt.DotLine)
        return pen

    def _get_shared_time_grid_positions(self, x_left, x_right, n_lines=6):
        if not self.show_grid or x_right <= x_left:
            return []
        return np.linspace(x_left, x_right, n_lines).tolist()

    def _add_shared_x_grid_lines(self, plot, x_positions, y_min, y_max):
        if not self.show_grid:
            return
        pen = self._make_grid_pen()
        for x in x_positions:
            plot.addItem(pg.PlotCurveItem([x, x], [y_min, y_max], pen=pen))

    def _add_shared_y_grid_lines(self, plot, y_positions, x_left, x_right):
        if not self.show_grid:
            return
        pen = self._make_grid_pen()
        for y in y_positions:
            plot.addItem(pg.PlotCurveItem([x_left, x_right], [y, y], pen=pen))

    def _update_xaxis_visibility(self):
        self.epoch_plot.showAxis("bottom", False)

        if self.xaxis_mode == self.XAXIS_ALL:
            for plot in (self.text_plot, self.txt_spectrogram_plot, self.lfp_plot, self.spectrogram_plot, self.mu_plot):
                plot.showAxis("bottom", True)
                plot.setLabel("bottom", "Time", units="s", color=self.label_color if hasattr(self, "label_color") else "k")
        else:
            for plot in (self.text_plot, self.txt_spectrogram_plot, self.lfp_plot, self.spectrogram_plot, self.mu_plot):
                plot.showAxis("bottom", False)
            if self.txt_position == self.TXT_POSITION_BOTTOM:
                bottom_plot = (
                    self.txt_spectrogram_plot if self.show_txt_spectrogram else self.text_plot
                )
            else:
                bottom_plot = self.mu_plot
            bottom_plot.showAxis("bottom", True)
            bottom_plot.setLabel("bottom", "Time", units="s", color=self.label_color if hasattr(self, "label_color") else "k")

    def _populate_epoch_jump_combo(self):
        names = sorted(self.epoc_names)

        self.epoch_jump_combo.blockSignals(True)
        self.epoch_jump_combo.clear()
        self.epoch_jump_combo.addItems(names)

        saved_epoch_name = self.settings.value("epoch_jump_name", "", type=str)

        if saved_epoch_name and saved_epoch_name in names:
            self.epoch_jump_combo.setCurrentText(saved_epoch_name)
        elif self.default_epoc_name in names:
            self.epoch_jump_combo.setCurrentText(self.default_epoc_name)
        elif names:
            self.epoch_jump_combo.setCurrentIndex(0)

        self.epoch_jump_combo.blockSignals(False)
        self._update_epoch_tick_controls()

        saved_tick = int(self.settings.value("epoch_tick_value", 1))
        self.epoch_tick_spin.blockSignals(True)
        self.epoch_tick_spin.setValue(min(max(1, saved_tick), self.epoch_tick_spin.maximum()))
        self.epoch_tick_spin.blockSignals(False)
        self._update_epoch_tick_label()

    def _update_epoch_tick_controls(self):
        ep_name = self.epoch_jump_combo.currentText()
        if not ep_name or ep_name not in self.epoc_onsets:
            self.epoch_tick_spin.blockSignals(True)
            self.epoch_tick_spin.setRange(1, 1)
            self.epoch_tick_spin.setValue(1)
            self.epoch_tick_spin.blockSignals(False)
            self.epoch_tick_label.setText("Time: - | Tick - / -")
            return

        arr = self.epoc_onsets[ep_name]
        n = max(1, int(arr.size))
        current_val = min(max(1, self.epoch_tick_spin.value()), n)

        self.epoch_tick_spin.blockSignals(True)
        self.epoch_tick_spin.setRange(1, n)
        self.epoch_tick_spin.setValue(current_val)
        self.epoch_tick_spin.blockSignals(False)

        self._update_epoch_tick_label()

    def _update_epoch_tick_label(self):
        ep_name = self.epoch_jump_combo.currentText()
        if not ep_name or ep_name not in self.epoc_onsets:
            self.epoch_tick_label.setText("Time: - | Tick - / -")
            return

        arr = self.epoc_onsets[ep_name]
        if arr is None or arr.size == 0:
            self.epoch_tick_label.setText("Time: - | Tick - / -")
            return

        idx = self.epoch_tick_spin.value() - 1
        total = int(arr.size)

        if idx < 0 or idx >= arr.size:
            self.epoch_tick_label.setText(f"Time: - | Tick {idx + 1} / {total}")
            return

        self.epoch_tick_label.setText(f"Time: {float(arr[idx]):.3f} s | Tick {idx + 1} / {total}")

    def jump_to_selected_epoch_tick(self):
        ep_name = self.epoch_jump_combo.currentText()
        if not ep_name or ep_name not in self.epoc_onsets:
            QtWidgets.QMessageBox.information(self, "Epoch Jump", "No epoch selected.")
            return

        arr = self.epoc_onsets[ep_name]
        if arr is None or arr.size == 0:
            QtWidgets.QMessageBox.information(self, "Epoch Jump", "Selected epoch has no ticks.")
            return

        idx = min(max(0, self.epoch_tick_spin.value() - 1), arr.size - 1)
        self.current_time = float(arr[idx])
        self._sync_time_widgets()
        self.refresh_plot()

    def jump_to_previous_epoch_tick(self):
        ep_name = self.epoch_jump_combo.currentText()
        if not ep_name or ep_name not in self.epoc_onsets:
            return

        arr = self.epoc_onsets[ep_name]
        if arr is None or arr.size == 0:
            return

        self.epoch_tick_spin.setValue(max(1, self.epoch_tick_spin.value() - 1))
        self.jump_to_selected_epoch_tick()

    def jump_to_next_epoch_tick(self):
        ep_name = self.epoch_jump_combo.currentText()
        if not ep_name or ep_name not in self.epoc_onsets:
            return

        arr = self.epoc_onsets[ep_name]
        if arr is None or arr.size == 0:
            return

        self.epoch_tick_spin.setValue(min(arr.size, self.epoch_tick_spin.value() + 1))
        self.jump_to_selected_epoch_tick()

    def _clear_plot_splitter(self):
        while self.plot_splitter.count():
            w = self.plot_splitter.widget(0)
            if w is not None:
                w.setParent(None)

    def _get_panel_order(self):
        if self.txt_position == self.TXT_POSITION_TOP:
            return [self.epoch_widget, self.text_widget, self.txt_spectrogram_widget, self.lfp_widget, self.spectrogram_widget, self.mu_widget]
        if self.txt_position == self.TXT_POSITION_BOTTOM:
            return [self.epoch_widget, self.lfp_widget, self.spectrogram_widget, self.mu_widget, self.text_widget, self.txt_spectrogram_widget]
        return [self.epoch_widget, self.lfp_widget, self.spectrogram_widget, self.text_widget, self.txt_spectrogram_widget, self.mu_widget]

    def _rebuild_plot_splitter(self):
        current_sizes = self.plot_splitter.sizes()
        self._clear_plot_splitter()

        panel_order = self._get_panel_order()
        for w in panel_order:
            self.plot_splitter.addWidget(w)

        stretch_by_widget = {
            self.epoch_widget: 1,
            self.text_widget: 4,
            self.txt_spectrogram_widget: 4,
            self.lfp_widget: 6,
            self.spectrogram_widget: 4,
            self.mu_widget: 6,
        }
        for index, widget in enumerate(panel_order):
            self.plot_splitter.setStretchFactor(index, stretch_by_widget[widget])

        if len(current_sizes) == 6 and sum(current_sizes) > 0:
            self.plot_splitter.setSizes(current_sizes)
        else:
            if self.txt_position == self.TXT_POSITION_TOP:
                self.plot_splitter.setSizes([60, 220, 220, 360, 260, 360])
            elif self.txt_position == self.TXT_POSITION_BOTTOM:
                self.plot_splitter.setSizes([60, 360, 260, 360, 220, 220])
            else:
                self.plot_splitter.setSizes([60, 360, 260, 220, 220, 360])

        self.epoch_plot.setXLink(self.lfp_plot)
        self.text_plot.setXLink(self.lfp_plot)
        self.txt_spectrogram_plot.setXLink(self.lfp_plot)
        self.spectrogram_plot.setXLink(self.lfp_plot)
        self.mu_plot.setXLink(self.lfp_plot)

        self._update_xaxis_visibility()
        QtCore.QTimer.singleShot(0, self._sync_plot_margins)

    def _sync_plot_margins(self):
        plots = [self.epoch_plot, self.text_plot, self.txt_spectrogram_plot, self.lfp_plot, self.spectrogram_plot, self.mu_plot]
        target_left = 130
        for p in plots:
            p.getAxis("left").setWidth(target_left)
            p.getAxis("right").setWidth(10)
        self.spectrogram_plot.getAxis("right").setWidth(58)
        self.txt_spectrogram_plot.getAxis("right").setWidth(58)

    def _set_left_axis_labels(self, plot, ticks):
        axis = plot.getAxis("left")
        if hasattr(axis, "set_manual_labels"):
            axis.set_manual_labels(ticks)
        else:
            axis.setTicks([ticks])

    def _clear_left_axis_labels(self, plot):
        axis = plot.getAxis("left")
        if hasattr(axis, "clear_manual_labels"):
            axis.clear_manual_labels()
        else:
            axis.setTicks([[]])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._sync_plot_margins)

    def _show_plot_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)

        grid_action = QtGui.QAction("Show Grid", self)
        grid_action.setCheckable(True)
        grid_action.setChecked(self.show_grid)
        grid_action.triggered.connect(self._toggle_grid_from_menu)
        menu.addAction(grid_action)

        line_width_menu = menu.addMenu("Line Width")
        for width in [0.5, 1.0, 1.5, 2.0, 3.0]:
            act = QtGui.QAction(str(width), self)
            act.setCheckable(True)
            act.setChecked(abs(self.line_width - width) < 1e-9)
            act.triggered.connect(lambda checked=False, w=width: self._set_line_width(w))
            line_width_menu.addAction(act)

        menu.addSeparator()

        include_settings_action = QtGui.QAction("Save image includes settings panel", self)
        include_settings_action.setCheckable(True)
        include_settings_action.setChecked(self.export_include_settings)
        include_settings_action.triggered.connect(self._toggle_export_include_settings)
        menu.addAction(include_settings_action)

        export_action = QtGui.QAction("Export Plot as Image...", self)
        export_action.triggered.connect(self.export_plot_image)
        menu.addAction(export_action)

        export_pub_action = QtGui.QAction("Export Publication PNG...", self)
        export_pub_action.triggered.connect(self.export_publication_png)
        menu.addAction(export_pub_action)

        menu.exec(self.right_panel.mapToGlobal(pos))

    def _set_line_width(self, width: float):
        self.line_width = float(width)
        self.refresh_plot()

    def _toggle_grid_from_menu(self, checked):
        self.show_grid = bool(checked)
        self.refresh_plot()

    def _toggle_export_include_settings(self, checked):
        self.export_include_settings = bool(checked)
        mode = "full window with settings panel" if self.export_include_settings else "plot panel only"
        self.statusBar().showMessage(f"Image export mode: {mode}", 2000)

    def _image_export_widget(self):
        if self.export_include_settings:
            return self.centralWidget(), "window"
        return self.right_panel, "plot"

    def export_plot_image(self):
        source_widget, scope = self._image_export_widget()
        if not self.block_path:
            default_name = f"tdt_{scope}.png"
        else:
            base = os.path.basename(self.block_path.rstrip("/"))
            default_name = f"{base}_{scope}.png"

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Plot Image",
            default_name,
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)"
        )
        if not out_path:
            return

        try:
            pixmap = source_widget.grab()
            pixmap.save(out_path)
            self.statusBar().showMessage(f"Exported image: {out_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

    def export_publication_png(self):
        source_widget, scope = self._image_export_widget()
        if not self.block_path:
            default_name = f"tdt_{scope}_publication.png"
        else:
            base = os.path.basename(self.block_path.rstrip("/"))
            default_name = f"{base}_{scope}_publication.png"

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Publication PNG",
            default_name,
            "PNG Image (*.png)"
        )
        if not out_path:
            return

        if not out_path.lower().endswith(".png"):
            out_path += ".png"

        scale, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Publication PNG Scale",
            "Export scale multiplier:\n2 = high quality, 3–4 = publication/supplementary figure",
            3.0,
            1.0,
            6.0,
            1,
        )
        if not ok:
            return

        try:
            source_size = source_widget.size()
            if source_size.width() <= 0 or source_size.height() <= 0:
                raise ValueError("Export panel has invalid size.")

            target_size = QtCore.QSize(
                max(1, int(source_size.width() * scale)),
                max(1, int(source_size.height() * scale)),
            )

            image = QtGui.QImage(target_size, QtGui.QImage.Format_ARGB32)
            image.fill(QtCore.Qt.white if self.white_mode else QtCore.Qt.black)

            painter = QtGui.QPainter(image)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            painter.scale(scale, scale)
            source_widget.render(painter, QtCore.QPoint(0, 0))
            painter.end()

            image.setDotsPerMeterX(int(300 / 0.0254))
            image.setDotsPerMeterY(int(300 / 0.0254))

            if not image.save(out_path, "PNG"):
                raise ValueError("Could not save PNG image.")

            width_px = target_size.width()
            height_px = target_size.height()
            self.statusBar().showMessage(
                f"Exported publication PNG: {out_path} ({width_px} × {height_px} px, 300 DPI metadata)"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Publication Export Error", str(e))

    def toggle_left_panel(self):
        sizes = self.main_splitter.sizes()
        if sizes[0] > 0:
            self._saved_left_width = sizes[0]
            self.main_splitter.setSizes([0, max(1, sum(sizes))])
            self.show_settings_btn.setText("Show")
        else:
            left_width = getattr(self, "_saved_left_width", 220)
            total = max(sum(sizes), self.width())
            self.main_splitter.setSizes([left_width, max(300, total - left_width)])
            self.show_settings_btn.setText("Hide")

    def _auto_set_txt_amplitude(self):
        if self.text_trace_data is None or self.text_trace_data.size == 0:
            return

        try:
            arr = np.asarray(self.text_trace_data, dtype=float)
            finite_vals = arr[np.isfinite(arr)]
            if finite_vals.size == 0:
                return

            p5 = np.percentile(finite_vals, 5)
            p95 = np.percentile(finite_vals, 95)
            span = float(p95 - p5)

            if span <= 0:
                span = float(np.nanmax(np.abs(finite_vals)))
                if span <= 0:
                    span = 1.0

            target_span = max(self.channel_spacing * 0.6, 1e-9)
            amp = target_span / span
            amp = float(np.clip(amp, 1e-6, 1e6))

            self.stream_amp["TXT_"] = amp
            self.txt_amp_spin.blockSignals(True)
            self.txt_amp_spin.setValue(amp)
            self.txt_amp_spin.blockSignals(False)
        except Exception:
            pass

    def reset_defaults(self):
        self.window_sec = self.default_window_sec
        self.channel_spacing = self.default_channel_spacing
        self.white_mode = self.default_white_mode
        self.show_grid = self.default_show_grid
        self.show_scale_bar = self.default_show_scale_bar
        self.export_include_settings = False
        self.show_cursor_on_traces = False
        self.scale_bar_value_lfp = self.default_lfp_bar
        self.scale_bar_value_mus = self.default_mu_bar
        self.scale_bar_value_txt = self.default_txt_bar
        self.signal_gain = self.default_signal_gain
        self.line_width = self.default_line_width
        self.text_trace_fs = self.default_text_trace_fs
        self.txt_position = self.default_txt_position
        self.xaxis_mode = self.default_xaxis_mode

        self.window_spin.blockSignals(True)
        self.window_spin.setValue(self.window_sec)
        self.window_spin.blockSignals(False)

        self.spacing_spin.blockSignals(True)
        self.spacing_spin.setValue(self.channel_spacing)
        self.spacing_spin.blockSignals(False)

        self.theme_check.blockSignals(True)
        self.theme_check.setChecked(self.white_mode)
        self.theme_check.blockSignals(False)

        self.scale_bar_check.blockSignals(True)
        self.scale_bar_check.setChecked(self.show_scale_bar)
        self.scale_bar_check.blockSignals(False)

        self.show_cursor_check.blockSignals(True)
        self.show_cursor_check.setChecked(self.show_cursor_on_traces)
        self.show_cursor_check.blockSignals(False)

        self.lfp_amp_spin.blockSignals(True)
        self.lfp_amp_spin.setValue(self.default_lfp_amp)
        self.lfp_amp_spin.blockSignals(False)

        self.mus_amp_spin.blockSignals(True)
        self.mus_amp_spin.setValue(self.default_mu_amp)
        self.mus_amp_spin.blockSignals(False)

        self.txt_amp_spin.blockSignals(True)
        self.txt_amp_spin.setValue(self.default_txt_amp)
        self.txt_amp_spin.blockSignals(False)

        self.gain_spin.blockSignals(True)
        self.gain_spin.setValue(self.default_signal_gain)
        self.gain_spin.blockSignals(False)

        self.txt_fs_spin.blockSignals(True)
        self.txt_fs_spin.setValue(self.text_trace_fs)
        self.txt_fs_spin.blockSignals(False)

        self.txt_position_combo.blockSignals(True)
        self.txt_position_combo.setCurrentText(self.txt_position)
        self.txt_position_combo.blockSignals(False)

        self.xaxis_mode_combo.blockSignals(True)
        self.xaxis_mode_combo.setCurrentText(self.xaxis_mode)
        self.xaxis_mode_combo.blockSignals(False)

        self.lfp_bar_spin.blockSignals(True)
        self.lfp_bar_spin.setValue(self.default_lfp_bar)
        self.lfp_bar_spin.blockSignals(False)

        self.mus_bar_spin.blockSignals(True)
        self.mus_bar_spin.setValue(self.default_mu_bar)
        self.mus_bar_spin.blockSignals(False)

        self.txt_bar_spin.blockSignals(True)
        self.txt_bar_spin.setValue(self.default_txt_bar)
        self.txt_bar_spin.blockSignals(False)

        self.stream_amp["LFP_"] = self.default_lfp_amp
        self.stream_amp["MUs_"] = self.default_mu_amp
        self.stream_amp["TXT_"] = self.default_txt_amp

        if self.text_trace_data is not None:
            self._auto_set_txt_amplitude()

        self._rebuild_plot_splitter()
        self.total_duration = self._get_total_duration()
        self._sync_time_widgets()
        self._apply_plot_theme()
        self.refresh_plot()

    def choose_block(self):
        block_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select TDT Block Folder")
        if not block_path:
            return
        self.load_block(block_path)

    def choose_text_trace(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Text Trace File",
            "",
            "Text Files (*.txt *.csv);;All Files (*)"
        )
        if not file_path:
            return
        self.load_text_trace(file_path)

    def show_credits(self):
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("About / Credits")
        dialog.setIcon(QtWidgets.QMessageBox.Information)
        dialog.setTextFormat(QtCore.Qt.RichText)
        dialog.setText(
            f"<h3>{APP_NAME}</h3>"
            "<p><b>Version:</b> 5.0</p>"
            "<p><b>Application design and development:</b><br>PingChou</p>"
            "<p><b>Scientific and software components:</b><br>"
            "TDT Python SDK, Python, NumPy, SciPy, PySide6, pyqtgraph and OpenCV.</p>"
            "<p>This application is an independent analysis and visualization tool. "
            "TDT and NeuroExplorer are trademarks or product names of their respective owners.</p>"
        )
        dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
        dialog.exec()

    def save_session_dialog(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Viewer Session",
            "viewer_session.tdtv",
            "TDT Viewer Session (*.tdtv)"
        )
        if not file_path:
            return
        self.save_session(file_path)

    def open_session_dialog(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Viewer Session",
            "",
            "TDT Viewer Session (*.tdtv)"
        )
        if not file_path:
            return
        self.load_session(file_path)

    def save_session(self, file_path):
        try:
            streams_cache = {}
            if self.data is not None:
                for name in self.stream_names:
                    obj = self.data.streams[name]
                    streams_cache[name] = {
                        "data": np.asarray(obj.data),
                        "fs": float(getattr(obj, "fs", 1.0)),
                        "start_time": float(getattr(obj, "start_time", 0.0)),
                    }

            session = {
                "block_path": self.block_path,
                "text_trace_path": self.text_trace_path,
                "text_trace_data": self.text_trace_data,
                "text_trace_fs": self.text_trace_fs,
                "text_trace_start_time": self.text_trace_start_time,
                "stream_names": list(self.stream_names),
                "epoc_names": list(self.epoc_names),
                "epoc_onsets": self.epoc_onsets,
                "epoc_values": self.epoc_values,
                "streams_cache": streams_cache,
                "cursor_time": self.cursor_time,
                "ui": {
                    "current_time": self.current_time,
                    "window_sec": self.window_sec,
                    "channel_spacing": self.channel_spacing,
                    "white_mode": self.white_mode,
                    "show_grid": self.show_grid,
                    "show_scale_bar": self.show_scale_bar,
                    "export_include_settings": self.export_include_settings,
                    "show_cursor_on_traces": self.show_cursor_on_traces,
                    "line_width": self.line_width,
                    "txt_position": self.txt_position,
                    "xaxis_mode": self.xaxis_mode,
                    "lfp_amp": self.lfp_amp_spin.value(),
                    "mus_amp": self.mus_amp_spin.value(),
                    "txt_amp": self.txt_amp_spin.value(),
                    "signal_gain": self.gain_spin.value(),
                    "lfp_bar": self.lfp_bar_spin.value(),
                    "mus_bar": self.mus_bar_spin.value(),
                    "txt_bar": self.txt_bar_spin.value(),
                    "checked_streams": self._checked_stream_names(),
                    "lfp_channels": self.selected_channels_for_stream("LFP_"),
                    "mus_channels": self.selected_channels_for_stream("MUs_"),
                    "txt_channels": self.selected_channels_for_stream("TXT_"),
                    "show_spectrogram": self.spectrogram_show_check.isChecked(),
                    "spectrogram_power_mode": self.spectrogram_power_mode_combo.currentText(),
                    "spectrogram_detrend": self.spectrogram_detrend_combo.currentText(),
                    "spectrogram_gaussian_smoothing": self.spectrogram_gaussian_check.isChecked(),
                    "spectrogram_gaussian_width_bins": self.spectrogram_gaussian_width_spin.value(),
                    "spectrogram_channels": self.spectrogram_channels.selected_channels(),
                    "spectrogram_fmin": self.spectrogram_fmin_spin.value(),
                    "spectrogram_fmax": self.spectrogram_fmax_spin.value(),
                    "spectrogram_resolution_hz": self.spectrogram_resolution_spin.value(),
                    "spectrogram_window_sec": self.spectrogram_window_spin.value(),
                    "spectrogram_time_step_sec": self.spectrogram_time_step_spin.value(),
                    "spectrogram_overlap": self.spectrogram_overlap_spin.value(),
                    "spectrogram_db_min": self.spectrogram_db_min_spin.value(),
                    "spectrogram_db_max": self.spectrogram_db_max_spin.value(),
                    "spectrogram_colormap": self.spectrogram_colormap_combo.currentText(),
                    "spectrogram_expanded": self.spectrogram_section.is_expanded(),
                    "spectrogram_channels_expanded": self.spectrogram_channels.is_expanded(),
                    "show_txt_spectrogram": self.txt_spectrogram_show_check.isChecked(),
                    "txt_spectrogram_power_mode": self.txt_spectrogram_power_mode_combo.currentText(),
                    "txt_spectrogram_detrend": self.txt_spectrogram_detrend_combo.currentText(),
                    "txt_spectrogram_gaussian_smoothing": self.txt_spectrogram_gaussian_check.isChecked(),
                    "txt_spectrogram_gaussian_width_bins": self.txt_spectrogram_gaussian_width_spin.value(),
                    "txt_spectrogram_channels": self.txt_spectrogram_channels.selected_channels(),
                    "txt_spectrogram_fmin": self.txt_spectrogram_fmin_spin.value(),
                    "txt_spectrogram_fmax": self.txt_spectrogram_fmax_spin.value(),
                    "txt_spectrogram_resolution_hz": self.txt_spectrogram_resolution_spin.value(),
                    "txt_spectrogram_window_sec": self.txt_spectrogram_window_spin.value(),
                    "txt_spectrogram_time_step_sec": self.txt_spectrogram_time_step_spin.value(),
                    "txt_spectrogram_overlap": self.txt_spectrogram_overlap_spin.value(),
                    "txt_spectrogram_db_min": self.txt_spectrogram_db_min_spin.value(),
                    "txt_spectrogram_db_max": self.txt_spectrogram_db_max_spin.value(),
                    "txt_spectrogram_colormap": self.txt_spectrogram_colormap_combo.currentText(),
                    "txt_spectrogram_expanded": self.txt_spectrogram_section.is_expanded(),
                    "txt_spectrogram_channels_expanded": self.txt_spectrogram_channels.is_expanded(),
                    "selected_epocs": self.epoc_group.selected_items(),
                    "lfp_expanded": self.lfp_channels.is_expanded(),
                    "mu_expanded": self.mus_channels.is_expanded(),
                    "txt_expanded": self.txt_channels.is_expanded(),
                    "epoc_expanded": self.epoc_group.is_expanded(),
                    "block_expanded": self.block_section.is_expanded(),
                    "text_expanded": self.text_section.is_expanded(),
                    "stream_expanded": self.stream_section.is_expanded(),
                    "epoch_jump_expanded": self.epoch_jump_section.is_expanded(),
                    "cursor_expanded": self.cursor_section.is_expanded(),
                    "plot_splitter_sizes": self.plot_splitter.sizes(),
                    "epoch_jump_name": self.epoch_jump_combo.currentText(),
                    "epoch_tick_value": self.epoch_tick_spin.value(),
                }
            }

            with open(file_path, "wb") as f:
                pickle.dump(session, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.statusBar().showMessage(f"Session saved: {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save Session Error", str(e))

    def load_session(self, file_path):
        try:
            with open(file_path, "rb") as f:
                session = pickle.load(f)

            self.block_path = session.get("block_path", "")
            self.text_trace_path = session.get("text_trace_path", "")
            self.text_trace_data = session.get("text_trace_data", None)
            self.text_trace_fs = float(session.get("text_trace_fs", self.default_text_trace_fs))
            self.text_trace_start_time = float(session.get("text_trace_start_time", 0.0))

            self.stream_names = session.get("stream_names", [])
            self.epoc_names = session.get("epoc_names", [])
            self.epoc_onsets = session.get("epoc_onsets", {})
            self.epoc_values = session.get("epoc_values", {})

            streams_cache = session.get("streams_cache", {})
            self.data = CachedData()
            for name, info in streams_cache.items():
                self.data.streams[name] = CachedStream(
                    data=np.asarray(info["data"]),
                    fs=float(info["fs"]),
                    start_time=float(info["start_time"]),
                )

            self.block_label.setPlainText(f"Block:\n{self.block_path or '(session cached data)'}")
            self.text_trace_label.setPlainText(f"Text Trace:\n{self.text_trace_path or '(cached in session)'}")

            self.total_duration = self._get_total_duration()

            self._populate_streams()
            self._populate_channel_groups()
            self._populate_epocs()
            self._configure_camera()

            if self.text_trace_data is not None:
                n_channels = self.text_trace_data.shape[0]
                self.txt_channels.set_channels("Ch", n_channels, checked_n=min(8, n_channels), columns=2)
                self.txt_spectrogram_channels.set_channels(
                    "Ch", n_channels, checked_n=0, columns=2
                )

            ui = session.get("ui", {})

            self.window_sec = float(ui.get("window_sec", self.default_window_sec))
            self.channel_spacing = float(ui.get("channel_spacing", self.default_channel_spacing))
            self.white_mode = bool(ui.get("white_mode", self.default_white_mode))
            self.show_grid = bool(ui.get("show_grid", self.default_show_grid))
            self.show_scale_bar = bool(ui.get("show_scale_bar", self.default_show_scale_bar))
            self.export_include_settings = bool(ui.get("export_include_settings", False))
            self.show_cursor_on_traces = bool(ui.get("show_cursor_on_traces", False))
            self.line_width = float(ui.get("line_width", self.default_line_width))
            self.txt_position = ui.get("txt_position", self.default_txt_position)
            self.xaxis_mode = ui.get("xaxis_mode", self.default_xaxis_mode)

            self.window_spin.blockSignals(True)
            self.window_spin.setValue(self.window_sec)
            self.window_spin.blockSignals(False)

            self.spacing_spin.blockSignals(True)
            self.spacing_spin.setValue(self.channel_spacing)
            self.spacing_spin.blockSignals(False)

            self.theme_check.blockSignals(True)
            self.theme_check.setChecked(self.white_mode)
            self.theme_check.blockSignals(False)

            self.scale_bar_check.blockSignals(True)
            self.scale_bar_check.setChecked(self.show_scale_bar)
            self.scale_bar_check.blockSignals(False)

            self.show_cursor_check.blockSignals(True)
            self.show_cursor_check.setChecked(self.show_cursor_on_traces)
            self.show_cursor_check.blockSignals(False)

            self.txt_fs_spin.blockSignals(True)
            self.txt_fs_spin.setValue(self.text_trace_fs)
            self.txt_fs_spin.blockSignals(False)

            self.txt_position_combo.blockSignals(True)
            self.txt_position_combo.setCurrentText(self.txt_position)
            self.txt_position_combo.blockSignals(False)

            self.xaxis_mode_combo.blockSignals(True)
            self.xaxis_mode_combo.setCurrentText(self.xaxis_mode)
            self.xaxis_mode_combo.blockSignals(False)

            self.lfp_amp_spin.blockSignals(True)
            self.lfp_amp_spin.setValue(float(ui.get("lfp_amp", self.default_lfp_amp)))
            self.lfp_amp_spin.blockSignals(False)

            self.mus_amp_spin.blockSignals(True)
            self.mus_amp_spin.setValue(float(ui.get("mus_amp", self.default_mu_amp)))
            self.mus_amp_spin.blockSignals(False)

            self.txt_amp_spin.blockSignals(True)
            self.txt_amp_spin.setValue(float(ui.get("txt_amp", self.default_txt_amp)))
            self.txt_amp_spin.blockSignals(False)

            self.gain_spin.blockSignals(True)
            self.gain_spin.setValue(float(ui.get("signal_gain", self.default_signal_gain)))
            self.gain_spin.blockSignals(False)

            self.lfp_bar_spin.blockSignals(True)
            self.lfp_bar_spin.setValue(float(ui.get("lfp_bar", self.default_lfp_bar)))
            self.lfp_bar_spin.blockSignals(False)

            self.mus_bar_spin.blockSignals(True)
            self.mus_bar_spin.setValue(float(ui.get("mus_bar", self.default_mu_bar)))
            self.mus_bar_spin.blockSignals(False)

            self.txt_bar_spin.blockSignals(True)
            self.txt_bar_spin.setValue(float(ui.get("txt_bar", self.default_txt_bar)))
            self.txt_bar_spin.blockSignals(False)

            spec_values = (
                (self.spectrogram_show_check, "setChecked", bool(ui.get("show_spectrogram", self.default_show_spectrogram))),
                (self.spectrogram_power_mode_combo, "setCurrentText", str(ui.get("spectrogram_power_mode", self.default_spectrogram_power_mode))),
                (self.spectrogram_detrend_combo, "setCurrentText", str(ui.get("spectrogram_detrend", self.default_spectrogram_detrend))),
                (self.spectrogram_gaussian_check, "setChecked", bool(ui.get("spectrogram_gaussian_smoothing", self.default_spectrogram_gaussian_smoothing))),
                (self.spectrogram_gaussian_width_spin, "setValue", float(ui.get("spectrogram_gaussian_width_bins", self.default_spectrogram_gaussian_width_bins))),
                (self.spectrogram_fmin_spin, "setValue", float(ui.get("spectrogram_fmin", self.default_spectrogram_fmin))),
                (self.spectrogram_fmax_spin, "setValue", float(ui.get("spectrogram_fmax", self.default_spectrogram_fmax))),
                (self.spectrogram_resolution_spin, "setValue", float(ui.get("spectrogram_resolution_hz", self.default_spectrogram_resolution_hz))),
                (self.spectrogram_window_spin, "setValue", float(ui.get("spectrogram_window_sec", self.default_spectrogram_window_sec))),
                (self.spectrogram_time_step_spin, "setValue", float(ui.get("spectrogram_time_step_sec", self.default_spectrogram_time_step_sec))),
                (self.spectrogram_overlap_spin, "setValue", int(ui.get("spectrogram_overlap", self.default_spectrogram_overlap))),
                (self.spectrogram_db_min_spin, "setValue", float(ui.get("spectrogram_db_min", self.default_spectrogram_db_min))),
                (self.spectrogram_db_max_spin, "setValue", float(ui.get("spectrogram_db_max", self.default_spectrogram_db_max))),
                (self.spectrogram_colormap_combo, "setCurrentText", str(ui.get("spectrogram_colormap", self.default_spectrogram_colormap))),
            )
            for widget, setter, value in spec_values:
                widget.blockSignals(True)
                getattr(widget, setter)(value)
                widget.blockSignals(False)
            self.show_spectrogram = self.spectrogram_show_check.isChecked()
            self.spectrogram_widget.setVisible(self.show_spectrogram)
            is_percentage = self.spectrogram_power_mode_combo.currentText() == "Percentage power"
            suffix = " %" if is_percentage else " dB"
            self.spectrogram_db_min_spin.setSuffix(suffix)
            self.spectrogram_db_max_spin.setSuffix(suffix)
            self._update_gaussian_controls()
            saved_spec_channels = ui.get("spectrogram_channels", None)
            if saved_spec_channels is None and "spectrogram_channel" in ui:
                saved_spec_channels = [max(0, int(ui["spectrogram_channel"]) - 1)]
            self.spectrogram_channels.set_selected_channels(saved_spec_channels or [0])

            txt_spec_values = (
                (self.txt_spectrogram_show_check, "setChecked", bool(ui.get("show_txt_spectrogram", self.default_show_txt_spectrogram))),
                (self.txt_spectrogram_power_mode_combo, "setCurrentText", str(ui.get("txt_spectrogram_power_mode", self.default_txt_spectrogram_power_mode))),
                (self.txt_spectrogram_detrend_combo, "setCurrentText", str(ui.get("txt_spectrogram_detrend", self.default_txt_spectrogram_detrend))),
                (self.txt_spectrogram_gaussian_check, "setChecked", bool(ui.get("txt_spectrogram_gaussian_smoothing", self.default_txt_spectrogram_gaussian_smoothing))),
                (self.txt_spectrogram_gaussian_width_spin, "setValue", float(ui.get("txt_spectrogram_gaussian_width_bins", self.default_txt_spectrogram_gaussian_width_bins))),
                (self.txt_spectrogram_fmin_spin, "setValue", float(ui.get("txt_spectrogram_fmin", self.default_txt_spectrogram_fmin))),
                (self.txt_spectrogram_fmax_spin, "setValue", float(ui.get("txt_spectrogram_fmax", self.default_txt_spectrogram_fmax))),
                (self.txt_spectrogram_resolution_spin, "setValue", float(ui.get("txt_spectrogram_resolution_hz", self.default_txt_spectrogram_resolution_hz))),
                (self.txt_spectrogram_window_spin, "setValue", float(ui.get("txt_spectrogram_window_sec", self.default_txt_spectrogram_window_sec))),
                (self.txt_spectrogram_time_step_spin, "setValue", float(ui.get("txt_spectrogram_time_step_sec", self.default_txt_spectrogram_time_step_sec))),
                (self.txt_spectrogram_overlap_spin, "setValue", int(ui.get("txt_spectrogram_overlap", self.default_txt_spectrogram_overlap))),
                (self.txt_spectrogram_db_min_spin, "setValue", float(ui.get("txt_spectrogram_db_min", self.default_txt_spectrogram_db_min))),
                (self.txt_spectrogram_db_max_spin, "setValue", float(ui.get("txt_spectrogram_db_max", self.default_txt_spectrogram_db_max))),
                (self.txt_spectrogram_colormap_combo, "setCurrentText", str(ui.get("txt_spectrogram_colormap", self.default_txt_spectrogram_colormap))),
            )
            for widget, setter, value in txt_spec_values:
                widget.blockSignals(True)
                getattr(widget, setter)(value)
                widget.blockSignals(False)
            self.show_txt_spectrogram = self.txt_spectrogram_show_check.isChecked()
            self.txt_spectrogram_widget.setVisible(self.show_txt_spectrogram)
            is_txt_percentage = self.txt_spectrogram_power_mode_combo.currentText() == "Percentage power"
            txt_suffix = " %" if is_txt_percentage else " dB"
            self.txt_spectrogram_db_min_spin.setSuffix(txt_suffix)
            self.txt_spectrogram_db_max_spin.setSuffix(txt_suffix)
            self._update_txt_gaussian_controls()
            saved_txt_spec_channels = ui.get("txt_spectrogram_channels", None)
            self.txt_spectrogram_channels.set_selected_channels(
                saved_txt_spec_channels or self.txt_channels.selected_channels() or [0]
            )

            self.stream_amp["LFP_"] = self.lfp_amp_spin.value()
            self.stream_amp["MUs_"] = self.mus_amp_spin.value()
            self.stream_amp["TXT_"] = self.txt_amp_spin.value()
            self.signal_gain = max(self.gain_spin.value(), 1e-12)

            self.scale_bar_value_lfp = self.lfp_bar_spin.value()
            self.scale_bar_value_mus = self.mus_bar_spin.value()
            self.scale_bar_value_txt = self.txt_bar_spin.value()

            self.lfp_channels.set_selected_channels(ui.get("lfp_channels", []))
            self.mus_channels.set_selected_channels(ui.get("mus_channels", []))
            self.txt_channels.set_selected_channels(ui.get("txt_channels", []))
            self.epoc_group.set_selected_items(ui.get("selected_epocs", []))
            self._set_checked_stream_names(ui.get("checked_streams", []))

            self.lfp_channels.set_expanded(bool(ui.get("lfp_expanded", False)))
            self.mus_channels.set_expanded(bool(ui.get("mu_expanded", False)))
            self.txt_channels.set_expanded(bool(ui.get("txt_expanded", False)))
            self.epoc_group.set_expanded(bool(ui.get("epoc_expanded", False)))
            self.block_section.set_expanded(bool(ui.get("block_expanded", False)))
            self.text_section.set_expanded(bool(ui.get("text_expanded", False)))
            self.stream_section.set_expanded(bool(ui.get("stream_expanded", False)))
            self.epoch_jump_section.set_expanded(bool(ui.get("epoch_jump_expanded", False)))
            self.cursor_section.set_expanded(bool(ui.get("cursor_expanded", False)))
            self.spectrogram_section.set_expanded(bool(ui.get("spectrogram_expanded", True)))
            self.spectrogram_channels.set_expanded(bool(ui.get("spectrogram_channels_expanded", True)))
            self.txt_spectrogram_section.set_expanded(bool(ui.get("txt_spectrogram_expanded", True)))
            self.txt_spectrogram_channels.set_expanded(bool(ui.get("txt_spectrogram_channels_expanded", True)))

            self._rebuild_plot_splitter()
            self._apply_plot_theme()

            plot_sizes = ui.get("plot_splitter_sizes", None)
            if plot_sizes and len(plot_sizes) == 6:
                self.plot_splitter.setSizes([int(x) for x in plot_sizes])

            self.current_time = min(max(0.0, float(ui.get("current_time", 0.0))), self.total_duration)
            self.cursor_time = float(session.get("cursor_time", self.current_time))

            saved_epoch_name = ui.get("epoch_jump_name", "")
            saved_epoch_tick = int(ui.get("epoch_tick_value", 1))
            if saved_epoch_name and saved_epoch_name in self.epoc_names:
                self.epoch_jump_combo.setCurrentText(saved_epoch_name)
            self._update_epoch_tick_controls()
            self.epoch_tick_spin.setValue(min(max(1, saved_epoch_tick), self.epoch_tick_spin.maximum()))
            self._update_epoch_tick_label()

            self._sync_time_widgets()
            self.refresh_plot()

            short_name = os.path.basename(file_path)
            self.setWindowTitle(f"{APP_NAME} - {short_name}")
            self.statusBar().showMessage(f"Session loaded: {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open Session Error", str(e))

    def load_text_trace(self, file_path):
        try:
            data = self._parse_text_trace(file_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Text Trace Load Error", str(e))
            return

        self.text_trace_path = file_path
        self.text_trace_data = data
        self.text_trace_label.setPlainText(f"Text Trace:\n{self.text_trace_path}")

        n_channels = data.shape[0]
        self.txt_channels.set_channels("Ch", n_channels, checked_n=min(8, n_channels), columns=2)
        self.txt_spectrogram_channels.set_channels("Ch", n_channels, checked_n=0, columns=2)
        self.txt_spectrogram_channels.set_selected_channels(
            self.txt_channels.selected_channels()
        )

        self._auto_set_txt_amplitude()

        self.total_duration = self._get_total_duration()
        self._sync_time_widgets()
        self.refresh_plot()
        self.statusBar().showMessage(f"Loaded text trace: {file_path}")

    def _parse_text_trace(self, file_path):
        try:
            arr = np.loadtxt(file_path, delimiter=",")
        except Exception:
            try:
                arr = np.loadtxt(file_path)
            except Exception as e:
                raise ValueError(f"Cannot parse text trace file:\n{e}")

        arr = np.asarray(arr, dtype=float)

        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim == 2:
            arr = arr.T
        else:
            raise ValueError("Unsupported text trace shape.")

        if arr.shape[1] == 0:
            raise ValueError("Text trace file is empty.")

        return arr

    def load_block(self, block_path, restore_selection=False):
        try:
            data = tdt.read_block(block_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))
            return

        streams = list(data.streams.keys()) if hasattr(data, "streams") and data.streams is not None else []
        if not streams:
            QtWidgets.QMessageBox.critical(self, "Load Error", "No streams found in this block.")
            return

        self.block_path = block_path
        self.data = data
        self.stream_names = streams
        self.stream_amp = {name: self.stream_amp.get(name, 1.0) for name in self.stream_names}
        self.stream_amp["TXT_"] = self.stream_amp.get("TXT_", self.default_txt_amp)

        self.epoc_names = []
        self.epoc_onsets = {}
        self.epoc_values = {}
        if hasattr(data, "epocs") and data.epocs is not None:
            for name, obj in data.epocs.items():
                onset = getattr(obj, "onset", None)
                if onset is None:
                    continue
                arr = np.asarray(onset, dtype=float).ravel()
                if arr.size == 0:
                    continue
                self.epoc_names.append(name)
                self.epoc_onsets[name] = arr
                if name.lower().startswith("cam"):
                    try:
                        values = np.asarray(getattr(obj, "data", []), dtype=float).ravel()
                    except (TypeError, ValueError):
                        values = np.array([], dtype=float)
                    if values.size == arr.size:
                        self.epoc_values[name] = values

        self.total_duration = self._get_total_duration()
        self.current_time = 0.0
        self.cursor_time = 0.0
        self.block_label.setPlainText(f"Block:\n{self.block_path}")

        self._populate_streams()
        self._populate_channel_groups()
        self._populate_epocs()
        self._configure_camera()

        if restore_selection:
            self.restore_channel_selection()
        else:
            self._sync_time_widgets()
            self.refresh_plot()

        short_name = self.block_path.rstrip("/").split("/")[-1]
        self.setWindowTitle(f"{APP_NAME} - {short_name}")
        self.statusBar().showMessage(f"Loaded block: {self.block_path}")

    def _populate_epocs(self):
        self.epoc_group.clear_checks()

        if self.epoc_names:
            names = sorted(self.epoc_names)
            self.epoc_group.set_items(names, checked_first_n=0)

            if self.default_epoc_name in names:
                self.epoc_group.set_selected_items([self.default_epoc_name])
            else:
                self.epoc_group.set_selected_items(names[:1])

        self._populate_epoch_jump_combo()

    def _get_total_duration(self):
        max_end = 0.0

        if self.data is not None:
            for name in self.stream_names:
                obj = self.data.streams[name]
                raw = np.asarray(obj.data)
                if raw.ndim == 1:
                    raw = raw[np.newaxis, :]
                fs = float(getattr(obj, "fs", 1.0))
                start_time = float(getattr(obj, "start_time", 0.0))
                dur = raw.shape[1] / fs if fs > 0 else 0.0
                max_end = max(max_end, start_time + dur)

        if self.epoc_onsets:
            for arr in self.epoc_onsets.values():
                if arr.size:
                    max_end = max(max_end, float(np.nanmax(arr)))

        if self.text_trace_data is not None:
            txt_dur = self.text_trace_data.shape[1] / float(self.text_trace_fs)
            max_end = max(max_end, self.text_trace_start_time + txt_dur)

        return max(max_end, 0.001)

    def _populate_streams(self):
        self.stream_list.blockSignals(True)
        self.stream_list.clear()
        for name in self.stream_names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.stream_list.addItem(item)
        self.stream_list.blockSignals(False)

        max_ms = int(self.total_duration * 1000)
        self.bottom_slider.setRange(0, max_ms)
        self.time_spin.setRange(0.0, self.total_duration)

    def _populate_channel_groups(self):
        self.lfp_channels.clear_checks()
        self.mus_channels.clear_checks()
        self.txt_channels.clear_checks()
        self.spectrogram_channels.clear_checks()
        self.txt_spectrogram_channels.clear_checks()

        if self.data is not None and "LFP_" in self.stream_names:
            raw = np.asarray(self.data.streams["LFP_"].data)
            if raw.ndim == 1:
                raw = raw[np.newaxis, :]
            self.lfp_channels.set_channels("Ch", raw.shape[0], checked_n=8, columns=2)
            self.spectrogram_channels.set_channels("Ch", raw.shape[0], checked_n=0, columns=2)
            self.spectrogram_channels.set_selected_channels(
                self.lfp_channels.selected_channels()
            )

        if self.data is not None and "MUs_" in self.stream_names:
            raw = np.asarray(self.data.streams["MUs_"].data)
            if raw.ndim == 1:
                raw = raw[np.newaxis, :]
            self.mus_channels.set_channels("Ch", raw.shape[0], checked_n=8, columns=2)

        if self.text_trace_data is not None:
            self.txt_channels.set_channels("Ch", self.text_trace_data.shape[0], checked_n=min(8, self.text_trace_data.shape[0]), columns=2)
            self.txt_spectrogram_channels.set_channels(
                "Ch", self.text_trace_data.shape[0], checked_n=0, columns=2
            )
            self.txt_spectrogram_channels.set_selected_channels(
                self.txt_channels.selected_channels()
            )

    def selected_streams(self):
        out = []
        for i in range(self.stream_list.count()):
            item = self.stream_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                out.append(item.text())
        return out

    def selected_channels_for_stream(self, stream_name):
        if stream_name == "LFP_":
            return self.lfp_channels.selected_channels()
        if stream_name == "MUs_":
            return self.mus_channels.selected_channels()
        if stream_name == "TXT_":
            return self.txt_channels.selected_channels()
        return []

    def _checked_stream_names(self):
        return self.selected_streams()

    def _set_checked_stream_names(self, names):
        names = set(names or [])
        self.stream_list.blockSignals(True)
        for i in range(self.stream_list.count()):
            item = self.stream_list.item(i)
            item.setCheckState(QtCore.Qt.Checked if item.text() in names else QtCore.Qt.Unchecked)
        self.stream_list.blockSignals(False)

    def _sync_time_widgets(self):
        self.current_time = min(max(0.0, self.current_time), self.total_duration)

        self.bottom_slider.blockSignals(True)
        self.time_spin.blockSignals(True)

        self.bottom_slider.setValue(int(self.current_time * 1000))
        self.time_spin.setValue(self.current_time)

        self.bottom_slider.blockSignals(False)
        self.time_spin.blockSignals(False)

    def _bottom_slider_changed(self, value):
        self.current_time = value / 1000.0
        self._sync_time_widgets()
        self.refresh_plot()

    def _time_spin_changed(self, value):
        self.current_time = float(value)
        self._sync_time_widgets()
        self.refresh_plot()

    def _step_left(self):
        step = self.window_sec / 4.0
        self.current_time = max(0.0, self.current_time - step)
        self._sync_time_widgets()
        self.refresh_plot()

    def _step_right(self):
        step = self.window_sec / 4.0
        self.current_time = min(self.total_duration, self.current_time + step)
        self._sync_time_widgets()
        self.refresh_plot()

    def _window_changed(self, value):
        self.window_sec = float(value)
        self.refresh_plot()

    def _txt_fs_changed(self, value):
        self.text_trace_fs = float(value)
        self.total_duration = self._get_total_duration()
        self._sync_time_widgets()
        self.refresh_plot()

    def _txt_position_changed(self, text):
        self.txt_position = text
        self._rebuild_plot_splitter()
        self.refresh_plot()

    def _spectrogram_settings_changed(self, *args):
        self.show_spectrogram = bool(self.spectrogram_show_check.isChecked())
        self.spectrogram_widget.setVisible(self.show_spectrogram)
        self.refresh_plot()

    def _update_gaussian_controls(self):
        self.spectrogram_gaussian_width_spin.setEnabled(
            self.spectrogram_gaussian_check.isChecked()
        )

    def _spectrogram_gaussian_toggled(self, checked):
        self._update_gaussian_controls()
        self.refresh_plot()

    def _spectrogram_power_mode_changed(self, mode):
        is_percentage = mode == "Percentage power"
        suffix = " %" if is_percentage else " dB"
        self.spectrogram_db_min_spin.setSuffix(suffix)
        self.spectrogram_db_max_spin.setSuffix(suffix)
        color_min = (
            self.default_spectrogram_percentage_min
            if is_percentage else self.default_spectrogram_db_min
        )
        color_max = (
            self.default_spectrogram_percentage_max
            if is_percentage else self.default_spectrogram_db_max
        )
        self.spectrogram_db_min_spin.blockSignals(True)
        self.spectrogram_db_max_spin.blockSignals(True)
        self.spectrogram_db_min_spin.setValue(color_min)
        self.spectrogram_db_max_spin.setValue(color_max)
        self.spectrogram_db_min_spin.blockSignals(False)
        self.spectrogram_db_max_spin.blockSignals(False)
        self.refresh_plot()

    def _txt_spectrogram_settings_changed(self, *args):
        self.show_txt_spectrogram = bool(self.txt_spectrogram_show_check.isChecked())
        self.txt_spectrogram_widget.setVisible(self.show_txt_spectrogram)
        self._update_xaxis_visibility()
        self.refresh_plot()

    def _update_txt_gaussian_controls(self):
        self.txt_spectrogram_gaussian_width_spin.setEnabled(
            self.txt_spectrogram_gaussian_check.isChecked()
        )

    def _txt_spectrogram_gaussian_toggled(self, checked):
        self._update_txt_gaussian_controls()
        self.refresh_plot()

    def _txt_spectrogram_power_mode_changed(self, mode):
        is_percentage = mode == "Percentage power"
        suffix = " %" if is_percentage else " dB"
        self.txt_spectrogram_db_min_spin.setSuffix(suffix)
        self.txt_spectrogram_db_max_spin.setSuffix(suffix)
        color_min = (
            self.default_txt_spectrogram_percentage_min
            if is_percentage else self.default_txt_spectrogram_db_min
        )
        color_max = (
            self.default_txt_spectrogram_percentage_max
            if is_percentage else self.default_txt_spectrogram_db_max
        )
        self.txt_spectrogram_db_min_spin.blockSignals(True)
        self.txt_spectrogram_db_max_spin.blockSignals(True)
        self.txt_spectrogram_db_min_spin.setValue(color_min)
        self.txt_spectrogram_db_max_spin.setValue(color_max)
        self.txt_spectrogram_db_min_spin.blockSignals(False)
        self.txt_spectrogram_db_max_spin.blockSignals(False)
        self.refresh_plot()

    def _xaxis_mode_changed(self, text):
        self.xaxis_mode = text
        self._update_xaxis_visibility()
        self.refresh_plot()

    def _lfp_amp_changed(self, value):
        self.stream_amp["LFP_"] = float(value)
        self.refresh_plot()

    def _mus_amp_changed(self, value):
        self.stream_amp["MUs_"] = float(value)
        self.refresh_plot()

    def _txt_amp_changed(self, value):
        self.stream_amp["TXT_"] = float(value)
        self.refresh_plot()

    def _gain_changed(self, value):
        self.signal_gain = max(float(value), 1e-12)
        self.refresh_plot()

    def _spacing_changed(self, value):
        self.channel_spacing = float(value)
        if self.text_trace_data is not None:
            self._auto_set_txt_amplitude()
        self.refresh_plot()

    def _theme_changed(self, checked):
        self.white_mode = bool(checked)
        self._apply_plot_theme()
        self.refresh_plot()

    def _scale_bar_toggled(self, checked):
        self.show_scale_bar = bool(checked)
        self.refresh_plot()

    def _show_cursor_toggled(self, checked):
        self.show_cursor_on_traces = bool(checked)
        self.refresh_plot()

    def _lfp_bar_changed(self, value):
        self.scale_bar_value_lfp = float(value)
        self.refresh_plot()

    def _mus_bar_changed(self, value):
        self.scale_bar_value_mus = float(value)
        self.refresh_plot()

    def _txt_bar_changed(self, value):
        self.scale_bar_value_txt = float(value)
        self.refresh_plot()

    def _apply_plot_theme(self):
        if self.white_mode:
            bg = "w"
            axis_pen = pg.mkPen("k")
            trace_pen = pg.mkPen("k")
            epoch_pen = pg.mkPen((120, 120, 120), width=1)
            label_color = "k"
        else:
            bg = "k"
            axis_pen = pg.mkPen("w")
            trace_pen = pg.mkPen("w")
            epoch_pen = pg.mkPen((180, 180, 180), width=1)
            label_color = "w"

        self.trace_pen = trace_pen
        self.epoch_pen = epoch_pen
        self.label_color = label_color

        for pw in (self.epoch_widget, self.text_widget, self.txt_spectrogram_widget, self.lfp_widget, self.spectrogram_widget, self.mu_widget):
            pw.setBackground(bg)

        for plot in (self.epoch_plot, self.text_plot, self.txt_spectrogram_plot, self.lfp_plot, self.spectrogram_plot, self.mu_plot):
            plot.showGrid(x=False, y=False)
            plot.getAxis("left").setPen(axis_pen)
            plot.getAxis("left").setTextPen(axis_pen)
            plot.getAxis("bottom").setPen(axis_pen)
            plot.getAxis("bottom").setTextPen(axis_pen)

        self.spectrogram_plot.showAxis("right", True)
        self.spectrogram_plot.getAxis("right").setPen(axis_pen)
        self.spectrogram_plot.getAxis("right").setTextPen(axis_pen)
        self.spectrogram_plot.getAxis("right").setLabel("Frequency", units="Hz", color=label_color)
        frequency_tick_font = QtGui.QFont()
        frequency_tick_font.setPointSize(6)
        self.spectrogram_plot.getAxis("right").setStyle(
            tickFont=frequency_tick_font,
            tickTextOffset=2,
        )
        self.spectrogram_colorbar.axis.setPen(axis_pen)
        self.spectrogram_colorbar.axis.setTextPen(axis_pen)

        self.txt_spectrogram_plot.showAxis("right", True)
        self.txt_spectrogram_plot.getAxis("right").setPen(axis_pen)
        self.txt_spectrogram_plot.getAxis("right").setTextPen(axis_pen)
        self.txt_spectrogram_plot.getAxis("right").setLabel("Frequency", units="Hz", color=label_color)
        self.txt_spectrogram_plot.getAxis("right").setStyle(
            tickFont=frequency_tick_font,
            tickTextOffset=2,
        )
        self.txt_spectrogram_colorbar.axis.setPen(axis_pen)
        self.txt_spectrogram_colorbar.axis.setTextPen(axis_pen)

        self._update_xaxis_visibility()

        self.epoch_plot.getAxis("left").setStyle(showValues=True)
        self.text_plot.getAxis("left").setStyle(showValues=True)
        self.txt_spectrogram_plot.getAxis("left").setStyle(showValues=True)
        self.lfp_plot.getAxis("left").setStyle(showValues=True)
        self.spectrogram_plot.getAxis("left").setStyle(showValues=True)
        self.mu_plot.getAxis("left").setStyle(showValues=True)
        self.spectrogram_plot.setLabel("left", "Channel", color=label_color)
        self.txt_spectrogram_plot.setLabel("left", "TXT Channel", color=label_color)

    def _add_scale_bar_outside(self, plot, x_left, x_right, y_min, y_max, sname):
        if not self.show_scale_bar:
            return

        amp = self.stream_amp.get(sname, 1.0)

        if sname == "LFP_":
            bar_value = self.scale_bar_value_lfp
            bar_label = f"{bar_value:g} µV"
            bar_height = bar_value * 1e-6 * amp
        elif sname == "MUs_":
            bar_value = self.scale_bar_value_mus
            bar_label = f"{bar_value:g} µV"
            bar_height = bar_value * 1e-6 * amp
        elif sname == "TXT_":
            bar_value = self.scale_bar_value_txt
            bar_label = f"{bar_value:g}"
            bar_height = bar_value * amp
        else:
            return

        x_span = x_right - x_left
        y_span = y_max - y_min
        if x_span <= 0 or y_span <= 0 or bar_height <= 0:
            return

        color = self.trace_pen.color()

        x_bar = x_right + 0.045 * x_span
        cap_half = 0.010 * x_span
        y_bottom = y_min + 0.16 * y_span
        y_top = y_bottom + bar_height

        bar_pen = pg.mkPen(color, width=2)
        bar_items = [
            pg.PlotCurveItem([x_bar, x_bar], [y_bottom, y_top], pen=bar_pen),
            pg.PlotCurveItem([x_bar - cap_half, x_bar + cap_half], [y_top, y_top], pen=bar_pen),
            pg.PlotCurveItem([x_bar - cap_half, x_bar + cap_half], [y_bottom, y_bottom], pen=bar_pen),
        ]
        for item in bar_items:
            item.setZValue(900)
            plot.addItem(item)

        txt = pg.TextItem(text=bar_label, anchor=(0, 0.5), color=color)
        txt.setPos(x_bar + 0.012 * x_span, (y_top + y_bottom) / 2)
        txt.setZValue(901)
        plot.addItem(txt)

    def _draw_epoch_panel(self, x0, x1):
        self.epoch_plot.clear()
        self.epoch_plot.setXRange(x0, x1, padding=0)

        selected_epocs = self.epoc_group.selected_items()
        if not selected_epocs:
            self.epoch_plot.setYRange(0, 1, padding=0)
            self._clear_left_axis_labels(self.epoch_plot)
            return

        row_gap = self.channel_spacing * 0.25
        half_h = row_gap * 0.35
        y_ticks = []

        for idx, ep_name in enumerate(selected_epocs):
            y = -idx * row_gap
            y_ticks.append((y, ep_name))
            arr = self.epoc_onsets.get(ep_name)
            if arr is None or arr.size == 0:
                continue

            in_range = arr[(arr >= x0) & (arr <= x1)]
            for x in in_range:
                self.epoch_plot.addItem(pg.PlotCurveItem([x, x], [y - half_h, y + half_h], pen=self.epoch_pen))

        ymin = -max(1, len(selected_epocs)) * row_gap
        ymax = row_gap * 0.8
        self.epoch_plot.setYRange(ymin, ymax, padding=0)
        self._set_left_axis_labels(self.epoch_plot, y_ticks)

    def _plot_stream_panel(self, plot, sname, x_left, x_right):
        chans = self.selected_channels_for_stream(sname)
        if self.data is None or sname not in self.stream_names or not chans:
            plot.clear()
            self._clear_left_axis_labels(plot)
            plot.setXRange(x_left, x_right, padding=0)
            plot.setYRange(-1, 1, padding=0)
            return False

        obj = self.data.streams[sname]
        raw = np.asarray(obj.data)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]

        fs = float(getattr(obj, "fs", 1.0))
        start = float(getattr(obj, "start_time", 0.0))

        i1 = max(0, int((self.current_time - start) * fs))
        i2 = min(raw.shape[1], int((self.current_time + self.window_sec - start) * fs))
        if i2 <= i1:
            plot.clear()
            self._clear_left_axis_labels(plot)
            plot.setXRange(x_left, x_right, padding=0)
            plot.setYRange(-1, 1, padding=0)
            return False

        seg = raw[:, i1:i2]
        t = np.arange(seg.shape[1], dtype=float) / fs + start + (i1 / fs)

        step = max(1, int(np.ceil(len(t) / self.target_points_lfp))) if sname == "LFP_" else 1
        t_plot = t if step == 1 else t[::step]
        amp = self.stream_amp.get(sname, 1.0)
        gain = max(float(self.signal_gain), 1e-12)

        y_ticks = []
        y_grid_positions = []
        plot.clear()

        for row_idx, ch in enumerate(chans):
            if ch >= seg.shape[0]:
                continue

            y0 = -row_idx * self.channel_spacing
            y_grid_positions.append(y0)

            normalized_channel = seg[ch].astype(float) / gain
            display_channel = normalized_channel if step == 1 else normalized_channel[::step]
            y = display_channel * amp + y0

            plot.addItem(pg.PlotCurveItem(t_plot, y, pen=pg.mkPen(self.trace_pen.color(), width=self.line_width)))
            y_ticks.append((y0, f"ch{ch + 1}"))

        self._set_left_axis_labels(plot, y_ticks)

        dx = max(1e-9, x_right - x_left)
        x_plot_right = x_right + 0.14 * dx

        ymin = -(max(1, len(chans)) * self.channel_spacing) - self.channel_spacing * 1.2
        ymax = self.channel_spacing * 1.2

        plot.setXRange(x_left, x_plot_right, padding=0)
        plot.setYRange(ymin, ymax, padding=0)

        x_grid_positions = self._get_shared_time_grid_positions(x_left, x_right, n_lines=6)
        self._add_shared_x_grid_lines(plot, x_grid_positions, ymin, ymax)
        self._add_shared_y_grid_lines(plot, y_grid_positions, x_left, x_plot_right)

        self._add_scale_bar_outside(plot, x_left, x_right, ymin, ymax, sname)
        return True

    def _spectrogram_colormap(self):
        if self.spectrogram_colormap_combo.currentText().lower() == "jet":
            positions = np.array([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
            colors = np.array(
                [
                    [0, 0, 128],
                    [0, 0, 255],
                    [0, 255, 255],
                    [255, 255, 0],
                    [255, 0, 0],
                    [128, 0, 0],
                ],
                dtype=np.ubyte,
            )
            return pg.ColorMap(positions, colors)
        return pg.colormap.get("viridis")

    def _txt_spectrogram_colormap(self):
        if self.txt_spectrogram_colormap_combo.currentText().lower() == "jet":
            positions = np.array([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
            colors = np.array(
                [
                    [0, 0, 128],
                    [0, 0, 255],
                    [0, 255, 255],
                    [255, 255, 0],
                    [255, 0, 0],
                    [128, 0, 0],
                ],
                dtype=np.ubyte,
            )
            return pg.ColorMap(positions, colors)
        return pg.colormap.get("viridis")

    @staticmethod
    def _nice_frequency_values(fmin, fmax, target_intervals=3):
        span = max(float(fmax) - float(fmin), 1e-12)
        rough_step = span / max(1, int(target_intervals))
        magnitude = 10.0 ** np.floor(np.log10(rough_step))
        normalized = rough_step / magnitude
        if normalized <= 1.0:
            nice_step = 1.0 * magnitude
        elif normalized <= 2.0:
            nice_step = 2.0 * magnitude
        elif normalized <= 5.0:
            nice_step = 5.0 * magnitude
        else:
            nice_step = 10.0 * magnitude

        first = np.ceil(float(fmin) / nice_step) * nice_step
        values = np.arange(first, float(fmax) + nice_step * 0.25, nice_step).tolist()
        if not values or abs(values[0] - float(fmin)) > nice_step * 0.05:
            values.insert(0, float(fmin))
        if abs(values[-1] - float(fmax)) > nice_step * 0.05:
            values.append(float(fmax))
        return values

    @staticmethod
    def _neuroexplorer_gaussian_kernel(width_bins):
        """Return the Gaussian post-processing kernel documented by NeuroExplorer."""
        width = max(float(width_bins), np.finfo(float).eps)
        d = (int(width) + 1) // 2
        indices = np.arange(-2 * d, 2 * d + 1, dtype=float)
        sigma = -(width * width) * 0.25 / np.log(0.5)
        kernel = np.exp(-(indices * indices) / sigma)
        return kernel / np.sum(kernel)

    def _draw_spectrogram_panel(self, x_left, x_right):
        plot = self.spectrogram_plot
        plot.clear()
        self.spectrogram_colorbar.setImageItem([])
        plot.setXRange(x_left, x_right, padding=0)
        plot.getAxis("right").setTicks([[]])

        channels = self.spectrogram_channels.selected_channels()

        if (
            not self.show_spectrogram
            or self.data is None
            or "LFP_" not in self.stream_names
            or "LFP_" not in self.selected_streams()
            or not channels
        ):
            self._clear_left_axis_labels(plot)
            plot.setYRange(-1, 0, padding=0)
            return False

        obj = self.data.streams["LFP_"]
        raw = np.asarray(obj.data)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]

        channels = [channel for channel in channels if channel < raw.shape[0]]
        if not channels:
            return False

        fs = float(getattr(obj, "fs", 1.0))
        start = float(getattr(obj, "start_time", 0.0))
        if fs <= 0:
            return False

        requested_fmin = float(self.spectrogram_fmin_spin.value())
        requested_fmax = float(self.spectrogram_fmax_spin.value())
        if requested_fmax <= requested_fmin:
            requested_fmax = requested_fmin + 1.0

        analysis_fs = fs
        target_fs = max(1000.0, np.ceil(requested_fmax / 0.45))
        do_resample = fs > 2000.0 and target_fs < fs * 0.95

        requested_resolution = float(self.spectrogram_resolution_spin.value())
        minimum_window_sec = 1.0 / requested_resolution
        effective_window_sec = max(float(self.spectrogram_window_spin.value()), minimum_window_sec)
        pad_sec = max(1.0, 2.0 * effective_window_sec)
        read_left = max(start, x_left - pad_sec)
        stream_end = start + raw.shape[1] / fs
        read_right = min(stream_end, x_right + pad_sec)
        i1 = max(0, int(np.floor((read_left - start) * fs)))
        i2 = min(raw.shape[1], int(np.ceil((read_right - start) * fs)))
        if i2 - i1 < 8:
            return False

        samples = raw[channels, i1:i2].astype(float) / max(float(self.signal_gain), 1e-12)
        segment_start = start + i1 / fs

        if do_resample:
            ratio = Fraction(float(target_fs / fs)).limit_denominator(1000)
            samples = scipy_signal.resample_poly(
                samples, ratio.numerator, ratio.denominator, axis=-1
            )
            analysis_fs = fs * ratio.numerator / ratio.denominator

        resolution_samples = int(np.ceil(analysis_fs / requested_resolution))
        requested_samples = int(round(float(self.spectrogram_window_spin.value()) * analysis_fs))
        nperseg = min(max(8, requested_samples, resolution_samples), samples.shape[-1])
        nfft = max(nperseg, int(round(analysis_fs / requested_resolution)))
        overlap_from_percent = int(round(nperseg * self.spectrogram_overlap_spin.value() / 100.0))
        max_hop_samples = max(
            1, int(np.floor(float(self.spectrogram_time_step_spin.value()) * analysis_fs))
        )
        overlap_for_time_step = nperseg - max_hop_samples
        noverlap = max(overlap_from_percent, overlap_for_time_step)
        noverlap = min(max(0, noverlap), nperseg - 1)

        detrend_name = self.spectrogram_detrend_combo.currentText()
        detrend_value = False if detrend_name == "None" else detrend_name.lower()

        freq, rel_time, psd = scipy_signal.spectrogram(
            samples,
            fs=analysis_fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend_value,
            scaling="density",
            mode="psd",
            axis=-1,
        )

        abs_time = segment_start + rel_time
        effective_fmax = min(requested_fmax, analysis_fs * 0.45)
        fmask = (freq >= requested_fmin) & (freq <= effective_fmax)
        tmask = (abs_time >= x_left) & (abs_time <= x_right)
        if not np.any(fmask) or not np.any(tmask):
            plot.setYRange(-len(channels), 0, padding=0)
            return False

        freq = freq[fmask]
        abs_time = abs_time[tmask]
        psd = psd[:, fmask, :][:, :, tmask]

        percentage_mode = self.spectrogram_power_mode_combo.currentText() == "Percentage power"
        if percentage_mode:
            total_power = np.sum(psd, axis=1, keepdims=True)
            display_power = 100.0 * psd / np.maximum(total_power, np.finfo(float).tiny)
            power_label = "Power (%)"
            title_power_label = "% of displayed-range power"
            suffix = " %"
        else:
            display_power = 10.0 * np.log10(
                np.maximum(psd, np.finfo(float).tiny) / 1e-12
            )
            power_label = "dB re 1 µV²/Hz"
            title_power_label = power_label
            suffix = " dB"

        smoothing_description = ""
        if self.spectrogram_gaussian_check.isChecked():
            width_bins = float(self.spectrogram_gaussian_width_spin.value())
            kernel = self._neuroexplorer_gaussian_kernel(width_bins)
            display_power = scipy_signal.convolve(
                display_power,
                kernel[np.newaxis, :, np.newaxis],
                mode="same",
                method="auto",
            )
            if percentage_mode:
                smoothed_total = np.sum(display_power, axis=1, keepdims=True)
                display_power = 100.0 * display_power / np.maximum(
                    smoothed_total, np.finfo(float).tiny
                )
            smoothing_description = f" | NEX Gaussian width={width_bins:g} bins"

        self.spectrogram_db_min_spin.setSuffix(suffix)
        self.spectrogram_db_max_spin.setSuffix(suffix)
        db_min = float(self.spectrogram_db_min_spin.value())
        db_max = float(self.spectrogram_db_max_spin.value())
        if db_max <= db_min:
            db_max = db_min + 1.0

        dt = float(np.median(np.diff(abs_time))) if abs_time.size > 1 else max(
            1.0 / analysis_fs, x_right - x_left
        )
        image_width = float((abs_time[-1] - abs_time[0]) + dt)
        image_left = float(abs_time[0] - dt / 2.0)
        color_map = self._spectrogram_colormap()
        spectrogram_images = []
        channel_ticks = []
        frequency_ticks = []
        frequency_values = self._nice_frequency_values(requested_fmin, effective_fmax)
        separator_pen = pg.mkPen((150, 150, 150), width=1)
        frequency_grid_pen = pg.mkPen((150, 150, 150, 110), width=0.7)

        for row_index, channel in enumerate(channels):
            y_bottom = -(row_index + 1.0)
            band_height = 0.92
            image = pg.ImageItem(axisOrder="col-major")
            image.setImage(
                display_power[row_index].T,
                autoLevels=False,
                levels=(db_min, db_max),
            )
            image.setColorMap(color_map)
            image.setRect(QtCore.QRectF(image_left, y_bottom, image_width, band_height))
            plot.addItem(image)
            spectrogram_images.append(image)
            channel_ticks.append((y_bottom + band_height / 2.0, f"ch{channel + 1}"))
            frequency_span = max(effective_fmax - requested_fmin, 1e-12)
            for frequency in frequency_values:
                relative = (frequency - requested_fmin) / frequency_span
                y_tick = y_bottom + band_height * relative
                frequency_ticks.append((y_tick, f"{frequency:g}"))
                plot.addItem(
                    pg.PlotCurveItem(
                        [x_left, x_right], [y_tick, y_tick], pen=frequency_grid_pen
                    )
                )
            plot.addItem(
                pg.PlotCurveItem(
                    [x_left, x_right], [y_bottom + band_height, y_bottom + band_height],
                    pen=separator_pen,
                )
            )

        self._set_left_axis_labels(plot, channel_ticks)
        plot.getAxis("right").setTicks([frequency_ticks])
        self.spectrogram_colorbar.setColorMap(color_map)
        self.spectrogram_colorbar.setLevels((db_min, db_max))
        self.spectrogram_colorbar.setImageItem(spectrogram_images)
        self.spectrogram_colorbar.getAxis("left").setLabel(power_label)

        for ep_name in self.epoc_group.selected_items():
            onsets = self.epoc_onsets.get(ep_name)
            if onsets is None:
                continue
            for onset in onsets[(onsets >= x_left) & (onsets <= x_right)]:
                plot.addItem(
                    pg.InfiniteLine(pos=float(onset), angle=90, movable=False, pen=self.epoch_pen)
                )

        plot.setXRange(x_left, x_right, padding=0)
        plot.setYRange(-len(channels), 0, padding=0)
        actual_bin_hz = analysis_fs / nfft
        actual_window_sec = nperseg / analysis_fs
        actual_time_step_sec = (nperseg - noverlap) / analysis_fs
        plot.setTitle(
            f"LFP spectrograms {requested_fmin:g}-{effective_fmax:g} Hz | "
            f"Δf={actual_bin_hz:.3f} Hz, Δt={actual_time_step_sec:.3f} s | "
            f"{title_power_label} | detrend={detrend_name}{smoothing_description}",
            color=self.label_color,
            size="9pt",
        )
        self.spectrogram_fs_label.setText(
            f"Analysis fs: {analysis_fs:.1f} Hz"
            + (f" (from {fs:.1f} Hz)" if do_resample else " (native)")
            + f" | Δf: {actual_bin_hz:.3f} Hz | Δt: {actual_time_step_sec:.3f} s"
            + f" | window: {actual_window_sec:.3f} s"
        )
        return True

    def _draw_txt_spectrogram_panel(self, x_left, x_right):
        plot = self.txt_spectrogram_plot
        plot.clear()
        self.txt_spectrogram_colorbar.setImageItem([])
        plot.setXRange(x_left, x_right, padding=0)
        plot.getAxis("right").setTicks([[]])

        channels = self.txt_spectrogram_channels.selected_channels()
        if not self.show_txt_spectrogram or self.text_trace_data is None or not channels:
            self._clear_left_axis_labels(plot)
            plot.setYRange(-1, 0, padding=0)
            self.txt_spectrogram_fs_label.setText("Analysis fs: -")
            return False

        raw = np.asarray(self.text_trace_data)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        channels = [channel for channel in channels if channel < raw.shape[0]]
        if not channels:
            return False

        fs = float(self.text_trace_fs)
        start = float(self.text_trace_start_time)
        if fs <= 0:
            return False

        requested_fmin = float(self.txt_spectrogram_fmin_spin.value())
        requested_fmax = float(self.txt_spectrogram_fmax_spin.value())
        if requested_fmax <= requested_fmin:
            requested_fmax = requested_fmin + 0.001

        requested_resolution = float(self.txt_spectrogram_resolution_spin.value())
        minimum_window_sec = 1.0 / requested_resolution
        effective_window_sec = max(
            float(self.txt_spectrogram_window_spin.value()), minimum_window_sec
        )
        pad_sec = max(1.0, 2.0 * effective_window_sec)
        read_left = max(start, x_left - pad_sec)
        trace_end = start + raw.shape[1] / fs
        read_right = min(trace_end, x_right + pad_sec)
        i1 = max(0, int(np.floor((read_left - start) * fs)))
        i2 = min(raw.shape[1], int(np.ceil((read_right - start) * fs)))
        if i2 - i1 < 8:
            return False

        samples = raw[channels, i1:i2].astype(float)
        segment_start = start + i1 / fs
        resolution_samples = int(np.ceil(fs / requested_resolution))
        requested_samples = int(round(float(self.txt_spectrogram_window_spin.value()) * fs))
        nperseg = min(max(8, requested_samples, resolution_samples), samples.shape[-1])
        nfft = max(nperseg, int(round(fs / requested_resolution)))
        overlap_from_percent = int(
            round(nperseg * self.txt_spectrogram_overlap_spin.value() / 100.0)
        )
        max_hop_samples = max(
            1, int(np.floor(float(self.txt_spectrogram_time_step_spin.value()) * fs))
        )
        noverlap = max(overlap_from_percent, nperseg - max_hop_samples)
        noverlap = min(max(0, noverlap), nperseg - 1)

        detrend_name = self.txt_spectrogram_detrend_combo.currentText()
        detrend_value = False if detrend_name == "None" else detrend_name.lower()
        freq, rel_time, psd = scipy_signal.spectrogram(
            samples,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend_value,
            scaling="density",
            mode="psd",
            axis=-1,
        )

        abs_time = segment_start + rel_time
        effective_fmax = min(requested_fmax, fs * 0.5)
        fmask = (freq >= requested_fmin) & (freq <= effective_fmax)
        tmask = (abs_time >= x_left) & (abs_time <= x_right)
        if not np.any(fmask) or not np.any(tmask):
            self._clear_left_axis_labels(plot)
            plot.setYRange(-len(channels), 0, padding=0)
            self.txt_spectrogram_fs_label.setText(
                f"Analysis fs: {fs:.1f} Hz | no bins in selected range"
            )
            return False

        freq = freq[fmask]
        abs_time = abs_time[tmask]
        psd = psd[:, fmask, :][:, :, tmask]
        percentage_mode = (
            self.txt_spectrogram_power_mode_combo.currentText() == "Percentage power"
        )
        if percentage_mode:
            total_power = np.sum(psd, axis=1, keepdims=True)
            display_power = 100.0 * psd / np.maximum(
                total_power, np.finfo(float).tiny
            )
            power_label = "Power (%)"
            title_power_label = "% of displayed-range power"
            suffix = " %"
        else:
            display_power = 10.0 * np.log10(np.maximum(psd, np.finfo(float).tiny))
            power_label = "dB re 1 unit²/Hz"
            title_power_label = power_label
            suffix = " dB"

        smoothing_description = ""
        if self.txt_spectrogram_gaussian_check.isChecked():
            width_bins = float(self.txt_spectrogram_gaussian_width_spin.value())
            kernel = self._neuroexplorer_gaussian_kernel(width_bins)
            display_power = scipy_signal.convolve(
                display_power,
                kernel[np.newaxis, :, np.newaxis],
                mode="same",
                method="auto",
            )
            if percentage_mode:
                smoothed_total = np.sum(display_power, axis=1, keepdims=True)
                display_power = 100.0 * display_power / np.maximum(
                    smoothed_total, np.finfo(float).tiny
                )
            smoothing_description = f" | NEX Gaussian width={width_bins:g} bins"

        self.txt_spectrogram_db_min_spin.setSuffix(suffix)
        self.txt_spectrogram_db_max_spin.setSuffix(suffix)
        color_min = float(self.txt_spectrogram_db_min_spin.value())
        color_max = float(self.txt_spectrogram_db_max_spin.value())
        if color_max <= color_min:
            color_max = color_min + 1.0

        dt = float(np.median(np.diff(abs_time))) if abs_time.size > 1 else max(
            1.0 / fs, x_right - x_left
        )
        image_width = float((abs_time[-1] - abs_time[0]) + dt)
        image_left = float(abs_time[0] - dt / 2.0)
        color_map = self._txt_spectrogram_colormap()
        images = []
        channel_ticks = []
        frequency_ticks = []
        frequency_values = self._nice_frequency_values(requested_fmin, effective_fmax)
        separator_pen = pg.mkPen((150, 150, 150), width=1)
        frequency_grid_pen = pg.mkPen((150, 150, 150, 110), width=0.7)

        for row_index, channel in enumerate(channels):
            y_bottom = -(row_index + 1.0)
            band_height = 0.92
            image = pg.ImageItem(axisOrder="col-major")
            image.setImage(
                display_power[row_index].T,
                autoLevels=False,
                levels=(color_min, color_max),
            )
            image.setColorMap(color_map)
            image.setRect(QtCore.QRectF(image_left, y_bottom, image_width, band_height))
            plot.addItem(image)
            images.append(image)
            channel_ticks.append((y_bottom + band_height / 2.0, f"ch{channel + 1}"))
            frequency_span = max(effective_fmax - requested_fmin, 1e-12)
            for frequency in frequency_values:
                relative = (frequency - requested_fmin) / frequency_span
                y_tick = y_bottom + band_height * relative
                frequency_ticks.append((y_tick, f"{frequency:g}"))
                plot.addItem(
                    pg.PlotCurveItem(
                        [x_left, x_right], [y_tick, y_tick], pen=frequency_grid_pen
                    )
                )
            plot.addItem(
                pg.PlotCurveItem(
                    [x_left, x_right],
                    [y_bottom + band_height, y_bottom + band_height],
                    pen=separator_pen,
                )
            )

        self._set_left_axis_labels(plot, channel_ticks)
        plot.getAxis("right").setTicks([frequency_ticks])
        self.txt_spectrogram_colorbar.setColorMap(color_map)
        self.txt_spectrogram_colorbar.setLevels((color_min, color_max))
        self.txt_spectrogram_colorbar.setImageItem(images)
        self.txt_spectrogram_colorbar.getAxis("left").setLabel(power_label)

        for ep_name in self.epoc_group.selected_items():
            onsets = self.epoc_onsets.get(ep_name)
            if onsets is None:
                continue
            for onset in onsets[(onsets >= x_left) & (onsets <= x_right)]:
                plot.addItem(
                    pg.InfiniteLine(
                        pos=float(onset), angle=90, movable=False, pen=self.epoch_pen
                    )
                )

        plot.setXRange(x_left, x_right, padding=0)
        plot.setYRange(-len(channels), 0, padding=0)
        actual_bin_hz = fs / nfft
        actual_window_sec = nperseg / fs
        actual_time_step_sec = (nperseg - noverlap) / fs
        plot.setTitle(
            f"TXT spectrograms {requested_fmin:g}-{effective_fmax:g} Hz | "
            f"Δf={actual_bin_hz:.3f} Hz, Δt={actual_time_step_sec:.3f} s | "
            f"{title_power_label} | detrend={detrend_name}{smoothing_description}",
            color=self.label_color,
            size="9pt",
        )
        self.txt_spectrogram_fs_label.setText(
            f"Analysis fs: {fs:.1f} Hz (TXT) | Δf: {actual_bin_hz:.3f} Hz | "
            f"Δt: {actual_time_step_sec:.3f} s | window: {actual_window_sec:.3f} s"
        )
        return True

    def _plot_text_trace_panel(self, plot, x_left, x_right):
        dx = max(1e-9, x_right - x_left)
        x_plot_right = x_right + 0.14 * dx

        if self.text_trace_data is None:
            plot.clear()
            self._clear_left_axis_labels(plot)
            plot.setXRange(x_left, x_plot_right, padding=0)
            plot.setYRange(-1, 1, padding=0)
            return False

        chans = self.txt_channels.selected_channels()
        if not chans:
            plot.clear()
            self._clear_left_axis_labels(plot)
            plot.setXRange(x_left, x_plot_right, padding=0)
            plot.setYRange(-1, 1, padding=0)
            return False

        raw = self.text_trace_data
        fs = float(self.text_trace_fs)
        start = float(self.text_trace_start_time)

        i1 = max(0, int((x_left - start) * fs))
        i2 = min(raw.shape[1], int((x_right - start) * fs))
        if i2 <= i1:
            plot.clear()
            self._clear_left_axis_labels(plot)
            plot.setXRange(x_left, x_plot_right, padding=0)
            plot.setYRange(-1, 1, padding=0)
            return False

        seg = raw[:, i1:i2]
        t = np.arange(i1, i2, dtype=float) / fs + start
        amp = self.stream_amp.get("TXT_", 1.0)

        plot.clear()
        y_ticks = []
        y_grid_positions = []

        for row_idx, ch in enumerate(chans):
            if ch >= seg.shape[0]:
                continue

            y0 = -row_idx * self.channel_spacing
            y_grid_positions.append(y0)

            y = seg[ch].astype(float) * amp + y0
            plot.addItem(pg.PlotCurveItem(t, y, pen=pg.mkPen(self.trace_pen.color(), width=self.line_width)))
            y_ticks.append((y0, f"ch{ch + 1}"))

        self._set_left_axis_labels(plot, y_ticks)

        ymin = -(max(1, len(chans)) * self.channel_spacing) - self.channel_spacing * 1.2
        ymax = self.channel_spacing * 1.2

        plot.setXRange(x_left, x_plot_right, padding=0)
        plot.setYRange(ymin, ymax, padding=0)

        x_grid_positions = self._get_shared_time_grid_positions(x_left, x_right, n_lines=6)
        self._add_shared_x_grid_lines(plot, x_grid_positions, ymin, ymax)
        self._add_shared_y_grid_lines(plot, y_grid_positions, x_left, x_plot_right)

        self._add_scale_bar_outside(plot, x_left, x_right, ymin, ymax, "TXT_")
        return True

    def refresh_plot(self):
        self._refresh_timer.start(30)

    def _do_refresh_plot(self):
        self.epoch_plot.clear()
        self.text_plot.clear()
        self.txt_spectrogram_plot.clear()
        self.lfp_plot.clear()
        self.spectrogram_plot.clear()
        self.spectrogram_colorbar.setImageItem([])
        self.txt_spectrogram_colorbar.setImageItem([])
        self.mu_plot.clear()

        x_left = self.current_time
        x_right = min(self.total_duration, self.current_time + self.window_sec)
        if x_right <= x_left:
            x_right = x_left + max(0.001, self.window_sec)

        if self.data is not None:
            streams = self.selected_streams()
            self._draw_epoch_panel(x_left, x_right)

            if "LFP_" in streams:
                self._plot_stream_panel(self.lfp_plot, "LFP_", x_left, x_right)
            else:
                self.lfp_plot.clear()
                self._clear_left_axis_labels(self.lfp_plot)
                self.lfp_plot.setXRange(x_left, x_right, padding=0)
                self.lfp_plot.setYRange(-1, 1, padding=0)

            self._draw_spectrogram_panel(x_left, x_right)

            if "MUs_" in streams:
                self._plot_stream_panel(self.mu_plot, "MUs_", x_left, x_right)
            else:
                self.mu_plot.clear()
                self._clear_left_axis_labels(self.mu_plot)
                self.mu_plot.setXRange(x_left, x_right, padding=0)
                self.mu_plot.setYRange(-1, 1, padding=0)
        else:
            self.epoch_plot.clear()
            self._clear_left_axis_labels(self.epoch_plot)
            self.epoch_plot.setXRange(x_left, x_right, padding=0)
            self.epoch_plot.setYRange(-1, 1, padding=0)

            self.lfp_plot.clear()
            self._clear_left_axis_labels(self.lfp_plot)
            self.lfp_plot.setXRange(x_left, x_right, padding=0)
            self.lfp_plot.setYRange(-1, 1, padding=0)

            self.spectrogram_plot.clear()
            self._clear_left_axis_labels(self.spectrogram_plot)
            self.spectrogram_plot.setXRange(x_left, x_right, padding=0)
            self.spectrogram_plot.setYRange(-1, 0, padding=0)

            self.mu_plot.clear()
            self._clear_left_axis_labels(self.mu_plot)
            self.mu_plot.setXRange(x_left, x_right, padding=0)
            self.mu_plot.setYRange(-1, 1, padding=0)

        self._plot_text_trace_panel(self.text_plot, x_left, x_right)
        self._draw_txt_spectrogram_panel(x_left, x_right)
        self._sync_plot_margins()
        self._rebuild_cursor_items_after_clear()
        if self.camera_dock.has_camera:
            self.camera_dock.seek(
                self.cursor_time if self.camera_dock.playing else self.current_time
            )

        msg_parts = []
        if self.block_path:
            msg_parts.append(f"Block path: {self.block_path}")
        if self.text_trace_path:
            msg_parts.append(f"Text path: {self.text_trace_path}")
        if self.data is not None:
            msg_parts.append(f"Gain: {self.signal_gain:g}")
        if self.data is None and self.text_trace_data is None:
            msg_parts.append("Open a TDT block or a text trace.")
        self.statusBar().showMessage(" | ".join(msg_parts))

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("block_path", self.block_path)
        self.settings.setValue("text_trace_path", self.text_trace_path)
        self.settings.setValue("text_trace_fs", self.text_trace_fs)
        self.settings.setValue("txt_position", self.txt_position)
        self.settings.setValue("xaxis_mode", self.xaxis_mode)
        self.settings.setValue("current_time", self.current_time)
        self.settings.setValue("cursor_time", self.cursor_time)
        self.settings.setValue("window_sec", self.window_sec)
        self.settings.setValue("channel_spacing", self.channel_spacing)
        self.settings.setValue("white_mode", self.white_mode)
        self.settings.setValue("show_grid", self.show_grid)
        self.settings.setValue("show_scale_bar", self.show_scale_bar)
        self.settings.setValue("export_include_settings", self.export_include_settings)
        self.settings.setValue("show_cursor_on_traces", self.show_cursor_on_traces)
        self.settings.setValue("line_width", self.line_width)
        self.settings.setValue("lfp_amp", self.lfp_amp_spin.value())
        self.settings.setValue("mus_amp", self.mus_amp_spin.value())
        self.settings.setValue("txt_amp", self.txt_amp_spin.value())
        self.settings.setValue("signal_gain", self.gain_spin.value())
        self.settings.setValue("lfp_bar", self.lfp_bar_spin.value())
        self.settings.setValue("mus_bar", self.mus_bar_spin.value())
        self.settings.setValue("txt_bar", self.txt_bar_spin.value())
        self.settings.setValue("show_spectrogram", self.spectrogram_show_check.isChecked())
        self.settings.setValue("spectrogram_power_mode", self.spectrogram_power_mode_combo.currentText())
        self.settings.setValue("spectrogram_detrend", self.spectrogram_detrend_combo.currentText())
        self.settings.setValue("spectrogram_gaussian_smoothing", self.spectrogram_gaussian_check.isChecked())
        self.settings.setValue("spectrogram_gaussian_width_bins", self.spectrogram_gaussian_width_spin.value())
        self.settings.setValue("spectrogram_channels", self.spectrogram_channels.selected_channels())
        self.settings.setValue("spectrogram_fmin", self.spectrogram_fmin_spin.value())
        self.settings.setValue("spectrogram_fmax", self.spectrogram_fmax_spin.value())
        self.settings.setValue("spectrogram_resolution_hz", self.spectrogram_resolution_spin.value())
        self.settings.setValue("spectrogram_window_sec", self.spectrogram_window_spin.value())
        self.settings.setValue("spectrogram_time_step_sec", self.spectrogram_time_step_spin.value())
        self.settings.setValue("spectrogram_overlap", self.spectrogram_overlap_spin.value())
        self.settings.setValue("spectrogram_db_min", self.spectrogram_db_min_spin.value())
        self.settings.setValue("spectrogram_db_max", self.spectrogram_db_max_spin.value())
        self.settings.setValue("spectrogram_colormap", self.spectrogram_colormap_combo.currentText())
        self.settings.setValue("spectrogram_expanded", self.spectrogram_section.is_expanded())
        self.settings.setValue("spectrogram_channels_expanded", self.spectrogram_channels.is_expanded())
        self.settings.setValue("show_txt_spectrogram", self.txt_spectrogram_show_check.isChecked())
        self.settings.setValue("txt_spectrogram_power_mode", self.txt_spectrogram_power_mode_combo.currentText())
        self.settings.setValue("txt_spectrogram_detrend", self.txt_spectrogram_detrend_combo.currentText())
        self.settings.setValue("txt_spectrogram_gaussian_smoothing", self.txt_spectrogram_gaussian_check.isChecked())
        self.settings.setValue("txt_spectrogram_gaussian_width_bins", self.txt_spectrogram_gaussian_width_spin.value())
        self.settings.setValue("txt_spectrogram_channels", self.txt_spectrogram_channels.selected_channels())
        self.settings.setValue("txt_spectrogram_fmin", self.txt_spectrogram_fmin_spin.value())
        self.settings.setValue("txt_spectrogram_fmax", self.txt_spectrogram_fmax_spin.value())
        self.settings.setValue("txt_spectrogram_resolution_hz", self.txt_spectrogram_resolution_spin.value())
        self.settings.setValue("txt_spectrogram_window_sec", self.txt_spectrogram_window_spin.value())
        self.settings.setValue("txt_spectrogram_time_step_sec", self.txt_spectrogram_time_step_spin.value())
        self.settings.setValue("txt_spectrogram_overlap", self.txt_spectrogram_overlap_spin.value())
        self.settings.setValue("txt_spectrogram_db_min", self.txt_spectrogram_db_min_spin.value())
        self.settings.setValue("txt_spectrogram_db_max", self.txt_spectrogram_db_max_spin.value())
        self.settings.setValue("txt_spectrogram_colormap", self.txt_spectrogram_colormap_combo.currentText())
        self.settings.setValue("txt_spectrogram_expanded", self.txt_spectrogram_section.is_expanded())
        self.settings.setValue("txt_spectrogram_channels_expanded", self.txt_spectrogram_channels.is_expanded())
        self.settings.setValue("checked_streams", self._checked_stream_names())
        self.settings.setValue("lfp_channels", self.selected_channels_for_stream("LFP_"))
        self.settings.setValue("mus_channels", self.selected_channels_for_stream("MUs_"))
        self.settings.setValue("txt_channels", self.selected_channels_for_stream("TXT_"))
        self.settings.setValue("selected_epocs", self.epoc_group.selected_items())
        self.settings.setValue("lfp_expanded", self.lfp_channels.is_expanded())
        self.settings.setValue("mu_expanded", self.mus_channels.is_expanded())
        self.settings.setValue("txt_expanded", self.txt_channels.is_expanded())
        self.settings.setValue("epoc_expanded", self.epoc_group.is_expanded())
        self.settings.setValue("plot_splitter_sizes", self.plot_splitter.sizes())

        self.settings.setValue("block_expanded", self.block_section.is_expanded())
        self.settings.setValue("text_expanded", self.text_section.is_expanded())
        self.settings.setValue("stream_expanded", self.stream_section.is_expanded())
        self.settings.setValue("epoch_jump_expanded", self.epoch_jump_section.is_expanded())
        self.settings.setValue("cursor_expanded", self.cursor_section.is_expanded())
        self.settings.setValue("epoch_jump_name", self.epoch_jump_combo.currentText())
        self.settings.setValue("epoch_tick_value", self.epoch_tick_spin.value())

    def load_settings(self):
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        window_state = self.settings.value("windowState")
        if window_state is not None:
            self.restoreState(window_state)

        self.text_trace_fs = float(self.settings.value("text_trace_fs", self.default_text_trace_fs))
        self.txt_position = self.settings.value("txt_position", self.default_txt_position, type=str)
        if self.txt_position not in (self.TXT_POSITION_TOP, self.TXT_POSITION_MIDDLE, self.TXT_POSITION_BOTTOM):
            self.txt_position = self.default_txt_position

        self.xaxis_mode = self.settings.value("xaxis_mode", self.default_xaxis_mode, type=str)
        if self.xaxis_mode not in (self.XAXIS_BOTTOM_ONLY, self.XAXIS_ALL):
            self.xaxis_mode = self.default_xaxis_mode

        self.window_sec = float(self.settings.value("window_sec", self.default_window_sec))
        self.channel_spacing = float(self.settings.value("channel_spacing", self.default_channel_spacing))
        self.white_mode = self.settings.value("white_mode", self.default_white_mode, type=bool)
        self.show_grid = self.settings.value("show_grid", self.default_show_grid, type=bool)
        self.show_scale_bar = self.settings.value("show_scale_bar", self.default_show_scale_bar, type=bool)
        self.export_include_settings = self.settings.value("export_include_settings", False, type=bool)
        self.show_cursor_on_traces = self.settings.value("show_cursor_on_traces", False, type=bool)
        self.line_width = float(self.settings.value("line_width", self.default_line_width))
        self.cursor_time = float(self.settings.value("cursor_time", 0.0))
        self.current_time = float(self.settings.value("current_time", 0.0))

        self.lfp_channels.set_expanded(self.settings.value("lfp_expanded", False, type=bool))
        self.mus_channels.set_expanded(self.settings.value("mu_expanded", False, type=bool))
        self.txt_channels.set_expanded(self.settings.value("txt_expanded", False, type=bool))
        self.spectrogram_channels.set_expanded(self.settings.value("spectrogram_channels_expanded", True, type=bool))
        self.txt_spectrogram_channels.set_expanded(self.settings.value("txt_spectrogram_channels_expanded", True, type=bool))
        self.epoc_group.set_expanded(self.settings.value("epoc_expanded", False, type=bool))

        self.block_section.set_expanded(self.settings.value("block_expanded", False, type=bool))
        self.text_section.set_expanded(self.settings.value("text_expanded", False, type=bool))
        self.stream_section.set_expanded(self.settings.value("stream_expanded", False, type=bool))
        self.epoch_jump_section.set_expanded(self.settings.value("epoch_jump_expanded", False, type=bool))
        self.cursor_section.set_expanded(self.settings.value("cursor_expanded", False, type=bool))
        self.spectrogram_section.set_expanded(self.settings.value("spectrogram_expanded", True, type=bool))
        self.txt_spectrogram_section.set_expanded(self.settings.value("txt_spectrogram_expanded", True, type=bool))

        lfp_amp = float(self.settings.value("lfp_amp", self.default_lfp_amp))
        mus_amp = float(self.settings.value("mus_amp", self.default_mu_amp))
        txt_amp = float(self.settings.value("txt_amp", self.default_txt_amp))
        signal_gain = float(self.settings.value("signal_gain", self.default_signal_gain))
        lfp_bar = float(self.settings.value("lfp_bar", self.default_lfp_bar))
        mus_bar = float(self.settings.value("mus_bar", self.default_mu_bar))
        txt_bar = float(self.settings.value("txt_bar", self.default_txt_bar))

        spec_values = (
            (self.spectrogram_show_check, "setChecked", self.settings.value("show_spectrogram", self.default_show_spectrogram, type=bool)),
            (self.spectrogram_power_mode_combo, "setCurrentText", self.settings.value("spectrogram_power_mode", self.default_spectrogram_power_mode, type=str)),
            (self.spectrogram_detrend_combo, "setCurrentText", self.settings.value("spectrogram_detrend", self.default_spectrogram_detrend, type=str)),
            (self.spectrogram_gaussian_check, "setChecked", self.settings.value("spectrogram_gaussian_smoothing", self.default_spectrogram_gaussian_smoothing, type=bool)),
            (self.spectrogram_gaussian_width_spin, "setValue", float(self.settings.value("spectrogram_gaussian_width_bins", self.default_spectrogram_gaussian_width_bins))),
            (self.spectrogram_fmin_spin, "setValue", float(self.settings.value("spectrogram_fmin", self.default_spectrogram_fmin))),
            (self.spectrogram_fmax_spin, "setValue", float(self.settings.value("spectrogram_fmax", self.default_spectrogram_fmax))),
            (self.spectrogram_resolution_spin, "setValue", float(self.settings.value("spectrogram_resolution_hz", self.default_spectrogram_resolution_hz))),
            (self.spectrogram_window_spin, "setValue", float(self.settings.value("spectrogram_window_sec", self.default_spectrogram_window_sec))),
            (self.spectrogram_time_step_spin, "setValue", float(self.settings.value("spectrogram_time_step_sec", self.default_spectrogram_time_step_sec))),
            (self.spectrogram_overlap_spin, "setValue", int(self.settings.value("spectrogram_overlap", self.default_spectrogram_overlap))),
            (self.spectrogram_db_min_spin, "setValue", float(self.settings.value("spectrogram_db_min", self.default_spectrogram_db_min))),
            (self.spectrogram_db_max_spin, "setValue", float(self.settings.value("spectrogram_db_max", self.default_spectrogram_db_max))),
            (self.spectrogram_colormap_combo, "setCurrentText", self.settings.value("spectrogram_colormap", self.default_spectrogram_colormap, type=str)),
        )
        for widget, setter, value in spec_values:
            widget.blockSignals(True)
            getattr(widget, setter)(value)
            widget.blockSignals(False)
        self.show_spectrogram = self.spectrogram_show_check.isChecked()
        self.spectrogram_widget.setVisible(self.show_spectrogram)
        is_percentage = self.spectrogram_power_mode_combo.currentText() == "Percentage power"
        suffix = " %" if is_percentage else " dB"
        self.spectrogram_db_min_spin.setSuffix(suffix)
        self.spectrogram_db_max_spin.setSuffix(suffix)
        self._update_gaussian_controls()

        txt_spec_values = (
            (self.txt_spectrogram_show_check, "setChecked", self.settings.value("show_txt_spectrogram", self.default_show_txt_spectrogram, type=bool)),
            (self.txt_spectrogram_power_mode_combo, "setCurrentText", self.settings.value("txt_spectrogram_power_mode", self.default_txt_spectrogram_power_mode, type=str)),
            (self.txt_spectrogram_detrend_combo, "setCurrentText", self.settings.value("txt_spectrogram_detrend", self.default_txt_spectrogram_detrend, type=str)),
            (self.txt_spectrogram_gaussian_check, "setChecked", self.settings.value("txt_spectrogram_gaussian_smoothing", self.default_txt_spectrogram_gaussian_smoothing, type=bool)),
            (self.txt_spectrogram_gaussian_width_spin, "setValue", float(self.settings.value("txt_spectrogram_gaussian_width_bins", self.default_txt_spectrogram_gaussian_width_bins))),
            (self.txt_spectrogram_fmin_spin, "setValue", float(self.settings.value("txt_spectrogram_fmin", self.default_txt_spectrogram_fmin))),
            (self.txt_spectrogram_fmax_spin, "setValue", float(self.settings.value("txt_spectrogram_fmax", self.default_txt_spectrogram_fmax))),
            (self.txt_spectrogram_resolution_spin, "setValue", float(self.settings.value("txt_spectrogram_resolution_hz", self.default_txt_spectrogram_resolution_hz))),
            (self.txt_spectrogram_window_spin, "setValue", float(self.settings.value("txt_spectrogram_window_sec", self.default_txt_spectrogram_window_sec))),
            (self.txt_spectrogram_time_step_spin, "setValue", float(self.settings.value("txt_spectrogram_time_step_sec", self.default_txt_spectrogram_time_step_sec))),
            (self.txt_spectrogram_overlap_spin, "setValue", int(self.settings.value("txt_spectrogram_overlap", self.default_txt_spectrogram_overlap))),
            (self.txt_spectrogram_db_min_spin, "setValue", float(self.settings.value("txt_spectrogram_db_min", self.default_txt_spectrogram_db_min))),
            (self.txt_spectrogram_db_max_spin, "setValue", float(self.settings.value("txt_spectrogram_db_max", self.default_txt_spectrogram_db_max))),
            (self.txt_spectrogram_colormap_combo, "setCurrentText", self.settings.value("txt_spectrogram_colormap", self.default_txt_spectrogram_colormap, type=str)),
        )
        for widget, setter, value in txt_spec_values:
            widget.blockSignals(True)
            getattr(widget, setter)(value)
            widget.blockSignals(False)
        self.show_txt_spectrogram = self.txt_spectrogram_show_check.isChecked()
        self.txt_spectrogram_widget.setVisible(self.show_txt_spectrogram)
        is_txt_percentage = self.txt_spectrogram_power_mode_combo.currentText() == "Percentage power"
        txt_suffix = " %" if is_txt_percentage else " dB"
        self.txt_spectrogram_db_min_spin.setSuffix(txt_suffix)
        self.txt_spectrogram_db_max_spin.setSuffix(txt_suffix)
        self._update_txt_gaussian_controls()

        self.window_spin.blockSignals(True)
        self.window_spin.setValue(self.window_sec)
        self.window_spin.blockSignals(False)

        self.spacing_spin.blockSignals(True)
        self.spacing_spin.setValue(self.channel_spacing)
        self.spacing_spin.blockSignals(False)

        self.theme_check.blockSignals(True)
        self.theme_check.setChecked(self.white_mode)
        self.theme_check.blockSignals(False)

        self.scale_bar_check.blockSignals(True)
        self.scale_bar_check.setChecked(self.show_scale_bar)
        self.scale_bar_check.blockSignals(False)

        self.show_cursor_check.blockSignals(True)
        self.show_cursor_check.setChecked(self.show_cursor_on_traces)
        self.show_cursor_check.blockSignals(False)

        self.lfp_amp_spin.blockSignals(True)
        self.lfp_amp_spin.setValue(lfp_amp)
        self.lfp_amp_spin.blockSignals(False)

        self.mus_amp_spin.blockSignals(True)
        self.mus_amp_spin.setValue(mus_amp)
        self.mus_amp_spin.blockSignals(False)

        self.txt_amp_spin.blockSignals(True)
        self.txt_amp_spin.setValue(txt_amp)
        self.txt_amp_spin.blockSignals(False)

        self.gain_spin.blockSignals(True)
        self.gain_spin.setValue(signal_gain)
        self.gain_spin.blockSignals(False)

        self.txt_fs_spin.blockSignals(True)
        self.txt_fs_spin.setValue(self.text_trace_fs)
        self.txt_fs_spin.blockSignals(False)

        self.txt_position_combo.blockSignals(True)
        self.txt_position_combo.setCurrentText(self.txt_position)
        self.txt_position_combo.blockSignals(False)

        self.xaxis_mode_combo.blockSignals(True)
        self.xaxis_mode_combo.setCurrentText(self.xaxis_mode)
        self.xaxis_mode_combo.blockSignals(False)

        self.lfp_bar_spin.blockSignals(True)
        self.lfp_bar_spin.setValue(lfp_bar)
        self.lfp_bar_spin.blockSignals(False)

        self.mus_bar_spin.blockSignals(True)
        self.mus_bar_spin.setValue(mus_bar)
        self.mus_bar_spin.blockSignals(False)

        self.txt_bar_spin.blockSignals(True)
        self.txt_bar_spin.setValue(txt_bar)
        self.txt_bar_spin.blockSignals(False)

        self.stream_amp["LFP_"] = lfp_amp
        self.stream_amp["MUs_"] = mus_amp
        self.stream_amp["TXT_"] = txt_amp
        self.signal_gain = max(signal_gain, 1e-12)
        self.scale_bar_value_lfp = lfp_bar
        self.scale_bar_value_mus = mus_bar
        self.scale_bar_value_txt = txt_bar

        self._rebuild_plot_splitter()
        self._apply_plot_theme()

        self.main_splitter.setSizes([220, max(800, self.width() - 220)])
        self.show_settings_btn.setText("Hide")

        plot_sizes = self.settings.value("plot_splitter_sizes")
        if plot_sizes:
            try:
                sizes = [int(x) for x in plot_sizes]
                if len(sizes) == 6:
                    self.plot_splitter.setSizes(sizes)
            except Exception:
                pass

        last_block = self.settings.value("block_path", "", type=str)
        if last_block:
            self.block_path = last_block
            self.block_label.setPlainText(f"Block:\n{last_block}")

        last_text = self.settings.value("text_trace_path", "", type=str)
        if last_text:
            self.text_trace_path = last_text
            self.text_trace_label.setPlainText(f"Text Trace:\n{last_text}")

        self.total_duration = 1.0
        self._sync_time_widgets()

        QtCore.QTimer.singleShot(0, lambda: self.main_splitter.setSizes([220, max(800, self.width() - 220)]))

    def restore_channel_selection(self):
        lfp_saved = self.settings.value("lfp_channels", [])
        mus_saved = self.settings.value("mus_channels", [])
        txt_saved = self.settings.value("txt_channels", [])
        spec_saved = self.settings.value("spectrogram_channels", [])
        txt_spec_saved = self.settings.value("txt_spectrogram_channels", [])
        checked_streams = self.settings.value("checked_streams", [])
        selected_epocs = self.settings.value("selected_epocs", [])

        if isinstance(checked_streams, str):
            checked_streams = [checked_streams]
        if isinstance(lfp_saved, str):
            lfp_saved = [lfp_saved]
        if isinstance(mus_saved, str):
            mus_saved = [mus_saved]
        if isinstance(txt_saved, str):
            txt_saved = [txt_saved]
        if isinstance(spec_saved, str):
            spec_saved = [spec_saved]
        if isinstance(txt_spec_saved, str):
            txt_spec_saved = [txt_spec_saved]
        if isinstance(selected_epocs, str):
            selected_epocs = [selected_epocs]

        self._set_checked_stream_names(checked_streams)
        self.lfp_channels.set_selected_channels([int(x) for x in lfp_saved])
        self.mus_channels.set_selected_channels([int(x) for x in mus_saved])
        self.txt_channels.set_selected_channels([int(x) for x in txt_saved])
        self.spectrogram_channels.set_selected_channels([int(x) for x in spec_saved] or [0])
        self.txt_spectrogram_channels.set_selected_channels(
            [int(x) for x in txt_spec_saved]
            or self.txt_channels.selected_channels()
            or [0]
        )

        if self.default_epoc_name in self.epoc_names:
            self.epoc_group.set_selected_items([self.default_epoc_name])
        else:
            self.epoc_group.set_selected_items(selected_epocs)

        saved_time = float(self.settings.value("current_time", 0.0))
        self.current_time = min(max(0.0, saved_time), self.total_duration)
        self._sync_time_widgets()
        self.refresh_plot()

    def closeEvent(self, event):
        self.camera_dock.stop_playback()
        self.camera_dock.release()
        self.current_time = float(self.time_spin.value())
        self.window_sec = float(self.window_spin.value())
        self.channel_spacing = float(self.spacing_spin.value())
        self.text_trace_fs = float(self.txt_fs_spin.value())
        self.save_settings()
        super().closeEvent(event)


class AppController:
    def __init__(self):
        self.windows = []

    def create_viewer(self, block_path=""):
        win = TdtViewerWindow(controller=self, block_path=block_path)
        self.windows.append(win)
        win.destroyed.connect(self._cleanup_windows)
        win.show()
        return win

    def _cleanup_windows(self, *args):
        self.windows = [w for w in self.windows if w is not None and not _is_deleted(w)]

    def open_new_viewer_dialog(self):
        block_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select TDT Block Folder")
        if not block_path:
            return
        self.create_viewer(block_path)


def _is_deleted(widget):
    try:
        widget.objectName()
        return False
    except RuntimeError:
        return True


def main():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    controller = AppController()
    win = controller.create_viewer()

    app._controller = controller
    app._main_window = win

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
