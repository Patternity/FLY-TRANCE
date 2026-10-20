"""Compact Qt heat map of measured temporal-layer activations, with optional manual control."""
import math
import time

import numpy as np
from PySide6.QtCore import QEvent, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QToolButton, QToolTip, QVBoxLayout, QWidget)

from .activations import BASE, CX512, MANUAL, RESULT, grouping
from .manual import MAX_STRENGTH, coordinates

BACKGROUND = QColor("#0d1726")
BACKGROUND_RGB = np.array([13, 23, 38], dtype=np.uint8)
ZERO = np.array([17, 26, 42], dtype=float)
NEGATIVE = np.array([79, 157, 255], dtype=float)
POSITIVE = np.array([255, 90, 79], dtype=float)
ENERGY = np.array([242, 227, 192], dtype=float)
SATURATED_RGB = np.array([255, 225, 77], dtype=np.uint8)
READ = QColor("#57d4e8")
INJECT = QColor("#f5a542")
BYPASS = QColor("#b48cff")
APPLIED = QColor("#6ee7a8")
COMMAND = QColor("#f1f6ff")
AUTO_TAU = 5.0  # seconds; slow adaptation keeps amplitude differences visible
AUTO_PERCENTILE = 99.0
DELTA_ZOOM = 10.0
PX_PER_RMS = 100.0     # vertical drag distance for one unit of signed strength
HEARTBEAT_MS = 150     # hold refresh; the worker releases a hold after manual.HOLD_TIMEOUT
UI_REFRESH_MS = 50     # drag feedback is redrawn at most 20 times per second
# Subcolumns of one layer column in the composed image: mean, RMS strip, clip tick, gap.
SUB_MEAN, SUB_STRIP, SUB_TICK, SUB_GAP = 6, 2, 1, 1
SUB = SUB_MEAN + SUB_STRIP + SUB_TICK + SUB_GAP

PANEL_HELP = (
    "Each column is one temporal layer of the loaded Magenta model. All columns come from one model "
    "frame and are measured at the same position in every block: the residual stream after the FFN "
    "(x + FFN(x)), or the forwarded block input when the block is in output bypass, before the CX512 "
    "and manual additions.\n\n"
    "Rows are contiguous groups of coordinates of that vector. They are a technical compression, "
    "not neurons, ensembles or musical features.\n\n"
    "Wide cell: signed group mean on one shared linear scale ±R (blue negative, red positive, dark near zero).\n"
    "Narrow strip: group RMS on one shared scale 0…R_rms, so a group whose positive and negative "
    "coordinates cancel still shows its energy.\n"
    "Both scales are common to all layers and adapt slowly (or are fixed); they are never normalized "
    "per column. Yellow ticks mark cells clipped beyond the scale.\n\n"
    "Markers: R read into CX512, I CX512 injection (dashed when Fly is off), B block output bypass, "
    "green frame = manual addition applied in this snapshot's frame, white dashed frame = your current "
    "command.\n\n"
    "Snapshots are taken while the music is generated and can run ahead of what you hear by the "
    "playback queue; they are not sample-synchronous with the audio.")

MANUAL_HELP = (
    "Manual intervention mode. Off: clicks only select (safe viewing).\n"
    "On: press a cell to target its coordinate group. Drag up or down to change the signed strength "
    "(100 px = 1.0). The left button starts with the chosen sign, the right button with the opposite one.\n"
    "Pulse: one addition on one model frame, sent when you release the button.\n"
    "Hold: an addition on every model frame while the button is held; release, focus loss, hiding the "
    "panel or a stalled GUI end it.\n\n"
    "Rule: every real coordinate of the group gets strength × RMS of that layer's vector just before the "
    "manual addition (after bypass and CX512). Other coordinates are untouched; nothing is renormalized. "
    f"|strength| ≤ {MAX_STRENGTH:g}.")

BYPASS_HELP = (
    "Block output bypass: the block still runs and its attention caches keep advancing, but the next "
    "block receives this block's input instead of its output. No computation is saved, and this is not "
    "a removal of the block from the model. Re-enabling never clears caches or restarts the stream; "
    "the context accumulated meanwhile already reflects the bypass.")


def diverging_rgb(values, scale):
    """Linear map: -scale blue, 0 dark, +scale red. Returns (uint8 RGB, saturated mask)."""
    values = np.asarray(values, dtype=float)
    t = np.clip(np.abs(values) / scale, 0, 1)[..., None]
    target = np.where((values >= 0)[..., None], POSITIVE, NEGATIVE)
    rgb = ZERO + (target - ZERO) * t
    return np.rint(rgb).astype(np.uint8), np.abs(values) > scale


def energy_rgb(values, scale):
    """Linear map of a nonnegative RMS: 0 dark, scale warm white."""
    values = np.asarray(values, dtype=float)
    t = np.clip(values / scale, 0, 1)[..., None]
    return np.rint(ZERO + (ENERGY - ZERO) * t).astype(np.uint8), values > scale


def percentile_scale(values):
    values = np.abs(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    return max(float(np.percentile(values, AUTO_PERCENTILE)), 1e-6) if values.size else 1.0


def compose_image(mean, rms, scales):
    """One [rows, cols*SUB] RGB image for all columns; drawn with a single scaled call."""
    cols, rows = mean.shape
    mean_rgb, mean_sat = diverging_rgb(mean, scales[0])
    rms_rgb, rms_sat = energy_rgb(rms, scales[1])
    image = np.empty((rows, cols, SUB, 3), dtype=np.uint8)
    image[:, :, :SUB_MEAN] = mean_rgb.transpose(1, 0, 2)[:, :, None]
    image[:, :, SUB_MEAN:SUB_MEAN + SUB_STRIP] = rms_rgb.transpose(1, 0, 2)[:, :, None]
    image[:, :, SUB_MEAN + SUB_STRIP] = np.where((mean_sat | rms_sat).T[..., None], SATURATED_RGB, BACKGROUND_RGB)
    image[:, :, SUB - SUB_GAP:] = BACKGROUND_RGB
    return np.ascontiguousarray(image.reshape(rows, cols * SUB, 3))


def describe_event(event):
    """One line for the timeline and the confirmation label, from a worker journal event."""
    kind, frame = event["type"], event["effective_frame"]
    where = ""
    if "layer" in event and "coordinates" in event:
        a, b = event["coordinates"]
        where = f"layer {event['layer'] + 1} coords {a}–{b - 1}"
    if kind == "manual_pulse":
        return (f"Pulse #{event['id']} applied at frame {frame} · {where} · {event['strength']:+.2f} × RMS "
                f"{event['reference_rms']:.3g} = {event['per_coordinate']:+.3g} per coordinate")
    if kind == "manual_hold_start":
        return (f"Hold #{event['id']} started at frame {frame} · {where} · {event['strength']:+.2f} × RMS "
                f"{event['reference_rms']:.3g} = {event['per_coordinate']:+.3g} per coordinate")
    if kind == "manual_hold_update":
        return (f"Hold #{event['id']} at frame {frame} · {event['strength']:+.2f} × RMS "
                f"{event['reference_rms']:.3g} = {event['per_coordinate']:+.3g} per coordinate")
    if kind == "manual_hold_end":
        return (f"Hold #{event['id']} ended at frame {frame} ({event['reason']}) after "
                f"{event['frames_applied']} frames · last {event['final_strength']:+.2f}")
    if kind == "bypass":
        return f"Layer {event['layer'] + 1} output bypass {'ON' if event['enabled'] else 'OFF'} at frame {frame}"
    if kind == "manual_clear":
        cleared = ", ".join(str(i + 1) for i in event["bypass_cleared"]) or "none"
        hold = f"#{event['released_hold']}" if event["released_hold"] else "none"
        return (f"Manual interventions cleared at frame {frame} · hold {hold} · "
                f"{event['dropped_pulses']} pending pulses dropped · bypass cleared: {cleared}")
    return f"{kind} at frame {frame}"


class ActivityMap(QWidget):
    """Columns of coordinate groups: wide cell = signed group mean, strip = group RMS."""
    column_selected = Signal(int)
    gesture_started = Signal(int, int, int)   # column, row, sign of the mouse button (+1 left, -1 right)
    gesture_moved = Signal(float)             # upward drag in pixels since the press
    gesture_finished = Signal()
    gesture_cancelled = Signal(str)

    TOP, BOTTOM = 16, 15

    def __init__(self, min_height=200):
        super().__init__()
        self.mean = self.rms = None
        self.labels, self.markers, self.names = [], {}, []
        self.scales = (1.0, 1.0)
        self.selected, self.group_size = None, 1
        self.selectable = True
        self.interactive = False
        self.applied = set()        # (column, row) cells with a manual addition in the snapshot frame
        self.command = None         # (column, row, text) of the GUI's current command
        self._image = None
        self._press_y = None
        self.paint_stats = {"count": 0, "seconds": 0.0}
        self.setMinimumHeight(min_height)
        # No hover tracking: moving the cursor over the map runs no Python code. Cell tooltips
        # use Qt's delayed ToolTip event and are rebuilt only when the cell changes.
        self.setMouseTracking(False)
        self.setFocusPolicy(Qt.ClickFocus)
        self._tip_cell = None

    def set_data(self, mean, rms, labels, names, group_size, scales, markers=None, applied=()):
        self.mean, self.rms = np.asarray(mean), np.asarray(rms)
        self.labels, self.names, self.group_size = labels, names, group_size
        self.scales, self.markers, self.applied = scales, markers or {}, set(applied)
        self._image = compose_image(self.mean, self.rms, scales)
        self.update()

    def set_command(self, command):
        if command != self.command:
            self.command = command
            self.update()

    def _geometry(self):
        cols, rows = self.mean.shape
        col_w = max(self.width() - 4, cols) / cols
        row_h = max(self.height() - self.TOP - self.BOTTOM, rows) / rows
        return cols, rows, col_w, row_h

    def _cell(self, pos):
        if self.mean is None:
            return None
        cols, rows, col_w, row_h = self._geometry()
        col = int((pos.x() - 2) // col_w)
        row = int((pos.y() - self.TOP) // row_h)
        return (col, row) if 0 <= col < cols and 0 <= row < rows else None

    def _cell_rectf(self, c, r):
        _, _, col_w, row_h = self._geometry()
        return QRectF(2 + c * col_w, self.TOP + r * row_h, col_w * (SUB - SUB_GAP) / SUB, row_h)

    def paintEvent(self, event):
        started = time.perf_counter()
        p = QPainter(self)
        p.fillRect(self.rect(), BACKGROUND)
        if self.mean is None:
            p.setPen(QColor("#8c9fbd"))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, "Waiting for an observed model frame")
            return
        cols, rows, col_w, row_h = self._geometry()
        top, height = self.TOP, rows * row_h
        image = self._image
        qimage = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        p.drawImage(QRectF(2, top, cols * col_w, height), qimage)
        body_w = col_w * (SUB - SUB_GAP) / SUB
        font = p.font(); font.setPointSizeF(7.5); p.setFont(font)
        p.setBrush(Qt.NoBrush)
        for c in range(cols):
            x = 2 + c * col_w
            p.setPen(QColor("#f1f6ff") if c == self.selected else QColor("#8c9fbd"))
            p.drawText(QRectF(x - 4, top + height, body_w + 8, self.BOTTOM), Qt.AlignCenter, self.labels[c])
            marks = self.markers.get(c, [])
            for k, (text, color, style) in enumerate(marks):
                p.setPen(QPen(color, 1.6, style))
                inset = 2 * k
                p.drawRect(QRectF(x - inset, top - inset, body_w + 2 * inset, height + 2 * inset))
                slot = body_w / len(marks)
                p.drawText(QRectF(x + k * slot - 4, 0, slot + 8, self.TOP - 2), Qt.AlignCenter, text)
        # Command and applied-state markers are outlines; they never alter the measured colors.
        p.setPen(QPen(APPLIED, 2))
        for c, r in self.applied:
            if c < cols and r < rows:
                p.drawRect(self._cell_rectf(c, r).adjusted(-1, -1, 1, 1))
        if self.interactive:
            p.setPen(QPen(QColor("#ffb455"), 1.5))
            p.drawRect(QRectF(0.75, 0.75, self.width() - 1.5, self.height() - 1.5))
        if self.command is not None:
            c, r, text = self.command
            rect = self._cell_rectf(c, r).adjusted(-2, -2, 2, 2)
            p.setPen(QPen(COMMAND, 1.4, Qt.DashLine))
            p.drawRect(rect)
            p.drawText(QRectF(rect.right() + 2, rect.top() - 6, 60, 14), Qt.AlignLeft | Qt.AlignVCenter, text)
        self.paint_stats["count"] += 1
        self.paint_stats["seconds"] += time.perf_counter() - started

    def mousePressEvent(self, event):
        cell = self._cell(event.position())
        if cell is None:
            return
        if self.interactive and event.button() in (Qt.LeftButton, Qt.RightButton):
            # Qt keeps delivering move and release events to this widget while the button is down.
            self._press_y = event.position().y()
            self.gesture_started.emit(cell[0], cell[1], 1 if event.button() == Qt.LeftButton else -1)
        elif self.selectable:
            self.selected = cell[0]
            self.update()
            self.column_selected.emit(cell[0])

    def mouseMoveEvent(self, event):
        # Delivered only while a button is held (no tracking).
        if self._press_y is not None:
            self.gesture_moved.emit(self._press_y - event.position().y())

    def mouseReleaseEvent(self, event):
        if self._press_y is not None and not (event.buttons() & (Qt.LeftButton | Qt.RightButton)):
            self._press_y = None
            self.gesture_finished.emit()

    def cancel(self, reason):
        if self._press_y is not None:
            self._press_y = None
            self.gesture_cancelled.emit(reason)

    def focusOutEvent(self, event):
        self.cancel("focus lost")
        super().focusOutEvent(event)

    def hideEvent(self, event):
        self.cancel("hidden")
        super().hideEvent(event)

    def event(self, event):
        if event.type() == QEvent.ToolTip:
            cell = self._cell(event.pos())
            if cell is None:
                self._tip_cell = None
                QToolTip.hideText()
            elif cell != self._tip_cell or not QToolTip.isVisible():
                self._tip_cell = cell
                c, r = cell
                first = r * self.group_size
                text = (f"{self.names[c]} · group {r + 1} · coordinates {first}–{first + self.group_size - 1}\n"
                        f"mean {self.mean[c, r]:+.4g} · RMS {self.rms[c, r]:.4g}")
                if self.interactive:
                    text += (f"\nManual: each of these real coordinates of the layer vector "
                             f"gets strength × reference RMS; no other coordinate changes")
                # The rect keeps the tooltip while the cursor stays inside this cell.
                QToolTip.showText(event.globalPos(), text, self, self._cell_rectf(c, r).toAlignedRect())
            return True
        return super().event(event)


def _muted(text=""):
    label = QLabel(text)
    label.setObjectName("Muted")
    label.setWordWrap(True)
    return label


def _spin():
    spin = QDoubleSpinBox()
    spin.setRange(0.001, 100000); spin.setDecimals(3); spin.setValue(1.0); spin.setEnabled(False)
    spin.setMinimumWidth(56)
    return spin


class TransformerActivityPanel(QScrollArea):
    """Measured residual activations of every temporal block for one observed model frame."""

    def __init__(self):
        super().__init__()
        self.snapshot = None
        self.auto = None
        self.auto_time = None
        self.selected = None
        self.dropped = 0
        self.mailbox = None
        self.layers = self.dims = None
        self.group_size = 1
        self._gesture = None
        self._bypass_boxes = []
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(240)
        body = QWidget(); self.setWidget(body)
        layout = QVBoxLayout(body); layout.setContentsMargins(6, 0, 4, 0); layout.setSpacing(3)
        title = QLabel("TRANSFORMER ACTIVITY  ⓘ"); title.setStyleSheet("color:#a8c0e5;font-weight:bold;")
        title.setToolTip(PANEL_HELP)
        layout.addWidget(title)
        self.position = _muted("Temporal layers · after FFN, before CX512 and manual additions")
        self.position.setToolTip(PANEL_HELP)
        layout.addWidget(self.position)
        self.map = ActivityMap()
        layout.addWidget(self.map, 1)
        self.legend = _muted()
        self.legend.setToolTip(PANEL_HELP)
        layout.addWidget(self.legend)
        scale_row = QWidget(); sr = QHBoxLayout(scale_row); sr.setContentsMargins(0, 0, 0, 0); sr.setSpacing(4)
        self.scale_mode = QComboBox(); self.scale_mode.addItems(["Auto (slow)", "Fixed"])
        self.scale_mean, self.scale_rms = _spin(), _spin()
        self.scale_mean.setToolTip("Shared signed scale ±R for group means")
        self.scale_rms.setToolTip("Shared scale 0…R_rms for the group RMS strips")
        sr.addWidget(self.scale_mode); sr.addWidget(QLabel("±")); sr.addWidget(self.scale_mean, 1)
        sr.addWidget(QLabel("RMS")); sr.addWidget(self.scale_rms, 1)
        layout.addWidget(scale_row)
        self.layer_info = QLabel("Click a column for layer details"); self.layer_info.setWordWrap(True)
        layout.addWidget(self.layer_info)
        self.frame_info = _muted()
        self.frame_info.setToolTip("Model frame of this snapshot. Generation runs ahead of the audible "
                                   "output by the playback queue; this is not sample-synchronous.")
        layout.addWidget(self.frame_info)
        manual_row = QWidget(); mr = QHBoxLayout(manual_row); mr.setContentsMargins(0, 0, 0, 0); mr.setSpacing(4)
        self.manual_more = QPushButton("Manual control ▸"); self.manual_more.setCheckable(True)
        self.clear_manual = QPushButton("Clear manual interventions")
        self.clear_manual.setStyleSheet("QPushButton {border:1px solid #f5a542;color:#ffd9a0;}")
        self.clear_manual.setToolTip("Releases holds, drops pending pulses and returns every block from bypass. "
                                     "Fly settings, CX512, caches, codec, RNG and the music so far are kept; "
                                     "this stops new manual changes, it does not undo their effect.")
        mr.addWidget(self.manual_more); mr.addWidget(self.clear_manual, 1)
        layout.addWidget(manual_row)
        self.manual_box = QWidget(); ml = QVBoxLayout(self.manual_box); ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(3)
        self.manual_mode = QCheckBox("Enable mouse intervention (off = clicks only select)")
        self.manual_mode.setToolTip(MANUAL_HELP)
        ml.addWidget(self.manual_mode)
        controls = QWidget(); cr = QHBoxLayout(controls); cr.setContentsMargins(0, 0, 0, 0); cr.setSpacing(4)
        self.manual_kind = QComboBox(); self.manual_kind.addItems(["Pulse (1 frame)", "Hold (while pressed)"])
        self.positive = QToolButton(); self.positive.setText("+"); self.positive.setCheckable(True)
        self.negative = QToolButton(); self.negative.setText("−"); self.negative.setCheckable(True)
        self.positive.setChecked(True)
        self.polarity = QButtonGroup(self); self.polarity.setExclusive(True)
        self.polarity.addButton(self.positive); self.polarity.addButton(self.negative)
        self.manual_strength = QDoubleSpinBox(); self.manual_strength.setRange(0.0, MAX_STRENGTH)
        self.manual_strength.setDecimals(2); self.manual_strength.setSingleStep(0.05); self.manual_strength.setValue(0.5)
        self.manual_strength.setToolTip("Starting |strength| in units of the layer's reference RMS")
        cr.addWidget(self.manual_kind, 1); cr.addWidget(self.positive); cr.addWidget(self.negative)
        cr.addWidget(self.manual_strength); cr.addWidget(QLabel("× RMS"))
        ml.addWidget(controls)
        self.manual_info = _muted("Mode off. Turn it on, then press a cell.")
        ml.addWidget(self.manual_info)
        self.manual_confirm = QLabel("No manual operation applied yet"); self.manual_confirm.setWordWrap(True)
        self.manual_confirm.setStyleSheet("color:#6ee7a8;")
        self.manual_confirm.setToolTip("Confirmed by the generation thread, with the model frame that applied it.")
        ml.addWidget(self.manual_confirm)
        bypass_title = QLabel("Block output bypass  ⓘ"); bypass_title.setToolTip(BYPASS_HELP)
        ml.addWidget(bypass_title)
        self.bypass_grid = QGridLayout(); self.bypass_grid.setContentsMargins(0, 0, 0, 0); self.bypass_grid.setSpacing(2)
        grid_host = QWidget(); grid_host.setLayout(self.bypass_grid); grid_host.setToolTip(BYPASS_HELP)
        ml.addWidget(grid_host)
        self.manual_box.hide(); layout.addWidget(self.manual_box)
        self.more = QPushButton("Base / CX512 / Manual / Result ▸"); self.more.setCheckable(True)
        layout.addWidget(self.more)
        self.detail = QWidget(); dl = QVBoxLayout(self.detail); dl.setContentsMargins(0, 0, 0, 0); dl.setSpacing(3)
        self.part_map = ActivityMap(min_height=120)
        self.part_map.selectable = False
        dl.addWidget(self.part_map)
        self.zoom = QCheckBox(f"Show deltas ×{DELTA_ZOOM:g}")
        self.zoom.setToolTip("Only the two delta columns are multiplied for display; their headers say so. "
                             "The numbers below are always unscaled.")
        dl.addWidget(self.zoom)
        self.part_info = QLabel(); self.part_info.setWordWrap(True)
        dl.addWidget(self.part_info)
        dl.addWidget(_muted("Local measurement at the selected layer: each delta is the measured difference the "
                            "model actually received. Not a comparison with a trajectory without the fly or "
                            "without manual changes."))
        self.detail.hide(); layout.addWidget(self.detail)
        self.more.toggled.connect(self.detail.setVisible)
        self.manual_more.toggled.connect(self.manual_box.setVisible)
        self.manual_more.toggled.connect(lambda shown: shown or self.manual_mode.setChecked(False))
        self.manual_mode.toggled.connect(self._manual_mode_changed)
        self.manual_kind.currentIndexChanged.connect(self._hint_if_off)
        self.manual_strength.valueChanged.connect(self._hint_if_off)
        self.polarity.buttonClicked.connect(self._hint_if_off)
        self.clear_manual.clicked.connect(self.clear_interventions)
        self.map.column_selected.connect(self.select_layer)
        self.map.gesture_started.connect(self._gesture_started)
        self.map.gesture_moved.connect(self._gesture_moved)
        self.map.gesture_finished.connect(self._gesture_finished)
        self.map.gesture_cancelled.connect(self.cancel_gesture)
        self.scale_mode.currentIndexChanged.connect(self._scale_mode_changed)
        self.scale_mean.valueChanged.connect(self._render)
        self.scale_rms.valueChanged.connect(self._render)
        self.zoom.toggled.connect(self._render)
        self._heartbeat = QTimer(self); self._heartbeat.setInterval(HEARTBEAT_MS)
        self._heartbeat.timeout.connect(self._refresh_hold)
        self._ui_timer = QTimer(self); self._ui_timer.setSingleShot(True); self._ui_timer.setInterval(UI_REFRESH_MS)
        self._ui_timer.timeout.connect(self._show_gesture)

    # ---- model and manual wiring -------------------------------------------------------------
    def attach_manual(self, mailbox):
        self.mailbox = mailbox

    def set_model(self, layers, dims):
        """Build one bypass checkbox per temporal block of the loaded model."""
        self.layers, self.dims = layers, dims
        self.group_size = grouping(dims)[0]
        for box in self._bypass_boxes:
            box.deleteLater()
        self._bypass_boxes = []
        current = self.mailbox.bypass() if self.mailbox is not None else frozenset()
        for i in range(layers):
            box = QCheckBox(str(i + 1)); box.setToolTip(f"Layer {i + 1}: block output bypass. " + BYPASS_HELP)
            box.setChecked(i in current)
            box.toggled.connect(lambda checked, layer=i: self._set_bypass(layer, checked))
            self.bypass_grid.addWidget(box, i // 6, i % 6)
            self._bypass_boxes.append(box)

    def _set_bypass(self, layer, enabled):
        if self.mailbox is not None:
            self.mailbox.set_bypass(layer, enabled)
            self.manual_info.setText(f"Layer {layer + 1} bypass {'ON' if enabled else 'OFF'} requested; "
                                     "applied at the next model frame")

    def _manual_mode_changed(self, enabled):
        self.map.interactive = enabled
        # The active mode must be obvious: framed map, crosshair cursor, highlighted switch.
        self.map.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.manual_mode.setStyleSheet("QCheckBox {color:#ffb455;font-weight:bold;}" if enabled else "")
        self.map.update()
        if not enabled:
            self.map.cancel("manual mode off")
            self.cancel_gesture("manual mode off")
            self.manual_info.setText("Mode off. Clicks only select columns.")
        else:
            self.manual_info.setText("Press a cell: Pulse sends on release, Hold applies while pressed.")

    def _hint_if_off(self, *_):
        if self.manual_box.isVisible() and not self.manual_mode.isChecked():
            self.manual_info.setText("Mouse intervention is off: check “Enable mouse intervention” above, "
                                     "then press a cell.")

    def new_session(self):
        """A new stream starts: drop any gesture and confirmations that belong to the previous one."""
        self.map.cancel("new session")
        self.cancel_gesture("new session")
        self.manual_confirm.setText("No manual operation applied in this session yet")

    def clear_interventions(self):
        self.map.cancel("clear")
        self.cancel_gesture("clear")
        if self.mailbox is not None:
            self.mailbox.clear()
        for box in self._bypass_boxes:
            box.blockSignals(True); box.setChecked(False); box.blockSignals(False)
        self.manual_info.setText("Clear requested; the worker confirms it at the next model frame.")

    # ---- gestures ----------------------------------------------------------------------------
    def _strength(self, drag_px=0.0):
        return float(np.clip(self._gesture["base"] + drag_px / PX_PER_RMS, -MAX_STRENGTH, MAX_STRENGTH))

    def _gesture_started(self, col, row, button_sign):
        if self.mailbox is None or self.layers is None:
            self.manual_info.setText("Load the model before manual interventions.")
            return
        sign = (1 if self.positive.isChecked() else -1) * button_sign
        hold = self.manual_kind.currentIndex() == 1
        self._gesture = {"layer": col, "group": row, "base": sign * self.manual_strength.value(),
                         "hold": hold, "id": None}
        self._gesture["strength"] = self._strength()
        if hold:
            try:
                self._gesture["id"] = self.mailbox.start_hold(col, row, self._gesture["strength"])
            except (ValueError, RuntimeError) as exc:
                self.manual_info.setText(str(exc)); self._gesture = None
                return
            self._heartbeat.start()
        self._show_gesture()

    def _gesture_moved(self, drag_px):
        g = self._gesture
        if g is None:
            return
        g["strength"] = self._strength(drag_px)
        if g["hold"]:
            # Latest value wins in the mailbox; the worker applies it at the next frame boundary.
            self.mailbox.update_hold(g["id"], g["strength"])
        if not self._ui_timer.isActive():
            self._ui_timer.start()

    def _gesture_finished(self):
        g, self._gesture = self._gesture, None
        self._heartbeat.stop()
        self._ui_timer.stop()
        self.map.set_command(None)
        if g is None:
            return
        if g["hold"]:
            self.mailbox.release(g["id"])
            self.manual_info.setText(f"Hold #{g['id']} released; the worker confirms its last frame.")
            return
        a, b = coordinates(g["group"], self.group_size, self.dims)
        try:
            pulse_id = self.mailbox.pulse(g["layer"], g["group"], g["strength"])
        except (ValueError, RuntimeError) as exc:
            self.manual_info.setText(str(exc))
            return
        self.manual_info.setText(
            f"Pulse #{pulse_id} queued: layer {g['layer'] + 1} coords {a}–{b - 1}, {g['strength']:+.2f} × RMS"
            if pulse_id is not None else "Pulse rejected: too many pulses are still waiting for a frame")

    def cancel_gesture(self, reason):
        g, self._gesture = self._gesture, None
        self._heartbeat.stop()
        self._ui_timer.stop()
        self.map.set_command(None)
        if g is not None:
            if g["hold"] and self.mailbox is not None:
                self.mailbox.release(g["id"], reason)
                self.manual_info.setText(f"Hold #{g['id']} released: {reason}")
            else:
                self.manual_info.setText(f"Pulse cancelled: {reason}")

    def _refresh_hold(self):
        g = self._gesture
        if g is not None and g["hold"] and self.mailbox is not None:
            self.mailbox.update_hold(g["id"])

    def _show_gesture(self):
        g = self._gesture
        if g is None:
            return
        a, b = coordinates(g["group"], self.group_size, self.dims)
        self.map.set_command((g["layer"], g["group"], f"{g['strength']:+.2f}"))
        kind = f"Hold #{g['id']}" if g["hold"] else "Pulse on release"
        self.manual_info.setText(f"{kind} · layer {g['layer'] + 1} · coords {a}–{b - 1} ({b - a}) · "
                                 f"{g['strength']:+.2f} × RMS per coordinate")

    def hideEvent(self, event):
        self.map.cancel("panel hidden")
        self.cancel_gesture("panel hidden")
        super().hideEvent(event)

    def on_manual_event(self, event):
        """Confirmation from the worker: what was applied, on which model frame."""
        self.manual_confirm.setText("✓ " + describe_event(event))

    # ---- snapshots ---------------------------------------------------------------------------
    def scales(self):
        if self.scale_mode.currentIndex() == 1 or self.auto is None:
            return self.scale_mean.value(), self.scale_rms.value()
        return self.auto

    def _scale_mode_changed(self, index):
        self.scale_mean.setEnabled(index == 1)
        self.scale_rms.setEnabled(index == 1)
        self._render()

    def _adapt(self, snapshot):
        target = np.array([percentile_scale(snapshot.mean[BASE]), percentile_scale(snapshot.rms[BASE])])
        now = time.monotonic()
        if self.auto is None:
            current = target
        else:
            alpha = 1 - math.exp(-min(now - self.auto_time, 2.0) / AUTO_TAU)
            current = np.array(self.auto) + (target - np.array(self.auto)) * alpha
        self.auto, self.auto_time = (float(current[0]), float(current[1])), now
        if self.scale_mode.currentIndex() == 0:
            for spin, value in zip((self.scale_mean, self.scale_rms), self.auto):
                spin.blockSignals(True); spin.setValue(value); spin.blockSignals(False)

    def update_snapshot(self, snapshot, dropped=0):
        self.snapshot, self.dropped = snapshot, dropped
        self._adapt(snapshot)
        if self.selected is None or self.selected >= snapshot.layers:
            self.selected = snapshot.config["inject"]
            self.map.selected = self.selected
        self._render()

    def select_layer(self, index):
        self.selected = index
        self._render()
        if self.manual_box.isVisible() and not self.manual_mode.isChecked():
            self.manual_info.setText(f"Selected layer {index + 1} only. Mouse intervention is off: check "
                                     "“Enable mouse intervention” above to act on cells.")

    def _render(self, *_):
        s = self.snapshot
        if s is None:
            return
        cfg = s.config
        scales = self.scales()
        # Highlights come from the snapshot's own configuration, not from the current controls.
        markers = {}
        markers.setdefault(cfg["capture"], []).append(("R", READ, Qt.SolidLine))
        markers.setdefault(cfg["inject"], []).append(("I", INJECT, Qt.SolidLine if cfg["enabled"] else Qt.DashLine))
        for layer in s.bypass:
            markers.setdefault(layer, []).append(("B", BYPASS, Qt.DotLine))
        applied = {(op[0], op[1] // s.group_size) for op in s.manual_ops}
        self.map.set_data(s.mean[BASE], s.rms[BASE], [str(i + 1) for i in range(s.layers)],
                          [f"Layer {i + 1}" for i in range(s.layers)], s.group_size, scales, markers, applied)
        self.position.setText(f"{s.layers} temporal layers × {s.dims}-d · after FFN (bypass: block input), "
                              "before CX512 and manual additions")
        layer = self.selected if self.selected is not None else cfg["inject"]
        rms, lo, hi, _ = s.stats[BASE, layer]
        roles = [name for name, idx in (("read", cfg["capture"]), ("inject", cfg["inject"])) if idx == layer]
        if layer in s.bypass:
            roles.append("output bypass: forwarded input shown")
        role = f" ({' + '.join(roles)})" if roles else ""
        self.layer_info.setText(f"<b>Layer {layer + 1}</b>{role} · {s.dims}-d · RMS {rms:.4g} · "
                                f"min {lo:+.4g} · max {hi:+.4g}")
        played, lag = s.timing.get("played_seconds"), s.timing.get("display_lag")
        if lag is not None:
            lead = f"heard now (shown {1000 * lag:.0f} ms after its sound began)"
        else:
            lead = (f"≈{s.timing['generated_seconds'] - played:.2f} s ahead of playback"
                    if played is not None else "audio monitor off")
        self.frame_info.setText(f"Frame {s.frame} · music ≈{s.music_time:.2f} s · Fly cfg v{s.config_version} · "
                                f"manual v{s.manual_version} · {lead} · skipped {self.dropped}")
        zoom = DELTA_ZOOM if self.zoom.isChecked() else 1.0
        factors = np.array([1.0, zoom, zoom, 1.0])[:, None]
        suffix = f" ×{zoom:g}" if zoom != 1 else ""
        parts = [BASE, CX512, MANUAL, RESULT]
        names = ["Base", f"CX512 delta{suffix}", f"Manual delta{suffix}", "Result"]
        self.part_map.set_data(s.mean[parts, layer] * factors, s.rms[parts, layer] * factors,
                               ["Base", f"CXΔ{suffix}", f"ManΔ{suffix}", "Result"],
                               [f"Layer {layer + 1} · {n}" for n in names], s.group_size, scales)
        fly_state = "Fly ON" if cfg["enabled"] and layer == cfg["inject"] else "no CX512 addition here"
        manual_ref = float(s.manual_ref[layer])
        manual_text = (f"Manual Δ RMS {s.stats[MANUAL, layer, 0]:.4g} = {100 * s.delta_ratio(MANUAL, layer):.2f}% "
                       f"of reference RMS {manual_ref:.4g}" if manual_ref > 0 else "no manual addition in this frame")
        self.part_info.setText(f"<b>Layer {layer + 1}</b> · base RMS {s.stats[BASE, layer, 0]:.4g}<br>"
                               f"CX512 Δ RMS {s.stats[CX512, layer, 0]:.4g} = "
                               f"{100 * s.delta_ratio(CX512, layer):.2f}% of base ({fly_state})<br>"
                               f"{manual_text}<br>Result RMS {s.stats[RESULT, layer, 0]:.4g}")
        self._update_legend(scales)

    def _update_legend(self, scales):
        s = self.snapshot
        cfg = s.config
        sat = int(((np.abs(s.mean[BASE]) > scales[0]) | (s.rms[BASE] > scales[1])).sum())
        fly = "ON" if cfg["enabled"] else "OFF, dashed"
        bypass = ", ".join(str(i + 1) for i in s.bypass) or "none"
        self.legend.setText(
            f"<span style='color:#57d4e8'>R</span> read into CX512: layer {cfg['capture'] + 1} · "
            f"<span style='color:#f5a542'>I</span> injection: layer {cfg['inject'] + 1} (Fly {fly}) · "
            f"<span style='color:#b48cff'>B</span> bypass: {bypass}<br>"
            f"Cell = mean of {s.group_size} coords: <span style='color:#4f9dff'>■</span> −{scales[0]:.3g} … "
            f"<span style='color:#ff5a4f'>■</span> +{scales[0]:.3g} · strip = RMS 0…{scales[1]:.3g} · "
            f"<span style='color:#ffe14d'>▮</span> clipped {sat}")
