"""A compact music adaptation of the LiquidAion Neural Observatory desktop."""
import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QSettings, QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget)

from .activity_view import TransformerActivityPanel, describe_event
from .anatomy import Anatomy
from .brain_view import BrainView
from .coupling import FlyConfig
from .engine import MODEL_ID
from .playback_sync import SAMPLE_RATE, PlaybackAligner
from .graph import load_graph, data_root
from .worker import MusicWorker

STYLE = """
QWidget {background:#0b111e;color:#dbe6f8;font-family:Inter,Arial,sans-serif;font-size:12px;}
QLabel#Title {font-size:24px;font-weight:700;color:#f1f6ff;}
QLabel#Muted {color:#8c9fbd;}
QGroupBox {border:1px solid #26334a;border-radius:9px;margin-top:12px;padding:14px 10px 9px;}
QGroupBox::title {subcontrol-origin:margin;left:12px;color:#a8c0e5;}
QLineEdit,QTextEdit {background:#101a2b;border:1px solid #26364e;border-radius:5px;padding:6px;}
QPushButton {background:#1a2a41;border:1px solid #344c6c;border-radius:5px;padding:7px 10px;}
QPushButton:hover {background:#243b59;} QPushButton:disabled {color:#62748c;}
QPushButton#Primary {background:#285faa;font-weight:bold;}
QComboBox,QSpinBox,QDoubleSpinBox {background:#132137;border:1px solid #30435f;padding:4px;border-radius:4px;}
QSlider::groove:horizontal {height:4px;background:#243752;}
QSlider::handle:horizontal {background:#71aef5;width:12px;margin:-4px 0;border-radius:5px;}
QCheckBox {spacing:8px;} QSplitter::handle {background:#172337;}
"""


class TracePlot(QWidget):
    def __init__(self):
        super().__init__()
        self.values = deque(maxlen=250)
        self.setMinimumHeight(70)

    def append(self, value):
        self.values.append(value)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0d1726"))
        values = list(self.values)
        if len(values) < 2:
            return
        p.setRenderHint(QPainter.Antialiasing)
        extent = max(max(abs(v) for v in values), 0.01)
        p.setPen(QPen(QColor("#f5d285"), 1.7))
        for i in range(1, len(values)):
            x0, x1 = [(j * (self.width()-8)/(len(values)-1)+4) for j in (i-1, i)]
            y0, y1 = [self.height()/2-v/extent*(self.height()/2-5) for v in values[i-1:i+1]]
            p.drawLine(round(x0), round(y0), round(x1), round(y1))


class Observatory(QMainWindow):
    def __init__(self, model_path=MODEL_ID, start_worker=True):
        super().__init__()
        self.setWindowTitle("FLY-TRANCE / Neural Music Observatory")
        self.resize(1500, 920)
        self.setStyleSheet(STYLE)
        self.settings = QSettings("Patternity", "FLY-TRANCE")
        self.selected = 0
        self.render_mark = (time.monotonic(), 0, 0.0)
        self.is_loaded = False
        self.is_running = False
        graph, self.nodes, _ = load_graph()
        anatomy = Anatomy.load(data_root() / "anatomy")
        if anatomy.neuron_ids != [n["root_id"] for n in self.nodes]:
            raise ValueError("Brain anatomy does not match connectome node order")
        root = QWidget(); self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("FLY-TRANCE"); title.setObjectName("Title")
        subtitle = QLabel("NEURAL MUSIC OBSERVATORY  /  Magenta RT2 Small × FlyWire CX512")
        subtitle.setObjectName("Muted")
        layout.addWidget(title); layout.addWidget(subtitle)
        split = QSplitter(Qt.Horizontal); layout.addWidget(split, 1)
        left = QWidget(); col = QVBoxLayout(left); col.setContentsMargins(0, 0, 0, 0)
        left.setMinimumWidth(340); left.setMaximumWidth(440)
        split.addWidget(left)
        model_box = QGroupBox("MUSIC MODEL"); form = QFormLayout(model_box)
        self.model_path = QLineEdit(model_path)
        form.addRow("Model / local folder", self.model_path)
        self.device = QComboBox(); self.device.addItems(["auto", "cuda", "cpu"])
        form.addRow("Device", self.device)
        self.load = QPushButton("Load Magenta RT2 Small"); self.load.setObjectName("Primary")
        form.addRow(self.load)
        self.style = QLineEdit(); self.style.setPlaceholderText("Neutral, or type a style prompt")
        form.addRow("Style", self.style)
        self.apply_style = QPushButton("Apply style"); self.apply_style.setEnabled(False); form.addRow(self.apply_style)
        self.monitor = QCheckBox("Play through audio device"); self.monitor.setChecked(True)
        form.addRow(self.monitor)
        self.volume = QSlider(Qt.Horizontal); self.volume.setRange(0, 100); self.volume.setValue(75)
        form.addRow("Playback volume", self.volume)
        actions = QWidget(); row = QHBoxLayout(actions); row.setContentsMargins(0, 0, 0, 0)
        self.start = QPushButton("Start"); self.start.setObjectName("Primary"); self.start.setEnabled(False)
        self.stop = QPushButton("Stop"); self.stop.setEnabled(False)
        row.addWidget(self.start); row.addWidget(self.stop); form.addRow(actions)
        col.addWidget(model_box)
        fly_box = QGroupBox("FLY CONNECTION"); f = QFormLayout(fly_box)
        self.fly = QCheckBox("Fly · inject into the music model")
        f.addRow(self.fly)
        self.capture = QComboBox(); self.inject = QComboBox()
        for i in range(12):
            self.capture.addItem(f"Temporal layer {i+1}", i)
            self.inject.addItem(f"Temporal layer {i+1}", i)
        self.capture.setCurrentIndex(5); self.inject.setCurrentIndex(11)
        f.addRow("Read", self.capture); f.addRow("Inject", self.inject)
        self.ensemble = QComboBox(); self.ensemble.addItems(["all", "DA", "ACH", "GABA", "GLUT", "SER"])
        f.addRow("Output ensemble", self.ensemble)
        self.strength = QDoubleSpinBox(); self.strength.setRange(0, 1); self.strength.setSingleStep(0.025); self.strength.setDecimals(3); self.strength.setValue(0.15)
        self.leak = QDoubleSpinBox(); self.leak.setRange(0.001, 1); self.leak.setSingleStep(0.025); self.leak.setDecimals(3); self.leak.setValue(0.2)
        f.addRow("Coupling strength", self.strength); f.addRow("Leak", self.leak)
        self.fly_status = QLabel("Fly OFF · brain continues listening"); self.fly_status.setWordWrap(True)
        f.addRow(self.fly_status)
        more = QPushButton("Dynamics ▸"); more.setCheckable(True); f.addRow(more)
        detail = QWidget(); df = QFormLayout(detail); df.setContentsMargins(0, 0, 0, 0)
        self.gain = QDoubleSpinBox(); self.gain.setRange(0, 0.999); self.gain.setDecimals(3); self.gain.setSingleStep(0.025); self.gain.setValue(0.9)
        self.input_gain = QDoubleSpinBox(); self.input_gain.setRange(0, 3); self.input_gain.setDecimals(3); self.input_gain.setSingleStep(0.05); self.input_gain.setValue(0.5)
        df.addRow("Recurrent gain", self.gain); df.addRow("Input gain", self.input_gain)
        detail.hide(); more.toggled.connect(detail.setVisible); f.addRow(detail)
        col.addWidget(fly_box)
        advanced = QGroupBox("SESSION / RECORDING"); af = QFormLayout(advanced)
        self.seed = QSpinBox(); self.seed.setRange(0, 2147483647); self.seed.setValue(42)
        af.addRow("Session seed", self.seed)
        # Off on every launch: long sessions write hundreds of MB of WAV and telemetry.
        self.record = QCheckBox("Save recording (WAV, metadata, telemetry)")
        self.record.setToolTip("Applies at the next Start. When off, nothing is written to disk.")
        af.addRow(self.record)
        self.outdir = QLineEdit(str(Path.cwd() / "outputs")); self.outdir.setEnabled(False)
        af.addRow("WAV output folder", self.outdir)
        self.record.toggled.connect(self.outdir.setEnabled)
        hint = QLabel("Start creates a new session. Fly toggles preserve all stream state.\nBlank style works offline once weights are cached.")
        hint.setWordWrap(True); hint.setObjectName("Muted"); af.addRow(hint)
        col.addWidget(advanced); col.addStretch()
        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0)
        split.addWidget(right); split.setStretchFactor(1, 1)
        self.brain = BrainView(anatomy, graph)
        # The brain stays visible next to the resizable, collapsible layer panel.
        self.brain_split = QSplitter(Qt.Horizontal)
        self.brain_split.addWidget(self.brain)
        self.activity = TransformerActivityPanel()
        self.brain_split.addWidget(self.activity)
        self.brain_split.setCollapsible(0, False)
        self.brain_split.setStretchFactor(0, 1)
        self.brain_split.setSizes([760, 360])
        rv.addWidget(self.brain_split, 1)
        views = QWidget(); vr = QHBoxLayout(views)
        for label, fn in [("Reset camera", self.brain.reset_camera), ("Focus CX512", self.brain.focus_neurons)]:
            b=QPushButton(label);b.clicked.connect(fn);vr.addWidget(b)
        edges = QCheckBox("Connections"); edges.toggled.connect(self.brain.set_connections_visible); vr.addWidget(edges)
        self.show_activity = QCheckBox("Transformer activity")
        self.show_activity.setToolTip("Show measured temporal-layer activations. Off stops their collection.")
        vr.addWidget(self.show_activity)
        self.sync_audio = QCheckBox("Sync with audio")
        self.sync_audio.setToolTip(
            "On: the brain, the neuron trace and the layer map show the state whose sound you hear now, "
            "instead of the state being generated (which runs ahead by the playback queue). Alignment uses "
            "the codec's one-frame latency and the output latency reported by the audio device; states "
            "arrive about ten times per second, so expect roughly ±0.1 s. Needs the audio device.")
        vr.addWidget(self.sync_audio)
        vr.addStretch(); rv.addWidget(views)
        self.neuron = QLabel(); rv.addWidget(self.neuron)
        self.trace = TracePlot(); rv.addWidget(self.trace)
        self.stats = QLabel("No model loaded · CX512: 512 neurons / 16,731 edges")
        # Text changes ten times per second must not trigger a relayout of the window.
        self.stats.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        rv.addWidget(self.stats)
        self.timeline = QTextEdit(); self.timeline.document().setMaximumBlockCount(300); self.timeline.setReadOnly(True); self.timeline.setMaximumHeight(115)
        rv.addWidget(self.timeline)
        self.status_label = QLabel("Load the model to begin"); layout.addWidget(self.status_label)
        self.select_neuron(0)
        self.brain.neuron_selected.connect(self.select_neuron)
        self.worker = MusicWorker()
        self.worker.status.connect(self.on_status)
        self.worker.loaded.connect(self.on_loaded)
        self.worker.telemetry.connect(self.on_telemetry)
        self.worker.error.connect(self.on_error)
        self.worker.event.connect(self.on_event)
        self.worker.recording.connect(lambda path: self.timeline.append(f"Saved WAV: {path}"))
        self.worker.activations.connect(self.on_activations)
        # Releases generated states when their sound is heard (or at once without an audio device).
        self.aligner = PlaybackAligner(self.worker.audible_sample, self._show_state, parent=self)
        self.display_lag = None
        self.activity.attach_manual(self.worker.manual)
        self.load.clicked.connect(self.load_model)
        self.start.clicked.connect(self.start_stream)
        self.stop.clicked.connect(lambda: self.worker.send("stop"))
        self.apply_style.clicked.connect(lambda: self.worker.send("style", style=self.style.text()))
        self.volume.valueChanged.connect(lambda v: self.worker.send("volume", value=v/100))
        self.fly.toggled.connect(self.apply_fly)
        for widget in (self.capture, self.inject, self.ensemble):
            widget.currentIndexChanged.connect(self.apply_fly)
        self.strength.valueChanged.connect(self.apply_fly); self.leak.valueChanged.connect(self.apply_fly)
        self.gain.valueChanged.connect(self.apply_fly); self.input_gain.valueChanged.connect(self.apply_fly)
        split_state = self.settings.value("activity_split")
        if split_state:
            self.brain_split.restoreState(split_state)
        self.show_activity.setChecked(self.settings.value("activity_visible", True, type=bool))
        self.set_activity_visible(self.show_activity.isChecked())
        self.show_activity.toggled.connect(self.set_activity_visible)
        self.sync_audio.setChecked(self.settings.value("sync_audio", True, type=bool))
        self.aligner.set_enabled(self.sync_audio.isChecked())
        self.sync_audio.toggled.connect(self.aligner.set_enabled)
        self.brain_split.splitterMoved.connect(self.on_activity_split)
        if start_worker:
            self.worker.start()
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def set_activity_visible(self, visible):
        self.activity.setVisible(visible)
        if visible and self.brain_split.sizes()[1] == 0:
            self.brain_split.setSizes([max(self.brain_split.width() - 360, 300), 360])
        # Observation runs only while shown; toggling never touches the stream.
        self.worker.send("observe", enabled=visible)

    def on_activity_split(self, *_):
        if self.brain_split.sizes()[1] == 0 and self.show_activity.isChecked():
            self.show_activity.setChecked(False)

    def on_activations(self):
        snapshot = self.worker.activity_slot.take()
        if snapshot is not None and self.show_activity.isChecked():
            self.aligner.push(snapshot.timing.get("sound_sample"), "snapshot", snapshot)

    def _show_state(self, kind, payload, lag_samples):
        """Display one generated state; lag_samples is None when shown at generation time."""
        self.display_lag = None if lag_samples is None else lag_samples / SAMPLE_RATE
        if kind == "snapshot":
            if self.show_activity.isChecked():
                payload.timing["display_lag"] = self.display_lag
                self.activity.update_snapshot(payload, self.worker.activity_slot.dropped)
        else:
            self.brain.update_state(payload["state"])
            self.trace.append(float(payload["state"][self.selected]))

    def config(self):
        return FlyConfig(self.fly.isChecked(), self.strength.value(), self.leak.value(),
                         self.capture.currentData(), self.inject.currentData(), self.ensemble.currentText(),
                         self.gain.value(), self.input_gain.value())

    def apply_fly(self, *_):
        self.fly_status.setText("Fly change pending at next frame" if self.is_loaded else "Fly configured · load model to apply")
        self.worker.send("fly", config=self.config())

    def load_model(self):
        self.load.setEnabled(False)
        self.worker.send("load", model=self.model_path.text(), device=self.device.currentText(), style=self.style.text())

    def on_loaded(self, data):
        self.is_loaded = True
        self.load.setEnabled(True); self.start.setEnabled(True)
        self.worker.send("fly", config=self.config())
        self.activity.set_model(data["layers"], data["dims"])
        self.timeline.append(f"Loaded Magenta RT2 Small on {data['device']}")

    def start_stream(self):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = str(Path(self.outdir.text()) / f"fly-trance-{stamp}.wav")
        record = self.record.isChecked()
        # No hold or pulse from before may continue into the new session.
        self.activity.new_session()
        self.aligner.clear()
        self.worker.manual.reset_session()
        self.worker.send("start", seed=self.seed.value(), config=self.config(), output=output,
                         monitor=self.monitor.isChecked(), record=record)
        self.timeline.append(f"Start · recording to {output}" if record else "Start · recording off, nothing is saved")
        self.start.setEnabled(False)

    def on_status(self, status):
        self.status_label.setText(status)
        if status == "Streaming":
            self.is_running = True
        elif status in ("Ready", "No model loaded", "Model loading failed"):
            self.is_running = False
        if status == "Model loading failed":
            self.is_loaded = False
            self.start.setEnabled(False)
        self.stop.setEnabled(self.is_running or status.startswith("Preparing"))
        if status in ("Ready", "No model loaded"):
            self.aligner.clear()  # the queued sound of a stopped stream is never heard
        if status == "Ready":
            self.start.setEnabled(True); self.stop.setEnabled(False)
        self.load.setEnabled(not self.is_running and not status.startswith(("Loading", "Preparing")))
        self.apply_style.setEnabled(self.is_loaded and not status.startswith(("Loading", "Preparing")))
        self.record.setEnabled(not self.is_running)
        self.outdir.setEnabled(self.record.isChecked() and not self.is_running)

    def on_error(self, error):
        self.status_label.setText(error)
        self.timeline.append(f"ERROR: {error}")
        self.load.setEnabled(not self.is_running)
        self.start.setEnabled(self.is_loaded and not self.is_running)

    def on_event(self, event):
        frame = event.get("effective_frame", 0)
        cfg = event.get("config")
        if event.get("type", "").startswith("manual_") or event.get("type") == "bypass":
            # Applied state reported by the generation thread, with its model frame.
            self.activity.on_manual_event(event)
            if event["type"] != "manual_hold_update":
                self.timeline.append(f"{frame/25:.2f}s · {describe_event(event)}")
        elif cfg:
            self.fly_status.setText(f"Fly {'ON' if cfg['enabled'] else 'OFF'} · applied at frame {frame}")
            self.timeline.append(f"{frame/25:.2f}s · Fly {'ON' if cfg['enabled'] else 'OFF'} · strength {cfg['strength']:.3f} · read {cfg['capture']+1} → inject {cfg['inject']+1}")

    def select_neuron(self, index):
        self.selected = index
        node = self.nodes[index]
        self.neuron.setText(f"Neuron {index} / {node['root_id']} · {node['nt_type'] or 'unclassified'} · {node['sub_class'] or 'CX'}")
        self.trace.values.clear(); self.trace.update()

    def on_telemetry(self, data):
        # The newest record wins; records that arrived while the GUI was busy are skipped.
        data = self.worker.telemetry_slot.take() or data
        # The brain state goes through the aligner; the diagnostics below are always immediate.
        self.aligner.push(data.get("sound_sample"), "telemetry", data)
        played = data["played_seconds"]
        played_text = f"{played:.2f}s" if played is not None else "monitor off"
        now = time.monotonic()
        renders = self.brain.render_stats
        span = max(now - self.render_mark[0], 1e-3)
        count = renders["count"] - self.render_mark[1]
        render_ms = 1000 * (renders["seconds"] - self.render_mark[2]) / max(count, 1)
        self.render_mark = (now, renders["count"], renders["seconds"])
        timing = (f" · step {data['step_ms_mean']:.0f}/{data['step_ms_p95']:.0f}/{data['step_ms_max']:.0f} ms "
                  f"(mean/p95/max), slow {data['slow_steps']}" if "step_ms_mean" in data else "")
        self.stats.setText(f"Generated {data['generated_seconds']:.2f}s · playback {played_text} · queue {data['queue_seconds']:.2f}s · RTF {data['rtf']:.2f} · underruns {data['underruns']} · Δ RMS {data['delta_rms']:.4f}"
                           f"{timing} · 3D {count / span:.0f}/s {render_ms:.1f} ms · display "
                           + (f"synced with audio (+{1000 * self.display_lag:.0f} ms)" if self.display_lag is not None
                              else "at generation time"))

    def changeEvent(self, event):
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            # A hold must not survive the window losing focus (e.g. Alt+Tab while pressed).
            self.activity.map.cancel("window inactive")
        super().changeEvent(event)

    def closeEvent(self, event):
        self.activity.map.cancel("closing")
        self.worker.shutdown.set()
        if self.worker.isRunning():
            self.status_label.setText("Finishing the current operation before closing...")
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("activity_visible", self.show_activity.isChecked())
        self.settings.setValue("sync_audio", self.sync_audio.isChecked())
        self.settings.setValue("activity_split", self.brain_split.saveState())
        self.brain.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = Observatory(args.model)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
