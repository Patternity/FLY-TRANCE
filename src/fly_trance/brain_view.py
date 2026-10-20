"""VTK rendering of registered FlyWire geometry and measured reservoir state."""
from __future__ import annotations

from pathlib import Path
import os
import time

# NVIDIA throttles buffer swaps of occluded GL windows to about one per second, which would
# block the GUI thread inside every render. Rates are limited here instead; an explicit user
# setting wins. Must be set before the GL library initializes.
os.environ.setdefault("__GL_SYNC_TO_VBLANK", "0")

import numpy as np
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QImage, QPixmap
import pyvista as pv
from pyvistaqt import QtInteractor
import vtk

DATA_INTERVAL = 0.1          # seconds between renders caused by new activity data (<= 10/s)
INTERACTIVE_INTERVAL = 1 / 30  # seconds between renders while the user rotates or zooms
INTERACTIVE_HOLD = 0.3       # seconds after the last interaction that keep the faster rate
CLICK_SLOP = 4               # pixels; a press-release within this distance is a click, not a drag


class ScheduledInteractor(QtInteractor):
    """QtInteractor whose renders are requested, not executed on the spot.

    pyvista API calls, Qt paint events and VTK interactor styles all end in a request; the owner
    performs at most one render per interval with the latest scene state.
    """

    # Requests made while the base class is still initializing are dropped; the owner
    # requests a render once the scene is built.
    _request = None
    _click = None
    _press_pos = None

    def __init__(self, parent, request, click):
        super().__init__(parent, auto_update=False)
        self._request = request
        self._click = click

    # Clicks are recognized from Qt events: during a drag the VTK style grabs mouse focus,
    # so interactor observers would not see the release.
    def mousePressEvent(self, ev):
        self._press_pos = ev.position() if ev.button() == Qt.LeftButton else None
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        press, self._press_pos = self._press_pos, None
        if (ev.button() == Qt.LeftButton and press is not None and self._click is not None
                and (ev.position() - press).manhattanLength() <= CLICK_SLOP):
            self._click(*self.iren.interactor.GetEventPosition())

    def render(self):
        self._rendered = True
        if self._request is not None:
            self._request(False)

    def render_now(self):
        QtInteractor._render(self)


class BrainView(QWidget):
    neuron_selected = Signal(int)

    def __init__(self, anatomy, graph=None, parent=None):
        super().__init__(parent)
        self.anatomy = anatomy
        self.positions = np.asarray(anatomy.positions, dtype=np.float64)
        if self.positions.shape != (len(anatomy.neuron_ids), 3) or not np.isfinite(self.positions).all():
            raise ValueError("Anatomy must contain registered, finite positions for every graph neuron")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.headless = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        # One coalescing render path; nothing renders synchronously on input or data events.
        self._dirty = False
        self._last_render = 0.0
        self._interactive_until = 0.0
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_now)
        self.render_stats = {"count": 0, "seconds": 0.0}
        if self.headless:
            self.plotter = pv.Plotter(off_screen=True, window_size=(1100, 600))
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setMinimumSize(350, 200)
            layout.addWidget(self.image_label)
        else:
            self.plotter = ScheduledInteractor(self, self.request_render, self._clicked)
            # Hover needs no VTK handling (no hover widgets, picking is by click); without tracking
            # the Python mouse handler runs only while a button is held for rotate/zoom/pan.
            self.plotter.setMouseTracking(False)
            layout.addWidget(self.plotter.interactor)
        self.plotter.set_background("#080e19")
        self.plotter.add_text("FLYWIRE / FAFB14.1\nReal anatomical coordinates · nm", position="upper_left", font_size=9, color="#9bacc9", name="provenance")
        palette = ["#8394b2", "#596b90", "#9eabc7", "#728baa"]
        mesh_paths = list(getattr(anatomy, "mesh_paths", ()) or anatomy.meshes.values())
        if not mesh_paths:
            raise ValueError("No real anatomy meshes are available; synthetic geometry is forbidden")
        self.mesh_actors = []
        self.geometry = []
        for index, path in enumerate(mesh_paths):
            mesh = pv.read(str(path))
            if not mesh.n_points:
                raise ValueError(f"Empty anatomical mesh: {path}")
            self.geometry.append(mesh)
            self.mesh_actors.append(self.plotter.add_mesh(mesh, color=palette[index % len(palette)], opacity=0.20,
                smooth_shading=True, name=f"anatomy_{index}", pickable=False))
        self.points = pv.PolyData(self.positions.copy())
        self.activity_scale = 0.3
        self.current_state = np.zeros(len(self.positions), dtype=float)
        self.points["activity_rgba"] = np.zeros((len(self.positions), 4), dtype=np.uint8)
        self.neuron_actor = self.plotter.add_mesh(self.points, scalars="activity_rgba", rgb=True,
            point_size=9, render_points_as_spheres=True, name="neurons", show_scalar_bar=False,
            lighting=False, pickable=True)
        self.edge_actor = None
        if graph is not None:
            rows, cols = graph.nonzero()
            lines = np.column_stack((np.full(len(rows), 2), rows, cols)).ravel()
            edge_mesh = pv.PolyData(self.positions.copy(), lines=lines)
            self.edge_actor = self.plotter.add_mesh(edge_mesh, color="#496282", opacity=0.12, line_width=1,
                name="connections", pickable=False)
            self.edge_actor.SetVisibility(False)
        self.skeleton_actors = {}
        index_of = {str(root_id): i for i, root_id in enumerate(anatomy.neuron_ids)}
        self.skeleton_index = {}
        for root_id in getattr(anatomy, "skeletons", {}):
            vertices, edges = anatomy.read_skeleton(root_id)
            skeleton = pv.PolyData(np.asarray(vertices, dtype=np.float64),
                lines=np.column_stack((np.full(len(edges), 2), edges)).ravel())
            self.skeleton_actors[str(root_id)] = self.plotter.add_mesh(skeleton, color="#91c9d8", opacity=0.0,
                line_width=1, pickable=False)
            self.skeleton_index[str(root_id)] = index_of[str(root_id)]
        # The selection marker is created once and moved; actors are never rebuilt for selection.
        self.selection = pv.PolyData(self.positions[:1].copy())
        self.selection_actor = self.plotter.add_mesh(self.selection, color="#f9e09c", point_size=17,
            render_points_as_spheres=True, lighting=False, pickable=False, name="selected")
        self.selection_actor.SetVisibility(False)
        if self.plotter.iren is not None:
            iren = self.plotter.iren.interactor
            # Interactor styles still move the camera; their renders become coalesced requests.
            iren.EnableRenderOff()
            iren.AddObserver("RenderEvent", lambda *_: self.request_render(True))
            # A pick runs only for a click (press and release without a drag), never on hover.
            self._picker = vtk.vtkPointPicker()
            self._picker.SetTolerance(0.025)
            self._picker.PickFromListOn()
            self._picker.AddPickList(self.neuron_actor)
        self.plotter.view_xy(render=False)
        self.plotter.renderer.reset_camera(render=False)
        self.plotter.camera.zoom(1.5)
        self.request_render()

    def request_render(self, interactive=False):
        """Mark the scene dirty; at most one render per interval uses the latest state."""
        self._dirty = True
        now = time.monotonic()
        if interactive:
            self._interactive_until = now + INTERACTIVE_HOLD
        interval = INTERACTIVE_INTERVAL if now < self._interactive_until else DATA_INTERVAL
        delay_ms = max(0, int(1000 * (interval - (now - self._last_render))))
        if not self._render_timer.isActive() or self._render_timer.remainingTime() > delay_ms:
            self._render_timer.start(delay_ms)

    def _render_now(self):
        if not self._dirty or getattr(self.plotter, "_closed", False):
            # A deferred render can arrive after the window has closed the plotter.
            return
        if not self.headless and not self.isVisible():
            return  # showEvent requests a render once there is a window to draw into
        self._dirty = False
        self._last_render = time.monotonic()
        started = time.perf_counter()
        if self.headless:
            self.plotter.window_size = (max(500, self.image_label.width()), max(300, self.image_label.height()))
            self.plotter.render()
            pixels = np.ascontiguousarray(self.plotter.screenshot(return_img=True))
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format_RGB888).copy()
            self.image_label.setPixmap(QPixmap.fromImage(image).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.plotter.render_now()
        self.render_stats["count"] += 1
        self.render_stats["seconds"] += time.perf_counter() - started

    def showEvent(self, event):
        super().showEvent(event)
        self.request_render()

    def resizeEvent(self, event):
        if hasattr(self, "points"):
            self.request_render()
        super().resizeEvent(event)

    def _clicked(self, x, y):
        """One pick per click; there is no picking on hover or during rotation."""
        if self._picker.Pick(x, y, 0, self.plotter.renderer) and self._picker.GetDataSet() is not None:
            index = int(self._picker.GetPointId())
            if 0 <= index < len(self.positions):
                self.select_neuron(index)
                self.neuron_selected.emit(index)

    def select_neuron(self, index):
        self.selection.points = self.positions[index:index + 1].copy()
        self.selection_actor.SetVisibility(True)
        self.request_render()

    def update_state(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.positions),) or not np.isfinite(values).all():
            raise ValueError("Measured state does not match the anatomical neuron order")
        self.current_state = values.copy()
        amplitude = np.clip(np.abs(values) / self.activity_scale, 0, 1)
        rgba = np.zeros((len(values), 4), dtype=np.uint8)
        rgba[:, :3] = np.where((values >= 0)[:, None], np.array([255, 101, 94]), np.array([93, 164, 255]))
        rgba[:, 3] = np.rint(amplitude * 255).astype(np.uint8)
        self.points["activity_rgba"] = rgba
        for root_id, actor in self.skeleton_actors.items():
            index = self.skeleton_index[root_id]
            actor.GetProperty().SetColor(*(rgba[index, :3] / 255))
            actor.GetProperty().SetOpacity(float(amplitude[index]) * 0.65)
        self.points.Modified()
        self.request_render()

    def set_activity_scale(self, value):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Activity display scale must be finite and positive")
        self.activity_scale = float(value)
        self.update_state(self.current_state)

    def set_connections_visible(self, visible):
        if self.edge_actor is not None:
            self.edge_actor.SetVisibility(bool(visible))
            self.request_render()

    def set_anatomy_visible(self, visible):
        for actor in self.mesh_actors:
            actor.SetVisibility(bool(visible))
        self.request_render()

    def reset_camera(self):
        self.plotter.renderer.reset_camera(render=False)
        self.request_render()

    def focus_neurons(self):
        self.plotter.renderer.reset_camera(render=False, bounds=self.points.bounds)
        self.request_render()

    def export_image(self, path: Path):
        self.plotter.screenshot(str(path))

    def closeEvent(self, event):
        self._render_timer.stop()
        self.plotter.close()
        super().closeEvent(event)
