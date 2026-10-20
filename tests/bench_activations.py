"""Manual benchmark: model frame cost with activation observation off, at 5 Hz and on every frame.

FLY_MODEL_DIR=/path/to/small [FLY_BENCH_DEVICE=cuda] .venv/bin/python tests/bench_activations.py
"""
import os
import random
import statistics
import time
from dataclasses import replace

import numpy as np
import torch

from fly_trance.engine import MusicEngine

device = os.environ.get("FLY_BENCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
rounds = int(os.environ.get("FLY_BENCH_ROUNDS", "4"))
block = int(os.environ.get("FLY_BENCH_FRAMES", "125"))
engine = MusicEngine(os.environ["FLY_MODEL_DIR"], device=device)
probe = engine.coupling.probe
engine.reset(42)
# Fly ON with the read/inject layers of the last recorded GUI session.
engine.configure(replace(engine.coupling.config, enabled=True, strength=0.5, leak=0.75,
                         capture=9, inject=7, gain=0.949, input_strength=0.7))
modes = {"off": (False, 0.2), "5 Hz": (True, 0.2), "every frame": (True, 0.0)}
times = {name: [] for name in modes}
observed = {name: 0 for name in modes}
wall = {name: 0.0 for name in modes}
blocks = {name: [] for name in modes}


def run(name, frames):
    enabled, interval = modes[name]
    engine.set_observe(enabled)
    probe.interval = interval
    start = time.perf_counter()
    for _ in range(frames):
        t0 = time.perf_counter()
        engine.frame()
        snapshot = engine.take_activations()
        times[name].append(time.perf_counter() - t0)
        observed[name] += snapshot is not None
    wall[name] += time.perf_counter() - start
    blocks[name].append(statistics.mean(times[name][-frames:]))


for name in modes:  # warm-up of every path, discarded
    run(name, 25)
for name in modes:
    times[name].clear(); observed[name] = 0; wall[name] = 0.0; blocks[name].clear()
order = list(modes)
for r in range(rounds):
    random.Random(r).shuffle(order)
    for name in order:
        run(name, block)

label = torch.cuda.get_device_name(0) if device == "cuda" else f"CPU, {torch.get_num_threads()} torch threads"
print(f"Device: {label}; torch {torch.__version__}; dtype {engine.model._dt}; "
      f"{rounds} interleaved rounds x {block} frames per mode; Fly ON, read 10 -> inject 8")
base = statistics.mean(times["off"])
for name, values in times.items():
    ms = np.array(values) * 1000
    rate = observed[name] / wall[name]
    print(f"{name:>12}: mean {ms.mean():6.2f} ms  median {np.median(ms):6.2f}  p95 {np.percentile(ms, 95):6.2f}  "
          f"RTF {ms.mean() / 40:.3f}  snapshots {rate:5.2f}/s  overhead {100 * (ms.mean() / 1000 / base - 1):+5.2f}%")
for name in ("5 Hz", "every frame"):
    # Paired per round: each block is compared with the OFF block of the same round.
    paired = 100 * (np.array(blocks[name]) / np.array(blocks["off"]) - 1)
    sem = paired.std(ddof=1) / np.sqrt(len(paired))
    print(f"{name:>12}: paired overhead {paired.mean():+.2f}% ± {1.96 * sem:.2f}% (95% CI over {len(paired)} rounds)")

# Direct host-side cost of the observation code on an observed frame.
spent = {"finish": [], "collect": []}
finish, collect = probe.finish, probe.collect
def timed(fn, key):
    def wrapper(*args):
        t0 = time.perf_counter()
        result = fn(*args)
        spent[key].append(time.perf_counter() - t0)
        return result
    return wrapper
probe.finish, probe.collect = timed(finish, "finish"), timed(collect, "collect")
engine.set_observe(True); probe.interval = 0
for _ in range(100):
    engine.frame()
    engine.take_activations()
probe.finish, probe.collect = finish, collect
print("Observation code per observed frame: finish median "
      f"{1000 * statistics.median(spent['finish']):.3f} ms, collect median {1000 * statistics.median(spent['collect']):.3f} ms")

# GUI-thread cost of one panel update with a real snapshot (offscreen raster).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from fly_trance.activity_view import TransformerActivityPanel
app = QApplication.instance() or QApplication([])
engine.set_observe(True); probe.interval = 0
engine.frame()
snapshot = engine.take_activations()
snapshot.timing = {"generated_seconds": 1.0, "played_seconds": 0.1, "queue_seconds": 0.9}
panel = TransformerActivityPanel(); panel.resize(360, 760); panel.more.setChecked(True); panel.show()
app.processEvents()
paint = []
for _ in range(50):
    t0 = time.perf_counter()
    panel.update_snapshot(snapshot)
    panel.repaint()
    paint.append(time.perf_counter() - t0)
print(f"Panel update + repaint (offscreen, 360x760, detail open): median {1000 * statistics.median(paint):.2f} ms")
