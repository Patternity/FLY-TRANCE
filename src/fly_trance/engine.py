"""Stateful model ownership and frame-boundary control; GUI-independent."""
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from .coupling import FlyConfig, FlyCoupling
from .graph import load_graph

MODEL_ID = "magenta-community/magenta-realtime-2-small"
MODEL_REVISION = "7037d99551c84ac5c6afb7f1a5e58c65e7233dbb"


class MusicEngine:
    def __init__(self, model_path=MODEL_ID, device="auto", style="", threads=4):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; choose CPU")
        self.device = device
        self.model_path = str(model_path)
        torch.set_num_threads(threads)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        local = Path(model_path).is_dir()
        if not local and model_path != MODEL_ID:
            raise ValueError("Choose MRT2 Small or a local folder containing that model")
        kwargs = {"local_files_only": True} if local else {"revision": MODEL_REVISION}
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                             dtype=dtype, **kwargs).to(device).eval()
        graph, self.nodes, self.manifest = load_graph()
        self.coupling = FlyCoupling(self.model.depthformer.decoder, graph, self.nodes)
        self.model._temporal_step = self.coupling.temporal_step
        self.model._depth_step = None
        self.generator_state = None
        self.source = None
        self.style = ""
        self.temperature = 1.3
        self.top_k = 40
        self.events = []
        self.set_style(style)
        self.reset(42)

    def set_style(self, style):
        style = style.strip()
        if style:
            if self.model.processor is None:
                # Keep the large style encoder off the realtime GPU path.
                from huggingface_hub import snapshot_download
                folder = snapshot_download("magenta-community/magenta-rt-musiccoca-torch",
                    revision="236c488e38aa98643805514996934d705668298b",
                    allow_patterns=["text_encoder.pt", "quantizer.pt", "spm.model"])
                self.model.load_processor(repo_id=folder, device="cpu")
            tokens = self.model._tokenize_style(style)
        else:
            tokens = None
        with torch.no_grad():
            cond = self.model._resolve_conditioning(tokens, None, None, None, None, None)
            self.source = self.model.depthformer.encode(cond).to(self.model._dt)
        self.style = style

    def reset(self, seed):
        decoder = self.model.depthformer.decoder
        self.dstate = decoder.init_streaming_f(1, self.model._dev, self.model._dt)
        self.decode_state = self.model.init_decode_state()
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.coupling.brain.reset()
        self.coupling.frame = 0
        self.coupling.probe.reset()
        self.coupling.manual.reset_session()
        self.new_events = []
        self.events = []
        self.seed = seed

    def configure(self, config):
        self.coupling.configure(config)
        event = {"type": "fly_config", "effective_frame": self.coupling.frame,
                 "monotonic": time.monotonic(), "config": asdict(config)}
        self.events.append(event)
        return event

    def set_observe(self, enabled):
        """Toggle activation observation; stream state is untouched."""
        self.coupling.probe.set_enabled(enabled)

    def take_events(self):
        """Journal events applied during the last frame(s), already in self.events."""
        events, self.new_events = self.new_events, []
        return events

    def end_manual(self, reason):
        """Close manual state when the stream ends, so nothing continues into a new session."""
        events = self.coupling.manual.end_session(self.coupling.frame, reason)
        self.events.extend(events)
        return events

    def take_activations(self):
        """Snapshot of the last frame if it was observed. Call after frame()."""
        return self.coupling.probe.collect()

    @torch.no_grad()
    def frame(self):
        # Use the sampler from the exact custom model implementation.
        import importlib
        module = importlib.import_module(type(self.model).__module__)
        sampler = module.make_sampler(self.temperature, self.top_k, self.generator)
        codes = self.model.depthformer.decoder.step_f(
            self.dstate, self.source, sampler=sampler,
            temporal_step=self.coupling.temporal_step, depth_step=None)
        audio = self.model.decode_stream(codes, self.decode_state)
        wav = audio[0].float().cpu().numpy()
        # After that sync, applied manual operations get their reference RMS values.
        applied = self.coupling.manual.resolve()
        if applied:
            self.events.extend(applied)
            self.new_events.extend(applied)
        if not np.isfinite(wav).all():
            raise RuntimeError("Nonfinite audio output")
        # Fixed headroom, matching the upstream playback convention.
        return np.clip(wav * 0.5, -1, 1).astype(np.float32), self.coupling.last

    def metadata(self):
        return {"model": self.model_path, "revision": MODEL_REVISION if self.model_path == MODEL_ID else "local-unverified", "device": self.device,
                "style": self.style, "seed": self.seed, "graph": self.manifest,
                "fly": asdict(self.coupling.config), "events": self.events,
                "note": "Computational rate model with connectome topology; not a biophysical simulation"}
