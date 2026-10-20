"""Render actual connectome-driven audio without opening a desktop."""
import argparse
import json
import math
import time
from pathlib import Path

import soundfile as sf

from .coupling import FlyConfig
from .engine import MODEL_ID, MusicEngine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--output", type=Path, default=Path("outputs/fly-trance.wav"))
    p.add_argument("--style", default="")
    p.add_argument("--fly", action="store_true")
    p.add_argument("--strength", type=float, default=0.15)
    p.add_argument("--leak", type=float, default=0.2)
    p.add_argument("--capture", type=int, default=5)
    p.add_argument("--inject", type=int, default=11)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        p.error("Seconds must be positive")
    engine = MusicEngine(args.model, args.device, args.style)
    engine.reset(args.seed)
    engine.configure(FlyConfig(args.fly, args.strength, args.leak, args.capture, args.inject))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with sf.SoundFile(args.output, "w", samplerate=48000, channels=2, subtype="PCM_16") as out:
        target = round(args.seconds * 48000)
        written = 0
        next_report = 0
        while written < target:
            audio, _ = engine.frame()
            audio = audio[:target-written]
            out.write(audio)
            written += len(audio)
            if written >= next_report:
                print(f"Generated {written / 48000:.1f}s / {args.seconds:.1f}s", flush=True)
                next_report += 48000
    meta = engine.metadata()
    meta["wall_seconds"] = time.monotonic() - started
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
