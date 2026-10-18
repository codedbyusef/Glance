"""Checks used by setup_gpu_pc.ps1 (kept in a file: Windows PowerShell 5.1 mangles quotes in `python -c`)."""
import sys
import time


def gpu():
    import cv2
    import torch
    import transformers

    ok = torch.cuda.is_available()
    print("torch", torch.__version__, "cuda", ok, torch.cuda.get_device_name(0) if ok else "-")
    print("transformers", transformers.__version__, "cv2", cv2.__version__,
          "csrt", hasattr(cv2, "TrackerCSRT_create") or hasattr(cv2, "legacy"))
    return 0 if ok else 3


def vision():
    import torch
    from object_detection.server import load_models

    t = time.time()
    models = load_models()
    print("loaded", sorted(models), "in", round(time.time() - t, 1), "s")
    print("GPU memory MB", round(torch.cuda.memory_allocated() / 1e6))
    return 0


def stt():
    from audio.server.stt_service import _load_whisper

    t = time.time()
    _load_whisper("small.en", "cuda")
    print("small.en on cuda loaded in", round(time.time() - t, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit({"gpu": gpu, "vision": vision, "stt": stt}[sys.argv[1]]())
