#!/usr/bin/env python3
"""Install backend dependencies with hardware-aware PyTorch selection.

Usage:
  python scripts/install_backend.py                    # core deps only (OpenAI embeddings)
  python scripts/install_backend.py --local-embeddings # + sentence-transformers stack
  python scripts/install_backend.py --local-embeddings --torch auto
  python scripts/install_backend.py --local-embeddings --torch cpu
  python scripts/install_backend.py --dev              # include pytest/ruff extras

Environment:
  TORCH_DEVICE=auto|cpu|cuda|cu124   overrides --torch (same values)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
# Default CUDA wheel index when an NVIDIA GPU is detected (PyTorch 2.x).
PYTORCH_CUDA_DEFAULT = "https://download.pytorch.org/whl/cu124"

PYTORCH_CUDA118_INDEX = "https://download.pytorch.org/whl/cu118"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _has_nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        subprocess.run(
            [exe, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def _cuda_major_from_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    match = re.search(r"CUDA Version:\s*(\d+)", out)
    return match.group(1) if match else None


def resolve_torch_mode(requested: str) -> str:
    """Return one of: skip, pypi, cpu, cu124, cu118."""
    mode = (os.environ.get("TORCH_DEVICE") or requested).strip().lower()
    if mode in ("", "skip", "none"):
        return "skip"
    if mode == "auto":
        if _has_nvidia_gpu():
            major = _cuda_major_from_nvidia_smi()
            if major == "11":
                return "cu118"
            return "cu124"
        # Linux/Windows without GPU: avoid PyPI's CUDA meta-package chain.
        if sys.platform in ("linux", "win32"):
            return "cpu"
        # macOS: PyPI wheels are CPU/MPS, no NVIDIA CUDA bundles.
        return "pypi"
    if mode == "cpu":
        return "cpu"
    if mode in ("cuda", "gpu"):
        return "cu124"
    if mode.startswith("cu"):
        return mode
    raise SystemExit(f"Unknown torch mode: {requested!r}. Use auto, cpu, cuda, cu124, or cu118.")


def torch_index_url(mode: str) -> str | None:
    if mode == "skip":
        return None
    if mode == "pypi":
        return None
    if mode == "cpu":
        return PYTORCH_CPU_INDEX
    if mode == "cu124":
        return PYTORCH_CUDA_DEFAULT
    if mode == "cu118":
        return PYTORCH_CUDA118_INDEX
    raise SystemExit(f"Internal error: unknown resolved torch mode {mode!r}")


def install_torch(mode: str) -> None:
    resolved = resolve_torch_mode(mode)
    if resolved == "skip":
        return
    index = torch_index_url(resolved)
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    _run(cmd)
    if index:
        print(f"Installing PyTorch ({resolved}) from {index}", flush=True)
        _run([sys.executable, "-m", "pip", "install", "torch", "--index-url", index])
    else:
        print("Installing PyTorch from PyPI (platform default wheel)", flush=True)
        _run([sys.executable, "-m", "pip", "install", "torch"])


def project_extras(local_embeddings: bool, dev: bool) -> str:
    parts: list[str] = []
    if local_embeddings:
        parts.append("local-embeddings")
    if dev:
        parts.append("dev")
    if not parts:
        return ""
    return f"[{','.join(parts)}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-embeddings",
        action="store_true",
        help="Install sentence-transformers and dependencies (requires PyTorch).",
    )
    parser.add_argument(
        "--torch",
        default="auto",
        help="PyTorch wheel set: auto (detect GPU), cpu, cuda/cu124, cu118, skip.",
    )
    parser.add_argument("--dev", action="store_true", help="Install dev extras (pytest, ruff).")
    args = parser.parse_args()

    if args.local_embeddings:
        install_torch(args.torch)

    extras = project_extras(args.local_embeddings, args.dev)
    target = f"-e .{extras}" if extras else "-e ."
    _run([sys.executable, "-m", "pip", "install", target])

    if args.local_embeddings:
        resolved = resolve_torch_mode(args.torch)
        print(f"\nLocal embeddings installed (torch mode: {resolved}).", flush=True)
    else:
        print(
            "\nCore backend installed without local embeddings (no PyTorch).",
            flush=True,
        )
        print(
            "For EMBEDDING_PROVIDER=local, re-run with: "
            "python scripts/install_backend.py --local-embeddings",
            flush=True,
        )


if __name__ == "__main__":
    main()
