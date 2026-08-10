# ruff: noqa: T201
"""Validate CrewAI Sentence Transformer embeddings on Intel XPU.

The test exercises CrewAI's local Sentence Transformer provider and fails
if PyTorch cannot see an XPU, the model is not loaded on XPU, the model
forward pass does not observe XPU tensors, or the returned embeddings are
invalid.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
import sys
from typing import Any


DEFAULT_TEXTS = (
    "Intel XPUs accelerate local artificial intelligence workloads.",
    "CrewAI orchestrates autonomous agents and their tools.",
    "A sentence embedding is a dense numerical vector.",
)


class SmokeTestFailure(RuntimeError):
    """Raised when the XPU hardware smoke test fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeTestFailure(message)


def _tensor_devices(value: Any, torch: Any) -> set[str]:
    """Collect tensor device names recursively from a model input or output."""
    if isinstance(value, torch.Tensor):
        return {str(value.device)}
    if isinstance(value, Mapping):
        devices: set[str] = set()
        for item in value.values():
            devices.update(_tensor_devices(item, torch))
        return devices
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        devices = set()
        for item in value:
            devices.update(_tensor_devices(item, torch))
        return devices
    return set()


def _module_devices(model: Any) -> tuple[set[str], set[str]]:
    """Return the devices holding a model's parameters and buffers."""
    parameter_devices = {str(parameter.device) for parameter in model.parameters()}
    buffer_devices = {str(buffer.device) for buffer in model.buffers()}
    return parameter_devices, buffer_devices


def _validate_vectors(raw_vectors: Any, expected_count: int) -> tuple[int, list[float]]:
    """Validate shape, finiteness, normalization, and non-degeneracy."""
    _require(len(raw_vectors) == expected_count, "Unexpected embedding count")

    vectors = [[float(value) for value in vector] for vector in raw_vectors]
    _require(bool(vectors), "No embeddings were returned")
    dimension = len(vectors[0])
    _require(dimension > 0, "Embeddings have zero dimensions")
    _require(
        all(len(vector) == dimension for vector in vectors),
        "Embedding dimensions are inconsistent",
    )
    _require(
        all(math.isfinite(value) for vector in vectors for value in vector),
        "Embeddings contain a NaN or infinity",
    )

    norms = [math.sqrt(sum(value * value for value in vector)) for vector in vectors]
    _require(all(norm > 0 for norm in norms), "An embedding is the zero vector")
    _require(
        all(math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3) for norm in norms),
        "Normalized embeddings do not have unit length",
    )
    _require(vectors[0] != vectors[1], "Distinct texts produced identical embeddings")
    return dimension, norms


def run_smoke_test(model_name: str, device: str, texts: Sequence[str]) -> dict[str, Any]:
    """Run the CrewAI embedding path and return the collected XPU devices."""
    try:
        import torch
    except ImportError as error:
        raise SmokeTestFailure("PyTorch is not installed") from error

    _require(hasattr(torch, "xpu"), "This PyTorch build has no torch.xpu support")
    xpu_runtime_version = getattr(torch.version, "xpu", None)
    _require(
        torch.xpu.is_available(),
        "torch.xpu.is_available() returned False "
        f"(torch={torch.__version__}, torch.version.xpu={xpu_runtime_version!r}). "
        "Install PyTorch from https://download.pytorch.org/whl/xpu in an "
        "isolated environment; the normal CrewAI workspace uses CPU wheels.",
    )
    xpu_count = torch.xpu.device_count()
    _require(xpu_count > 0, "PyTorch reported zero XPU devices")

    try:
        from crewai.rag.embeddings.factory import build_embedder
    except ImportError as error:
        raise SmokeTestFailure(
            "CrewAI embedding dependencies are unavailable in this environment: "
            f"{type(error).__name__}: {error}"
        ) from error

    try:
        embedder = build_embedder(
            {
                "provider": "sentence-transformer",
                "config": {
                    "model_name": model_name,
                    "device": device,
                    "normalize_embeddings": True,
                },
            }
        )
    except (ImportError, ValueError) as error:
        raise SmokeTestFailure(
            "Could not construct the CrewAI Sentence Transformer embedder; "
            f"ensure sentence-transformers is installed. {type(error).__name__}: {error}"
        ) from error

    model = getattr(embedder, "_model", None)
    _require(model is not None, "The model from the embedding function is not accessible")

    parameter_devices, buffer_devices = _module_devices(model)
    _require(bool(parameter_devices), "The embedding model has no parameters")
    _require(
        all(value.startswith("xpu") for value in parameter_devices),
        f"CPU fallback detected in model parameters: {sorted(parameter_devices)}",
    )
    _require(
        all(value.startswith("xpu") for value in buffer_devices),
        f"CPU fallback detected in model buffers: {sorted(buffer_devices)}",
    )

    forward_devices: set[str] = set()

    def observe_forward(_module: Any, args: tuple[Any, ...], output: Any) -> None:
        forward_devices.update(_tensor_devices(args, torch))
        forward_devices.update(_tensor_devices(output, torch))

    hook = model.register_forward_hook(observe_forward)
    try:
        raw_vectors = embedder(list(texts))
        torch.xpu.synchronize()
    finally:
        hook.remove()

    _require(
        any(value.startswith("xpu") for value in forward_devices),
        f"The model forward pass observed no XPU tensors: {sorted(forward_devices)}",
    )
    dimension, norms = _validate_vectors(raw_vectors, len(texts))
    allocated_bytes = torch.xpu.memory_allocated()
    _require(allocated_bytes > 0, "PyTorch reports no allocated XPU memory")

    return {
        "status": "PASS",
        "torch_version": torch.__version__,
        "torch_xpu_runtime_version": xpu_runtime_version,
        "xpu_available": True,
        "xpu_count": xpu_count,
        "xpu_name": torch.xpu.get_device_name(0),
        "requested_device": device,
        "model_name": model_name,
        "parameter_devices": sorted(parameter_devices),
        "buffer_devices": sorted(buffer_devices),
        "forward_tensor_devices": sorted(forward_devices),
        "embedding_count": len(raw_vectors),
        "embedding_dimension": dimension,
        "embedding_norms": norms,
        "xpu_memory_allocated_bytes": allocated_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence Transformer model name or local model path.",
    )
    parser.add_argument(
        "--device",
        default="xpu",
        help="PyTorch XPU device passed through CrewAI (default: xpu).",
    )
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="Text to embed; repeat at least twice. Defaults to three sample texts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    texts = tuple(args.texts) if args.texts else DEFAULT_TEXTS
    if len(texts) < 2:
        print("FAIL: provide at least two --text values", file=sys.stderr)
        return 1
    if not args.device.startswith("xpu"):
        print("FAIL: --device must select an XPU", file=sys.stderr)
        return 1

    try:
        evidence = run_smoke_test(args.model, args.device, texts)
    except SmokeTestFailure as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())