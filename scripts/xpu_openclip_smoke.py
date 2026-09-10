# ruff: noqa: T201
"""Validate CrewAI OpenCLIP text and image embeddings on Intel XPU.

The test exercises CrewAI's OpenCLIP provider through the embedding factory and
fails if PyTorch cannot see an XPU, the model is not loaded on XPU, the forward
pass does not observe XPU tensors, or the returned embeddings are invalid.

Both text and image embedding paths are exercised to cover the dual-encoder
architecture of CLIP models.
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
    "A photograph of a dog running through a green field.",
    "CrewAI orchestrates autonomous agents and their tools.",
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
    parameter_devices = {str(p.device) for p in model.parameters()}
    buffer_devices = {str(b.device) for b in model.buffers()}
    return parameter_devices, buffer_devices


def _register_forward_device_hooks(
    model: Any, observed_devices: set[str], torch: Any
) -> list[Any]:
    """Observe tensor devices for direct encoder calls and regular forwards."""

    def observe(_module: Any, args: tuple[Any, ...], output: Any) -> None:
        observed_devices.update(_tensor_devices(args, torch))
        observed_devices.update(_tensor_devices(output, torch))

    return [module.register_forward_hook(observe) for module in model.modules()]


def _validate_vectors(
    raw_vectors: Any, expected_count: int, label: str
) -> tuple[int, list[float]]:
    """Validate shape, finiteness, normalization, and non-degeneracy."""
    _require(
        len(raw_vectors) == expected_count,
        f"{label}: expected {expected_count} embeddings, got {len(raw_vectors)}",
    )

    vectors = [[float(v) for v in vec] for vec in raw_vectors]
    _require(bool(vectors), f"{label}: no embeddings were returned")
    dimension = len(vectors[0])
    _require(dimension > 0, f"{label}: embeddings have zero dimensions")
    _require(
        all(len(vec) == dimension for vec in vectors),
        f"{label}: embedding dimensions are inconsistent",
    )
    _require(
        all(math.isfinite(v) for vec in vectors for v in vec),
        f"{label}: embeddings contain a NaN or infinity",
    )

    norms = [math.sqrt(sum(v * v for v in vec)) for vec in vectors]
    _require(all(n > 0 for n in norms), f"{label}: an embedding is the zero vector")
    _require(
        all(math.isclose(n, 1.0, rel_tol=1e-3, abs_tol=1e-3) for n in norms),
        f"{label}: normalized embeddings do not have unit length",
    )
    if len(vectors) >= 2:
        _require(
            vectors[0] != vectors[1],
            f"{label}: distinct inputs produced identical embeddings",
        )
    return dimension, norms


def _make_test_images(count: int, size: int = 224) -> list[Any]:
    """Create simple synthetic test images as numpy arrays (H, W, 3) uint8."""
    import numpy as np

    images = []
    rng = np.random.RandomState(42)
    for i in range(count):
        img = rng.randint(0, 255, (size, size, 3), dtype=np.uint8)
        img[:, :, i % 3] = min(50 + i * 80, 255)
        images.append(img)
    return images


def run_smoke_test(
    model_name: str, checkpoint: str, device: str, texts: Sequence[str]
) -> dict[str, Any]:
    """Run the CrewAI OpenCLIP embedding path and return collected evidence."""
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
                "provider": "openclip",
                "config": {
                    "model_name": model_name,
                    "checkpoint": checkpoint,
                    "device": device,
                },
            }
        )
    except (ImportError, ValueError) as error:
        raise SmokeTestFailure(
            "Could not construct the CrewAI OpenCLIP embedder; "
            "ensure open-clip-torch and pillow are installed. "
            f"{type(error).__name__}: {error}"
        ) from error

    model = getattr(embedder, "_model", None)
    _require(model is not None, "The model from the embedding function is not accessible")

    parameter_devices, buffer_devices = _module_devices(model)
    _require(bool(parameter_devices), "The embedding model has no parameters")
    _require(
        all(v.startswith("xpu") for v in parameter_devices),
        f"CPU fallback detected in model parameters: {sorted(parameter_devices)}",
    )
    _require(
        all(v.startswith("xpu") for v in buffer_devices),
        f"CPU fallback detected in model buffers: {sorted(buffer_devices)}",
    )

    # --- Text embeddings ---
    text_forward_devices: set[str] = set()
    hooks = _register_forward_device_hooks(model, text_forward_devices, torch)
    try:
        text_vectors = embedder(list(texts))
        torch.xpu.synchronize()
    finally:
        for hook in hooks:
            hook.remove()

    _require(
        any(v.startswith("xpu") for v in text_forward_devices),
        f"Text forward pass observed no XPU tensors: {sorted(text_forward_devices)}",
    )
    text_dim, text_norms = _validate_vectors(text_vectors, len(texts), "text")

    # --- Image embeddings ---
    test_images = _make_test_images(2)
    image_forward_devices: set[str] = set()
    hooks = _register_forward_device_hooks(model, image_forward_devices, torch)
    try:
        image_vectors = embedder(test_images)
        torch.xpu.synchronize()
    finally:
        for hook in hooks:
            hook.remove()

    _require(
        any(v.startswith("xpu") for v in image_forward_devices),
        f"Image forward pass observed no XPU tensors: {sorted(image_forward_devices)}",
    )
    image_dim, image_norms = _validate_vectors(image_vectors, len(test_images), "image")

    _require(
        text_dim == image_dim,
        f"Text dimension ({text_dim}) != image dimension ({image_dim}); "
        "CLIP text and image encoders must share the same embedding space",
    )

    # --- Cross-modal sanity: text and image embeddings should differ ---
    _require(
        [float(v) for v in text_vectors[0]] != [float(v) for v in image_vectors[0]],
        "First text and first image embedding are identical; "
        "cross-modal embeddings should differ",
    )

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
        "checkpoint": checkpoint,
        "parameter_devices": sorted(parameter_devices),
        "buffer_devices": sorted(buffer_devices),
        "text_forward_tensor_devices": sorted(text_forward_devices),
        "text_embedding_count": len(text_vectors),
        "text_embedding_dimension": text_dim,
        "text_embedding_norms": text_norms,
        "image_forward_tensor_devices": sorted(image_forward_devices),
        "image_count": len(test_images),
        "image_embedding_dimension": image_dim,
        "image_embedding_norms": image_norms,
        "xpu_memory_allocated_bytes": allocated_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        help="OpenCLIP model name (default: ViT-B-32).",
    )
    parser.add_argument(
        "--checkpoint",
        default="laion2b_s34b_b79k",
        help="OpenCLIP checkpoint (default: laion2b_s34b_b79k).",
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
        evidence = run_smoke_test(args.model, args.checkpoint, args.device, texts)
    except SmokeTestFailure as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
