"""Backend-agnostic entry point: ``load_activation_pipeline``.

This is the single call an experiment script makes. It resolves a backend from the model + the host
(Apple Silicon -> MLX, CUDA server / RunPod -> PyTorch) and returns something satisfying the
``ActivationPipeline`` contract. The caller then runs generation + capture and trains probes on the
returned ``{layer: tensor}`` map without knowing or caring which backend produced it.

    pipeline = load_activation_pipeline("llama31_8b_instruct", dtype="float16", max_length=2048)
    pre = pipeline.capture(messages, layers=[8], add_generation_prompt=True)
    reply = pipeline.generate(messages, max_new_tokens=120)
"""
from __future__ import annotations

from geoprobe.models.interface import ActivationPipeline
from geoprobe.models.registry import resolve_backend, resolve_mlx_model_name


def load_activation_pipeline(
    model_key: str,
    *,
    backend: str = "auto",
    device: str | None = None,
    dtype: str = "float16",
    max_length: int = 2048,
    mlx_model: str | None = None,
) -> ActivationPipeline:
    """Load the right backend for ``model_key`` and return the common pipeline interface.

    backend: "auto" (default) picks MLX on Apple Silicon when an MLX model is known, else PyTorch;
        "mlx" or "hf" force a backend.
    mlx_model: explicit mlx-lm path/repo to use for the MLX backend (e.g. a locally converted fp16
        model). Falls back to the registry's ``mlx_name`` and then to ``model_key``.
    """
    resolved = resolve_backend(model_key, backend)
    if resolved == "mlx":
        # Imported lazily so a CUDA/RunPod host never needs mlx installed.
        from geoprobe.models.mlx_capture import MlxActivationPipeline

        name = mlx_model or resolve_mlx_model_name(model_key) or model_key
        return MlxActivationPipeline.load(name, max_length=max_length, dtype=dtype)

    from geoprobe.models.hf_capture import HfActivationPipeline

    return HfActivationPipeline.load(model_key, device=device, dtype=dtype, max_length=max_length)
