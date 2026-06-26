"""Native-MLX activation backend for Apple Silicon.

Implements the same ``ActivationPipeline`` surface as ``hf_capture.HfActivationPipeline`` so the two
are interchangeable behind the factory. MLX has no forward-hook mechanism, so per-layer residuals are
collected by running the decoder blocks manually; the layer indexing is kept identical to HF
``output_hidden_states`` (layer 0 = embeddings, layer k = residual stream after block k-1, layer
n_layers = post-final-norm) so MLX and PyTorch activations line up.

Precision is selectable (default float16) via ``nn.Module.set_dtype``. For a genuinely fp16/bf16 run
point at an unquantized model; a 4-bit model keeps its quantized weights and only upcasts the compute
dtype.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Any

import numpy as np
import torch

from geoprobe.models.interface import PipelineMeta, ResidualSteeringSpec, SteeringSpec
from geoprobe.models.tokenization import chat_token_ids


def ensure_mlx_memory_limit(fraction: float = 0.55) -> None:
    """Cap MLX Metal memory before importing MLX, unless the caller already set a cap."""
    if "MLX_METAL_MEM_LIMIT" in os.environ:
        return
    try:
        total_ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except Exception:
        return
    os.environ["MLX_METAL_MEM_LIMIT"] = str(int(total_ram * fraction))


def _import_mlx():
    ensure_mlx_memory_limit()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    return mx, nn, load, stream_generate, make_sampler


def _import_mlx_core():
    ensure_mlx_memory_limit()
    import mlx.core as mx

    return mx


def clear_mlx_cache() -> None:
    mx = _import_mlx_core()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


def _resolve_mlx_dtype(mx: Any, name: str):
    table = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
    if name not in table:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(table)}")
    return table[name]


def _n_layers(model: Any) -> int:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("expected an mlx-lm decoder model with model.layers")
    return len(model.model.layers)


def _hidden_size(model: Any) -> int | None:
    args = getattr(model, "args", None) or getattr(model.model, "args", None)
    hidden = getattr(args, "hidden_size", None)
    return int(hidden) if hidden is not None else None


def load_mlx_model(model_name: str, *, dtype: str = "float16"):
    """Load an mlx-lm model and cast its (non-quantized) float params to ``dtype``."""
    mx, nn, load, *_ = _import_mlx()
    model, tokenizer = load(model_name)
    if hasattr(nn.Module, "set_dtype"):
        model.set_dtype(_resolve_mlx_dtype(mx, dtype))
    meta = PipelineMeta(
        name=model_name,
        backend="mlx",
        device="mlx",
        dtype=dtype,
        n_layers=_n_layers(model),
        hidden_size=_hidden_size(model),
    )
    return model, tokenizer, meta


def _causal_mask(mx: Any, n_tokens: int, window_size: int | None = None):
    rinds = mx.arange(n_tokens)
    linds = rinds[:, None]
    mask = linds >= rinds[None]
    if window_size is not None:
        mask = mask & (linds < rinds[None] + window_size)
    return mask


def _attention_mask(mx: Any, h: Any, cache: Any = None, window_size: int | None = None):
    n_tokens = h.shape[1]
    if cache is not None and hasattr(cache, "make_mask"):
        return cache.make_mask(n_tokens, window_size=window_size)
    if n_tokens == 1:
        return None
    if window_size is not None and n_tokens > window_size:
        return _causal_mask(mx, n_tokens, window_size=window_size)
    return "causal"


def _last_token_to_torch(mx: Any, hidden: Any) -> torch.Tensor:
    vec = hidden[0, -1, :].astype(mx.float32)
    mx.eval(vec)
    return torch.from_numpy(np.array(vec, dtype=np.float32)).unsqueeze(0)


def _torch_direction_to_mlx(mx: Any, direction: torch.Tensor, dtype: Any):
    vec = direction.detach().float().cpu().numpy()
    return mx.array(vec, dtype=dtype)


def _add_to_last_token(mx: Any, h: Any, vector: Any, alpha: float):
    token_mask = (mx.arange(h.shape[1]) == (h.shape[1] - 1)).astype(h.dtype)[None, :, None]
    return h + (alpha * vector[None, None, :] * token_mask)


def _normalize_steering(steering: SteeringSpec | None) -> list[ResidualSteeringSpec]:
    if steering is None:
        return []
    if isinstance(steering, ResidualSteeringSpec):
        return [steering] if steering.alpha != 0 else []
    return [spec for spec in steering if spec.alpha != 0]


def _add_steering_at_layer(mx: Any, h: Any, specs: list[ResidualSteeringSpec], layer: int):
    for spec in specs:
        if spec.layer == layer:
            vector = _torch_direction_to_mlx(mx, spec.direction, h.dtype)
            h = _add_to_last_token(mx, h, vector, spec.alpha)
    return h


def _forward_logits_with_steering(model: Any, token_ids: list[int], steering: SteeringSpec | None = None):
    mx = _import_mlx_core()
    n_layers = _n_layers(model)
    specs = _normalize_steering(steering)
    for spec in specs:
        if spec.layer < 0 or spec.layer > n_layers:
            raise ValueError(f"MLX residual steering layer must be in [0, {n_layers}], got {spec.layer}")

    inputs = mx.array([token_ids], dtype=mx.int32)
    h = model.model.embed_tokens(inputs)
    h = _add_steering_at_layer(mx, h, specs, 0)

    cache = [None] * n_layers
    fa_idx = getattr(model.model, "fa_idx", 0)
    fa_mask = _attention_mask(mx, h, cache[fa_idx])
    swa_idx = getattr(model.model, "swa_idx", None)
    swa_mask = None
    if swa_idx is not None:
        swa_mask = _attention_mask(
            mx,
            h,
            cache[swa_idx],
            window_size=getattr(model.model, "sliding_window", None),
        )

    for idx, (layer, layer_cache) in enumerate(zip(model.model.layers, cache), start=1):
        mask = swa_mask if getattr(layer, "use_sliding", False) else fa_mask
        try:
            h = layer(h, mask, cache=layer_cache)
        except TypeError as exc:
            if "cache" not in str(exc):
                raise
            h = layer(h, mask, layer_cache)
        if idx < n_layers:
            h = _add_steering_at_layer(mx, h, specs, idx)

    h = model.model.norm(h)
    h = _add_steering_at_layer(mx, h, specs, n_layers)
    logits = model.lm_head(h)
    mx.eval(logits)
    return logits


def _eos_token_ids(tokenizer: Any) -> set[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(x) for x in eos}
    return {int(eos)}


def _decode_token_ids(tokenizer: Any, ids: list[int]) -> str:
    if not ids:
        return ""
    return tokenizer.decode(ids)


def generate_greedy_with_steering(
    model: Any,
    tokenizer: Any,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    steering: SteeringSpec | None = None,
    max_generation_seconds: float | None = None,
) -> str:
    mx = _import_mlx_core()
    generated: list[int] = []
    eos_ids = _eos_token_ids(tokenizer)
    ids = list(token_ids)
    deadline = None
    if max_generation_seconds is not None and max_generation_seconds > 0:
        deadline = time.monotonic() + max_generation_seconds
    try:
        for _ in range(max_new_tokens):
            if deadline is not None and time.monotonic() >= deadline:
                break
            logits = _forward_logits_with_steering(model, ids, steering=steering)
            next_id = int(np.array(mx.argmax(logits[0, -1, :], axis=-1)))
            if next_id in eos_ids:
                break
            generated.append(next_id)
            ids.append(next_id)
    finally:
        clear_mlx_cache()
    return _decode_token_ids(tokenizer, generated).strip()


def last_token_residuals(model: Any, token_ids: list[int], layers: list[int]) -> dict[int, torch.Tensor]:
    """Last-token residual at each requested layer; indexing matches HF ``output_hidden_states``."""
    mx = _import_mlx_core()
    n_layers = _n_layers(model)
    wanted = {int(layer) for layer in layers}
    bad = sorted(layer for layer in wanted if layer < 0 or layer > n_layers)
    if bad:
        raise ValueError(f"requested layers outside [0, {n_layers}]: {bad}")

    inputs = mx.array([token_ids], dtype=mx.int32)
    h = model.model.embed_tokens(inputs)
    cache = [None] * n_layers
    fa_idx = getattr(model.model, "fa_idx", 0)
    fa_mask = _attention_mask(mx, h, cache[fa_idx])
    swa_idx = getattr(model.model, "swa_idx", None)
    swa_mask = None
    if swa_idx is not None:
        swa_mask = _attention_mask(
            mx,
            h, cache[swa_idx], window_size=getattr(model.model, "sliding_window", None)
        )

    out: dict[int, torch.Tensor] = {}
    if 0 in wanted:
        out[0] = _last_token_to_torch(mx, h)

    for idx, (layer, layer_cache) in enumerate(zip(model.model.layers, cache), start=1):
        mask = swa_mask if getattr(layer, "use_sliding", False) else fa_mask
        try:
            h = layer(h, mask, cache=layer_cache)
        except TypeError as exc:
            # Older mlx-lm blocks take the cache positionally; only fall back for that signature
            # mismatch, never for a genuine TypeError raised inside the block.
            if "cache" not in str(exc):
                raise
            h = layer(h, mask, layer_cache)
        if idx in wanted and idx < n_layers:
            out[idx] = _last_token_to_torch(mx, h)

    if n_layers in wanted:
        h = model.model.norm(h)
        out[n_layers] = _last_token_to_torch(mx, h)

    clear_mlx_cache()
    return out


class MlxActivationPipeline:
    """Backend-agnostic pipeline implemented on top of an mlx-lm model."""

    def __init__(self, model: Any, tokenizer: Any, meta: PipelineMeta, max_length: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.meta = meta
        self.max_length = max_length

    @classmethod
    def load(cls, model_name: str, *, max_length: int = 2048, dtype: str = "float16") -> "MlxActivationPipeline":
        model, tokenizer, meta = load_mlx_model(model_name, dtype=dtype)
        return cls(model=model, tokenizer=tokenizer, meta=meta, max_length=max_length)

    def capture(
        self,
        messages: list[dict],
        layers: list[int],
        *,
        add_generation_prompt: bool,
    ) -> dict[int, torch.Tensor]:
        ids = chat_token_ids(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            max_length=self.max_length,
        )
        return last_token_residuals(self.model, ids, layers)

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        max_generation_seconds: float | None = None,
        steering: SteeringSpec | None = None,
    ) -> str:
        if _normalize_steering(steering):
            if temperature != 0:
                raise ValueError("MLX steered generation currently supports deterministic temperature=0 only")
            ids = chat_token_ids(
                self.tokenizer, messages, add_generation_prompt=True, max_length=self.max_length
            )
            return generate_greedy_with_steering(
                self.model,
                self.tokenizer,
                ids,
                max_new_tokens=max_new_tokens,
                steering=steering,
                max_generation_seconds=max_generation_seconds,
            )

        mx, _, _, stream_generate, make_sampler = _import_mlx()
        if seed is not None:
            mx.random.seed(seed)
        ids = chat_token_ids(
            self.tokenizer, messages, add_generation_prompt=True, max_length=self.max_length
        )
        sampler = make_sampler(temp=temperature, top_p=top_p)
        deadline = None
        if max_generation_seconds is not None and max_generation_seconds > 0:
            deadline = time.monotonic() + max_generation_seconds
        chunks = []
        try:
            for response in stream_generate(
                self.model, self.tokenizer, ids, max_tokens=max_new_tokens, sampler=sampler
            ):
                chunks.append(response.text)
                if deadline is not None and time.monotonic() >= deadline:
                    break
        finally:
            clear_mlx_cache()
        return "".join(chunks).strip()
