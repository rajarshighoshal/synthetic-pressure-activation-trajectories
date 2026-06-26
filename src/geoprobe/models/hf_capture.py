"""PyTorch activation backend (CUDA on servers / RunPod, MPS on Mac, CPU fallback).

Mirrors ``mlx_capture.MlxActivationPipeline`` so the two are interchangeable behind the
``ActivationPipeline`` contract. Uses the shared canonical tokenizer and HF's
``output_hidden_states`` + ``last_token_residual`` extractor.
"""
from __future__ import annotations

from typing import Any
from contextlib import contextmanager

import torch

from geoprobe.models.hooks import last_token_residual
from geoprobe.models.interface import PipelineMeta, ResidualSteeringSpec, SteeringSpec, resolve_torch_dtype
from geoprobe.models.loader import choose_device, cleanup, load_hf_model
from geoprobe.models.tokenization import chat_token_ids


def _transformer_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("cannot locate transformer block list for steering")


def _normalize_steering(steering: SteeringSpec | None) -> list[ResidualSteeringSpec]:
    if steering is None:
        return []
    if isinstance(steering, ResidualSteeringSpec):
        return [steering] if steering.alpha != 0 else []
    return [spec for spec in steering if spec.alpha != 0]


@contextmanager
def _residual_steering_hook(model: Any, steering: SteeringSpec | None):
    specs = _normalize_steering(steering)
    if not specs:
        yield
        return
    blocks = _transformer_layers(model)
    param = next(model.parameters())
    by_layer: dict[int, list[tuple[torch.Tensor, float]]] = {}
    for spec in specs:
        if spec.layer <= 0 or spec.layer > len(blocks):
            raise ValueError(f"HF residual steering layer must be in [1, {len(blocks)}], got {spec.layer}")
        vector = spec.direction.detach().to(param.device, dtype=param.dtype)
        by_layer.setdefault(int(spec.layer), []).append((vector, float(spec.alpha)))

    handles = []
    for layer_index, nudges in by_layer.items():
        def hook(_module, _inputs, output, *, nudges=nudges):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                for vector, alpha in nudges:
                    hidden[:, -1, :] = hidden[:, -1, :] + alpha * vector
                return (hidden, *output[1:])
            hidden = output.clone()
            for vector, alpha in nudges:
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * vector
            return hidden

        handles.append(blocks[layer_index - 1].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


class HfActivationPipeline:
    """Backend-agnostic pipeline implemented on top of a HuggingFace model."""

    def __init__(self, model: Any, tokenizer: Any, meta: PipelineMeta, max_length: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.meta = meta
        self.max_length = max_length

    @classmethod
    def load(
        cls,
        model_key: str,
        *,
        device: str | None = None,
        dtype: str = "float16",
        max_length: int = 2048,
    ) -> "HfActivationPipeline":
        device = device or choose_device()
        model, tokenizer, info = load_hf_model(
            model_key, device=device, dtype=resolve_torch_dtype(dtype)
        )
        effective = str(next(model.parameters()).dtype).replace("torch.", "")
        meta = PipelineMeta(
            name=info["name"],
            backend="hf",
            device=device,
            dtype=effective,
            n_layers=info.get("num_layers"),
            hidden_size=info.get("hidden_size"),
        )
        return cls(model, tokenizer, meta, max_length)

    def _input_ids(self, messages: list[dict], *, add_generation_prompt: bool) -> torch.Tensor:
        ids = chat_token_ids(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            max_length=self.max_length,
        )
        return torch.tensor([ids], dtype=torch.long, device=self.meta.device)

    def capture(
        self,
        messages: list[dict],
        layers: list[int],
        *,
        add_generation_prompt: bool,
    ) -> dict[int, torch.Tensor]:
        input_ids = self._input_ids(messages, add_generation_prompt=add_generation_prompt)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        residual = last_token_residual(output.hidden_states, attention_mask)
        result = {int(layer): residual[int(layer)] for layer in layers}
        del output, residual
        cleanup()
        return result

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
        input_ids = self._input_ids(messages, add_generation_prompt=True)
        if seed is not None:
            torch.manual_seed(seed)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if max_generation_seconds is not None:
            kwargs["max_time"] = max_generation_seconds
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=top_p)
        with _residual_steering_hook(self.model, steering):
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids, attention_mask=torch.ones_like(input_ids), **kwargs
                )
        reply = self.tokenizer.decode(
            generated[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        del generated
        cleanup()
        return reply
