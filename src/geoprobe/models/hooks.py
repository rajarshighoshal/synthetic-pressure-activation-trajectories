from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActivationBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    activations: dict[int, torch.Tensor]


def last_token_residual(hidden_states: tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> dict[int, torch.Tensor]:
    positions = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    return {
        layer: state[batch_idx, positions].detach().float().cpu()
        for layer, state in enumerate(hidden_states)
    }
