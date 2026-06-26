"""Canonical chat tokenization shared by every activation backend.

The single source of truth for turning chat ``messages`` into token ids. Both the PyTorch and MLX
backends call ``chat_token_ids`` so they tokenize the *same* prefix and their captured activations
are directly comparable.

Why this exists: the previous PyTorch path rendered the chat template to text
(``apply_chat_template(tokenize=False)``) and then tokenized that text again, which re-applied the
tokenizer's BOS post-processor and produced a **double** ``<|begin_of_text|>``. The MLX path used
``tokenize=True`` (single BOS). The two backends therefore disagreed by a leading token, shifting
every position. Using ``apply_chat_template(tokenize=True)`` everywhere adds the template's special
tokens exactly once, so a single canonical routine keeps the backends in lock-step.
"""
from __future__ import annotations

from typing import Any


def normalize_token_ids(raw: Any) -> list[int]:
    """Coerce the many shapes ``apply_chat_template`` / ``tokenizer(...)`` can return into ``list[int]``.

    Across transformers / tokenizers versions and across HF vs mlx-lm tokenizer wrappers, the return
    value may be a plain ``list[int]``, a ``tokenizers.Encoding`` (has ``.ids``), a ``BatchEncoding``
    / dict (``["input_ids"]``), or a singly-nested batch (``[[...]]`` or ``[Encoding]``). Normalizing
    in one place keeps the fragility out of every call site.
    """
    obj = raw
    if hasattr(obj, "input_ids"):  # BatchEncoding
        obj = obj.input_ids
    elif isinstance(obj, dict):
        obj = obj["input_ids"]
    if hasattr(obj, "ids"):  # tokenizers.Encoding
        return [int(i) for i in obj.ids]
    seq = list(obj)
    if seq and hasattr(seq[0], "ids"):  # [Encoding]
        return [int(i) for i in seq[0].ids]
    if seq and isinstance(seq[0], (list, tuple)):  # [[ids]]
        seq = list(seq[0])
    return [int(i) for i in seq]


def chat_token_ids(
    tokenizer: Any,
    messages: list[dict],
    *,
    add_generation_prompt: bool,
    max_length: int | None = None,
) -> list[int]:
    """Canonical token ids for ``messages``, identical across backends.

    Uses ``apply_chat_template(tokenize=True)`` so special tokens are added exactly once per the
    model's template (no double BOS). Left-truncates to ``max_length`` so the most recent / final
    tokens — where the last-token readout lives — are always kept.
    """
    raw = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    ids = normalize_token_ids(raw)
    if max_length is not None and len(ids) > max_length:
        ids = ids[-max_length:]
    return ids
