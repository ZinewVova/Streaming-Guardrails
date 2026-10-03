"""Response tokenization shared by guard adapters."""

from __future__ import annotations

from typing import Any

from streamguard_bench.streaming import TokenizedResponse


def tokenize_response(tokenizer: Any, response: str) -> TokenizedResponse:
    """Tokenize once and preserve the character end offset of every token."""

    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    token_ids = tuple(int(item) for item in flatten(encoded["input_ids"]))
    raw_offsets = encoded.get("offset_mapping")
    if raw_offsets is None:
        end_characters = _decode_prefix_offsets(tokenizer, token_ids, response)
    else:
        end_characters = tuple(int(end) for _, end in _normalize_offsets(raw_offsets))

    decoded = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != response:
        raise ValueError("Tokenizer did not reproduce the unchanged response text")
    if end_characters and end_characters[-1] != len(response):
        raise ValueError("Token offsets do not cover the complete response")
    return TokenizedResponse(token_ids=token_ids, end_characters=end_characters)


def flatten(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    while (
        isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    return list(value)


def _decode_prefix_offsets(
    tokenizer: Any, token_ids: tuple[int, ...], response: str
) -> tuple[int, ...]:
    offsets = []
    for end in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(
            list(token_ids[:end]),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not response.startswith(decoded):
            raise ValueError("Decoded token prefix is not a prefix of the source response")
        offsets.append(len(decoded))
    return tuple(offsets)


def _normalize_offsets(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    while (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
        and value[0]
        and isinstance(value[0][0], (list, tuple))
    ):
        value = value[0]
    return [(int(start), int(end)) for start, end in value]
