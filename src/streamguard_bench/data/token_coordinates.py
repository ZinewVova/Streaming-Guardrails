"""Map an exact Unicode character onset into response-only token coordinates."""

from __future__ import annotations

from collections.abc import Sequence


def unsafe_start_token(
    *, unsafe_start_character: int, response: str, offsets: Sequence[tuple[int, int]]
) -> int:
    """Return the 1-based first token intersecting the unsafe suffix."""

    if not 0 <= unsafe_start_character <= len(response):
        raise ValueError("unsafe_start_character is outside the response")
    if unsafe_start_character == len(response):
        return len(offsets) + 1
    for index, (start, end) in enumerate(offsets, start=1):
        if start <= unsafe_start_character < end:
            return index
    raise ValueError("Token offsets do not cover unsafe_start_character")
