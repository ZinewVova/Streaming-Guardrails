"""Character-to-token coordinate mapping without loading guard-model weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResponseTokenization:
    token_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    roundtrip_ok: bool


class QwenTokenCoordinateMapper:
    """Map response spans through the pinned Qwen3Guard tokenizer and chat template."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, repository: str, revision: str) -> QwenTokenCoordinateMapper:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            repository,
            revision=revision,
            trust_remote_code=True,
            use_fast=True,
        )
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("Qwen tokenizer must be fast to expose offset mappings")
        return cls(tokenizer)

    def tokenize_response(self, prompt: str, response: str) -> ResponseTokenization:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        response_start = rendered.rfind(response)
        if response_start < 0:
            raise ValueError("Response text was not found in the rendered Qwen chat template")
        response_end = response_start + len(response)
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        response_ids: list[int] = []
        response_offsets: list[tuple[int, int]] = []
        for token_id, (start, end) in zip(
            encoded["input_ids"], encoded["offset_mapping"], strict=True
        ):
            if end <= response_start or start >= response_end or start == end:
                continue
            response_ids.append(int(token_id))
            response_offsets.append(
                (max(0, int(start) - response_start), min(len(response), int(end) - response_start))
            )
        decoded = self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return ResponseTokenization(
            token_ids=tuple(response_ids),
            offsets=tuple(response_offsets),
            roundtrip_ok=decoded == response,
        )

    def map_interval(
        self, prompt: str, response: str, lower_character: int, upper_character: int
    ) -> tuple[int, int]:
        if not 0 <= lower_character <= upper_character <= len(response):
            raise ValueError("Character interval lies outside the response")
        tokenization = self.tokenize_response(prompt, response)
        if not tokenization.roundtrip_ok:
            raise ValueError("Qwen response-token roundtrip changed the source text")
        lower_token = _first_overlapping_token(tokenization.offsets, lower_character)
        upper_token = sum(start < upper_character for start, _ in tokenization.offsets)
        return lower_token, upper_token


def _first_overlapping_token(offsets: tuple[tuple[int, int], ...], position: int) -> int:
    for index, (_, end) in enumerate(offsets):
        if end > position:
            return index
    return len(offsets)
