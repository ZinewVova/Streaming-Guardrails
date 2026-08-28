"""Shared, model-agnostic data contracts used across the project."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefixAnnotation:
    """One labeled sentence-boundary prefix of a complete response."""

    prefix_index: int
    end_character: int
    end_byte: int
    end_sentence: int
    binary_label: str | None
    original_label: str
    harm_categories: tuple[str, ...]
    source_rows: tuple[int, ...]


@dataclass(frozen=True)
class HarmOnset:
    """Interval containing the first unsafe content in several coordinate systems."""

    lower_character: int
    upper_character: int
    lower_byte: int
    upper_byte: int
    sentence_index: int
    lower_qwen_token: int | None
    upper_qwen_token: int | None
    source: str


@dataclass(frozen=True)
class NormalizedTrace:
    """A full StreamSafe response with ordered prefix annotations and provenance."""

    trace_id: str
    source_split: str
    prompt: str
    response: str
    binary_label: str | None
    original_label: str
    harm_categories: tuple[str, ...]
    prefix_annotations: tuple[PrefixAnnotation, ...]
    harm_onset: HarmOnset | None
    language: str
    source_rows: tuple[int, ...]
    dataset_revision: str
    exclusion_reason: str | None
