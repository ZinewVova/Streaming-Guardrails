"""Token-native adapter for the official Qwen3Guard-Stream implementation."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

from streamguard_bench.streaming import TokenDecision, TokenizedResponse


class Qwen3GuardStreamAdapter:
    """Keep one Qwen3Guard stream state and score only newly generated token IDs."""

    def __init__(
        self,
        model_id_or_path: str = "Qwen/Qwen3Guard-Stream-0.6B",
        *,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device: str | None = None,
        torch_dtype: Any | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if (model is None) != (tokenizer is None):
            raise ValueError("model and tokenizer must be supplied together")

        self.model_id = model_id_or_path
        self.tokenizer_id = model_id_or_path
        self.model_revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.stream_state: Any | None = None
        self.prompt = ""
        self._torch: Any | None = None

        if model is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = device or "test"
        else:
            self._load_runtime(device=device, torch_dtype=torch_dtype)

    def _load_runtime(self, *, device: str | None, torch_dtype: Any | None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                'Qwen3Guard requires: python -m pip install -e ".[models]"'
            ) from error

        self._torch = torch
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if torch_dtype is None:
            if device == "cuda":
                torch_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            elif device == "mps":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32

        tokenizer_kwargs = {"trust_remote_code": True}
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        if self.tokenizer_revision:
            tokenizer_kwargs["revision"] = self.tokenizer_revision
        if self.model_revision:
            model_kwargs["revision"] = self.model_revision

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **tokenizer_kwargs)
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        else:
            self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
            self.model.to(device)
        self.model.eval()

        self.model_revision = self.model_revision or getattr(
            getattr(self.model, "config", None), "_commit_hash", None
        )
        self.tokenizer_revision = self.tokenizer_revision or getattr(
            self.tokenizer, "init_kwargs", {}
        ).get("_commit_hash")

    def tokenize_response(self, response: str) -> TokenizedResponse:
        """Tokenize once and preserve the character end offset of every token."""

        encoded = self.tokenizer(
            response,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        token_ids = tuple(int(item) for item in _flatten(encoded["input_ids"]))
        raw_offsets = encoded.get("offset_mapping")
        if raw_offsets is None:
            end_characters = self._decode_prefix_offsets(token_ids, response)
        else:
            offsets = _normalize_offsets(raw_offsets)
            end_characters = tuple(int(end) for _, end in offsets)

        decoded = self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != response:
            raise ValueError("Qwen tokenizer did not reproduce the unchanged response text")
        if end_characters and end_characters[-1] != len(response):
            raise ValueError("Token offsets do not cover the complete response")
        return TokenizedResponse(token_ids=token_ids, end_characters=end_characters)

    def _decode_prefix_offsets(
        self, token_ids: tuple[int, ...], response: str
    ) -> tuple[int, ...]:
        offsets = []
        for end in range(1, len(token_ids) + 1):
            decoded = self.tokenizer.decode(
                list(token_ids[:end]),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not response.startswith(decoded):
                raise ValueError("Decoded token prefix is not a prefix of the source response")
            offsets.append(len(decoded))
        return tuple(offsets)

    def start(self, prompt: str) -> dict[str, Any]:
        """Start a new conversation using the tokenizer's official chat template."""

        self.close()
        self.prompt = prompt
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=False,
        )
        model_input = self._model_input(_flatten(prompt_ids))
        with self._inference_context():
            result, self.stream_state = self.model.stream_moderate_from_ids(
                model_input,
                role="user",
                stream_state=None,
            )
        return result

    def score_token(
        self,
        *,
        token_id: int,
        token_index: int,
        end_character: int,
    ) -> TokenDecision:
        """Score exactly one new assistant token in the active stream."""

        if self.stream_state is None:
            raise RuntimeError("start(prompt) must be called before score_token")
        started = time.perf_counter()
        with self._inference_context():
            result, self.stream_state = self.model.stream_moderate_from_ids(
                self._model_input([token_id]),
                role="assistant",
                stream_state=self.stream_state,
            )
        latency_ms = (time.perf_counter() - started) * 1000

        label = str(_last(result.get("risk_level"), "safe")).lower()
        if label not in {"safe", "controversial", "unsafe"}:
            raise RuntimeError(f"Qwen3Guard returned an unknown risk level: {label!r}")
        confidence = _optional_float(_last(result.get("risk_prob"), None))
        category = _last(result.get("category"), None)
        categories = () if category in {None, "", "None"} else (str(category),)
        return TokenDecision(
            token_index=token_index,
            token_id=token_id,
            end_character=end_character,
            risk_label=label,
            risk_categories=categories,
            label_confidence=confidence,
            latency_ms=latency_ms,
        )

    def close(self) -> None:
        """Close the official generator state, including after failed traces."""

        if self.stream_state is not None:
            self.model.close_stream(self.stream_state)
            self.stream_state = None

    def _model_input(self, token_ids: list[int]) -> Any:
        if self._torch is None:
            return list(token_ids)
        return self._torch.tensor(token_ids, dtype=self._torch.long, device=self.device)

    def _inference_context(self) -> Any:
        if self._torch is None:
            return nullcontext()
        return self._torch.inference_mode()

    def __enter__(self) -> Qwen3GuardStreamAdapter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _flatten(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(
        value[0], (list, tuple)
    ):
        value = value[0]
    return list(value)


def _normalize_offsets(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(
        value[0], (list, tuple)
    ) and value[0] and isinstance(value[0][0], (list, tuple)):
        value = value[0]
    return [(int(start), int(end)) for start, end in value]


def _last(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if hasattr(value, "detach"):
        value = value.detach().cpu().flatten().tolist()
    if isinstance(value, (list, tuple)):
        return value[-1] if value else default
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
