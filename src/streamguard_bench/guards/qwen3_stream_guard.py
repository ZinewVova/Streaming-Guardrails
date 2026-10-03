"""Token-native adapter for the official Qwen3Guard-Stream implementation."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

from streamguard_bench.contracts import PromptDecision, ResponseTokenDecision
from streamguard_bench.guards.tokenization import flatten as _flatten
from streamguard_bench.guards.tokenization import tokenize_response
from streamguard_bench.streaming import TokenizedResponse


class GuardOutputError(RuntimeError):
    """Raised when the model violates the documented output protocol."""


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
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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

        return tokenize_response(self.tokenizer, response)

    def start(
        self, prompt: str, *, trace_id: str = "", query_ground_truth: str = ""
    ) -> PromptDecision:
        """Start a new conversation using the tokenizer's official chat template."""

        self.close()
        self.prompt = prompt
        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=False,
        )
        model_input = self._model_input(_flatten(prompt_ids))
        started = time.perf_counter()
        with self._inference_context():
            parsed = self._moderate(model_input, role="user")
        return PromptDecision(
            trace_id=trace_id,
            query_ground_truth=query_ground_truth,
            risk_label=parsed["risk_label"],
            risk_categories=parsed["risk_categories"],
            confidence=parsed["confidence"],
            latency_ms=(time.perf_counter() - started) * 1000,
            **_score_fields(parsed),
        )

    def score_token(
        self,
        *,
        token_id: int,
        token_index: int,
        end_character: int,
    ) -> ResponseTokenDecision:
        """Score exactly one new assistant token in the active stream."""

        if self.stream_state is None:
            raise RuntimeError("start(prompt) must be called before score_token")
        started = time.perf_counter()
        with self._inference_context():
            parsed = self._moderate(self._model_input([token_id]), role="assistant")
        latency_ms = (time.perf_counter() - started) * 1000

        return ResponseTokenDecision(
            token_index=token_index,
            token_id=token_id,
            end_character=end_character,
            risk_label=parsed["risk_label"],
            risk_categories=parsed["risk_categories"],
            confidence=parsed["confidence"],
            latency_ms=latency_ms,
            **_score_fields(parsed),
        )

    def _moderate(self, model_input: Any, *, role: str) -> dict[str, Any]:
        """Advance the stream and return the parsed decision for the last token.

        With torch available the logits are read from the model's own stream generator, the
        same one `stream_moderate_from_ids` drives, so the label is unchanged while the
        probability of every risk level is kept instead of only the winning one.
        """
        if self._torch is None:
            result, self.stream_state = self.model.stream_moderate_from_ids(
                model_input, role=role, stream_state=self.stream_state
            )
            return parse_guard_output(result)

        token_ids = model_input.to(self.model.device)
        if self.stream_state is None:
            self.stream_state = self.model.stream_generate(token_ids)
            logits = next(self.stream_state)
        else:
            logits = self.stream_state.send(token_ids)
        if role == "user":
            risk_logits, category_logits = logits[2], logits[3]
            risk_map, category_map = self.model.query_risk_level_map, self.model.query_category_map
        else:
            risk_logits, category_logits = logits[0], logits[1]
            risk_map = self.model.response_risk_level_map
            category_map = self.model.response_category_map
        risk_probs = self._torch.softmax(risk_logits[0, -1].float(), dim=-1)
        risk_index = int(risk_probs.argmax())
        parsed = parse_guard_output(
            {
                "risk_level": [risk_map[risk_index]],
                "risk_prob": [float(risk_probs[risk_index])],
                "category": [category_map[int(category_logits[0, -1].argmax())]],
            }
        )
        parsed["scores"] = {
            str(risk_map[index]).lower(): float(value) for index, value in enumerate(risk_probs)
        }
        return parsed

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


def _score_fields(parsed: dict[str, Any]) -> dict[str, float | None]:
    scores = parsed.get("scores", {})
    return {f"{label}_score": scores.get(label) for label in ("safe", "controversial", "unsafe")}


def _last_required(value: Any, field: str) -> Any:
    if value is None:
        raise GuardOutputError(f"Missing or null {field}")
    if hasattr(value, "detach"):
        value = value.detach().cpu().flatten().tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            raise GuardOutputError(f"Empty {field}")
        return value[-1]
    return value


def _last_optional(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().flatten().tolist()
    if isinstance(value, (list, tuple)):
        return value[-1] if value else None
    return value


def parse_guard_output(result: Any) -> dict[str, Any]:
    """Strict parser shared by prompt and response moderation."""

    if not isinstance(result, dict) or "risk_level" not in result:
        raise GuardOutputError("Missing risk_level")
    raw_label = _last_required(result["risk_level"], "risk_level")
    if raw_label is None or not str(raw_label).strip():
        raise GuardOutputError("Empty risk_level")
    label = str(raw_label).strip().lower()
    if label not in {"safe", "controversial", "unsafe"}:
        raise GuardOutputError(f"Unknown risk_level: {raw_label!r}")
    confidence = _optional_float(_last_optional(result.get("risk_prob")))
    category = _last_optional(result.get("category"))
    categories = () if category in {None, "", "None"} else (str(category),)
    return {"risk_label": label, "risk_categories": categories, "confidence": confidence}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
