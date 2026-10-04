"""Adapter for Qwen3Guard-Gen, a guard that reads a finished dialogue rather than a stream.

Gen has no streaming API and no token head. To fit the benchmark it is called once per
buffer checkpoint of the configured modes, with the query and the response up to that
point. Tokens between checkpoints are not scored and stay `safe` with no confidence; the
replay then shows each mode only the decisions made at its own checkpoints (see
`restrict_to_checkpoints`). Listing the `token` mode scores every token, which costs one
model call per token.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any, Protocol

from streamguard_bench.contracts import PromptDecision, ResponseTokenDecision
from streamguard_bench.guards.tokenization import tokenize_response
from streamguard_bench.streaming import BufferMode, TokenizedResponse, checkpoint_tokens

LABELS = ("safe", "controversial", "unsafe")
LABEL_WORDS = (" Safe", " Controversial", " Unsafe")
# The model answers "Safety: <label>"; the label is read from the logits after this prefix.
ANSWER_PREFIX = "Safety:"


class DialogueClassifier(Protocol):
    """Probabilities of (safe, controversial, unsafe) for one dialogue."""

    device: str

    def classify(self, messages: list[dict[str, str]]) -> tuple[float, float, float]: ...


class Qwen3GuardGenAdapter:
    checkpoint_scoring = True

    def __init__(
        self,
        model_id_or_path: str = "Qwen/Qwen3Guard-Gen-0.6B",
        *,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device: str | None = None,
        modes: Iterable[str] = tuple(item.value for item in BufferMode if item.value != "token"),
        max_sentence_tokens: int = 128,
        classifier: DialogueClassifier | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if (classifier is None) != (tokenizer is None):
            raise ValueError("classifier and tokenizer must be supplied together")

        self.model_id = model_id_or_path
        self.tokenizer_id = model_id_or_path
        self.model_revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.modes = tuple(BufferMode(item) for item in modes)
        self.max_sentence_tokens = max_sentence_tokens
        self._query = ""
        self._response = ""
        self._scored: set[int] = set()

        if classifier is not None:
            self.classifier, self.tokenizer = classifier, tokenizer
        else:
            self.classifier, self.tokenizer = _load_runtime(
                model_id_or_path,
                revision=revision,
                tokenizer_revision=self.tokenizer_revision,
                device=device,
            )
        self.device = self.classifier.device

    def tokenize_response(self, response: str) -> TokenizedResponse:
        tokenized = tokenize_response(self.tokenizer, response)
        self._response = response
        if BufferMode.TOKEN in self.modes:
            self._scored = set(range(1, len(tokenized.token_ids) + 1))
        else:
            self._scored = checkpoint_tokens(
                tokenized.end_characters, response, self.modes, self.max_sentence_tokens
            )
        return tokenized

    def start(
        self, prompt: str, *, trace_id: str = "", query_ground_truth: str = ""
    ) -> PromptDecision:
        self._query = prompt
        started = time.perf_counter()
        scores = self.classifier.classify([{"role": "user", "content": prompt}])
        label, confidence = _decide(scores)
        return PromptDecision(
            trace_id=trace_id,
            query_ground_truth=query_ground_truth,
            risk_label=label,
            risk_categories=(),
            confidence=confidence,
            latency_ms=(time.perf_counter() - started) * 1000,
            safe_score=scores[0],
            controversial_score=scores[1],
            unsafe_score=scores[2],
        )

    def score_token(
        self, *, token_id: int, token_index: int, end_character: int
    ) -> ResponseTokenDecision:
        prefix = self._response[:end_character]
        if token_index not in self._scored or not prefix.strip():
            return ResponseTokenDecision(token_index, token_id, end_character, "safe")
        started = time.perf_counter()
        scores = self.classifier.classify(
            [
                {"role": "user", "content": self._query},
                {"role": "assistant", "content": prefix},
            ]
        )
        label, confidence = _decide(scores)
        return ResponseTokenDecision(
            token_index=token_index,
            token_id=token_id,
            end_character=end_character,
            risk_label=label,
            confidence=confidence,
            latency_ms=(time.perf_counter() - started) * 1000,
            safe_score=scores[0],
            controversial_score=scores[1],
            unsafe_score=scores[2],
        )

    def close(self) -> None:
        """Nothing is held between traces: every call reads a complete dialogue."""

    def __enter__(self) -> Qwen3GuardGenAdapter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _decide(scores: tuple[float, float, float]) -> tuple[str, float]:
    index = max(range(len(LABELS)), key=lambda item: scores[item])
    return LABELS[index], scores[index]


class TorchDialogueClassifier:
    """Read the label word after `Safety:` from the logits of one forward pass."""

    def __init__(self, model: Any, tokenizer: Any, *, device: str) -> None:
        import torch

        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._label_ids = [
            int(tokenizer.encode(word, add_special_tokens=False)[0]) for word in LABEL_WORDS
        ]
        if len(set(self._label_ids)) != len(LABELS):
            raise RuntimeError("The first tokens of Safe, Controversial and Unsafe coincide")

    def classify(self, messages: list[dict[str, str]]) -> tuple[float, float, float]:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False) + ANSWER_PREFIX
        inputs = self.tokenizer([text], return_tensors="pt", add_special_tokens=False).to(
            self.device
        )
        with self._torch.inference_mode():
            try:
                output = self.model(**inputs, logits_to_keep=1)
            except TypeError:
                output = self.model(**inputs)
        logits = output.logits[0, -1].float()[self._label_ids]
        probabilities = self._torch.softmax(logits, dim=-1).tolist()
        return probabilities[0], probabilities[1], probabilities[2]


def _load_runtime(
    model_id: str, *, revision: str | None, tokenizer_revision: str | None, device: str | None
) -> tuple[TorchDialogueClassifier, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError('Qwen3Guard-Gen requires: pip install -e ".[models]"') from error

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(device, torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return TorchDialogueClassifier(model, tokenizer, device=device), tokenizer
