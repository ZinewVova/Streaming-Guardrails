"""Adapter for the Streaming Content Monitor (SCM) token scorer.

SCM has no streaming API of its own. The backbone is causal, so the adapter keeps its
key-value cache between calls and scores one new token at a time; the result equals a
single forward pass over the whole response.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from streamguard_bench.contracts import PromptDecision, ResponseTokenDecision
from streamguard_bench.guards.tokenization import flatten, tokenize_response
from streamguard_bench.streaming import TokenizedResponse

# SCM was trained on `prompt + "\n\n" + response` with no chat template.
PROMPT_SEPARATOR = "\n\n"
PROMPT_NOT_EVALUATED = "not_evaluated"


class TokenScorer(Protocol):
    """Incremental scorer: P(harmful) for each new token given everything before it."""

    device: str

    def reset(self, prompt_ids: list[int]) -> None: ...

    def step(self, token_id: int) -> float: ...

    def close(self) -> None: ...


class SCMAdapter:
    """Turn SCM token scores into the benchmark's per-token decisions."""

    def __init__(
        self,
        model_id_or_path: str = "liyang-ict/SCM-0.5B",
        *,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device: str | None = None,
        threshold: float = 0.7,
        scorer: TokenScorer | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if (scorer is None) != (tokenizer is None):
            raise ValueError("scorer and tokenizer must be supplied together")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie strictly between 0 and 1")

        self.model_id = model_id_or_path
        self.tokenizer_id = model_id_or_path
        self.model_revision = revision
        self.tokenizer_revision = tokenizer_revision or revision
        self.threshold = threshold
        self._started = False

        if scorer is not None:
            self.scorer = scorer
            self.tokenizer = tokenizer
        else:
            self.scorer, self.tokenizer = _load_runtime(
                model_id_or_path,
                revision=revision,
                tokenizer_revision=self.tokenizer_revision,
                device=device,
            )
        self.device = self.scorer.device

    def tokenize_response(self, response: str) -> TokenizedResponse:
        return tokenize_response(self.tokenizer, response)

    def start(
        self, prompt: str, *, trace_id: str = "", query_ground_truth: str = ""
    ) -> PromptDecision:
        """Feed the prompt as context. SCM does not moderate prompts, so none is judged."""

        self.close()
        prompt_ids = flatten(
            self.tokenizer(prompt + PROMPT_SEPARATOR, add_special_tokens=False)["input_ids"]
        )
        started = time.perf_counter()
        self.scorer.reset([int(item) for item in prompt_ids])
        self._started = True
        return PromptDecision(
            trace_id=trace_id,
            query_ground_truth=query_ground_truth,
            risk_label=PROMPT_NOT_EVALUATED,
            risk_categories=(),
            confidence=None,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def score_token(
        self,
        *,
        token_id: int,
        token_index: int,
        end_character: int,
    ) -> ResponseTokenDecision:
        """Score exactly one new assistant token in the active stream."""

        if not self._started:
            raise RuntimeError("start(prompt) must be called before score_token")
        started = time.perf_counter()
        score = self.scorer.step(token_id)
        latency_ms = (time.perf_counter() - started) * 1000
        return ResponseTokenDecision(
            token_index=token_index,
            token_id=token_id,
            end_character=end_character,
            risk_label="unsafe" if score >= self.threshold else "safe",
            confidence=score,
            latency_ms=latency_ms,
            unsafe_score=score,
            safe_score=1.0 - score,
        )

    def close(self) -> None:
        if self._started:
            self.scorer.close()
            self._started = False

    def __enter__(self) -> SCMAdapter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class TorchTokenScorer:
    """Run the SCM backbone with a key-value cache and read the token head."""

    def __init__(self, model: Any, *, device: str) -> None:
        import torch

        self._torch = torch
        self.model = model
        self.device = device
        self._past: Any | None = None

    def reset(self, prompt_ids: list[int]) -> None:
        self._past = None
        with self._torch.inference_mode():
            self._forward(prompt_ids)

    def step(self, token_id: int) -> float:
        if self._past is None:
            raise RuntimeError("reset(prompt_ids) must be called before step")
        with self._torch.inference_mode():
            logits = self.model.token_classifier(self._forward([token_id])[:, -1])
            return float(self._torch.softmax(logits.float(), dim=-1)[0, 1])

    def score_sequence(self, prompt_ids: list[int], response_ids: list[int]) -> list[float]:
        """Score a whole response in one pass; equals repeated `step` calls."""

        self._past = None
        with self._torch.inference_mode():
            hidden = self._forward(prompt_ids + response_ids)[:, len(prompt_ids) :]
            logits = self.model.token_classifier(hidden)
            scores = self._torch.softmax(logits.float(), dim=-1)[0, :, 1].tolist()
        self._past = None
        return scores

    def close(self) -> None:
        self._past = None

    def _forward(self, token_ids: list[int]) -> Any:
        ids = self._torch.tensor([token_ids], dtype=self._torch.long, device=self.device)
        output = self.model.model(input_ids=ids, past_key_values=self._past, use_cache=True)
        self._past = output.past_key_values
        return output.last_hidden_state


def scm_model_class() -> Any:
    """Build the SCM architecture locally instead of executing the repository's code.

    The published module cannot be imported through `trust_remote_code` because of the dot
    in the repository name, and the architecture is only a Qwen2 backbone with two linear
    heads. Dropout from the original token head is omitted: it is inactive at inference.
    """

    from torch import nn
    from transformers import Qwen2Model, Qwen2PreTrainedModel

    class SCMForTokenScoring(Qwen2PreTrainedModel):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.model = Qwen2Model(config)
            self.token_classifier = nn.Linear(config.hidden_size, config.num_token_labels)
            # Unused at inference; kept so every checkpoint weight has a destination.
            self.sequence_classifier = nn.Linear(
                config.hidden_size, config.num_sequence_labels, bias=False
            )
            self.post_init()

    return SCMForTokenScoring


def _load_runtime(
    model_id: str,
    *,
    revision: str | None,
    tokenizer_revision: str | None,
    device: str | None,
) -> tuple[TorchTokenScorer, Any]:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError('SCM requires: python -m pip install -e ".[models]"') from error

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer_kwargs = {"revision": tokenizer_revision} if tokenizer_revision else {}
    model_kwargs: dict[str, Any] = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    # float32: in bfloat16 token scores drift by up to 0.03 between cached and one-pass
    # scoring, enough to move a token across the decision threshold.
    model, loading = scm_model_class().from_pretrained(
        model_id, torch_dtype=torch.float32, output_loading_info=True, **model_kwargs
    )
    mismatched = {key: value for key, value in loading.items() if value}
    if mismatched:
        raise RuntimeError(f"SCM checkpoint does not match the local architecture: {mismatched}")
    model.to(device)
    model.eval()
    return TorchTokenScorer(model, device=device), tokenizer
