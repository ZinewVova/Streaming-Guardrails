"""Sentence-prefix moderation on the prepared dataset's token coordinates."""

from __future__ import annotations

import re
import time
from typing import Any

from streamguard_bench.contracts import PromptDecision, ResponseTokenDecision
from streamguard_bench.guards.tokenization import tokenize_response

COORDINATE_TOKENIZER = "Qwen/Qwen3Guard-Stream-0.6B"
COORDINATE_REVISION = "419364a715de9840d47b1457982f64ff37f90ed4"
BOUNDARY = re.compile(r"[.!?。！？]")


class PrefixGuardAdapter:
    """Emit one observation at a sentence boundary; retain unscored token positions.

    Tokenization sees the complete recorded response, but inference receives only
    the prefix through the current token. The final incomplete sentence is checked.
    Generation tokenization is independent of benchmark coordinate tokenization.
    """

    def __init__(
        self,
        model_id_or_path: str,
        *,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 512,
        quantization: str | None = None,
        generator: Any = None,
        tokenizer: Any = None,
    ) -> None:
        if (generator is None) != (tokenizer is None):
            raise ValueError("generator and tokenizer must be supplied together")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if quantization not in (None, "nf4"):
            raise ValueError("quantization must be None or nf4")
        self.model_id = model_id_or_path
        self.model_revision = revision
        self.tokenizer_id = COORDINATE_TOKENIZER
        self.tokenizer_revision = COORDINATE_REVISION
        self.inference_settings = {
            "adapter": type(self).__name__,
            "max_new_tokens": max_new_tokens,
            "generation_tokenizer_revision": tokenizer_revision or revision,
            "schedule": "sentence_terminal_and_final_v1",
        }
        if quantization is not None:
            self.inference_settings["quantization"] = quantization
        if generator is None:
            generator = HFGenerator(
                model_id_or_path,
                revision=revision,
                tokenizer_revision=tokenizer_revision or revision,
                device=device,
                quantization=quantization,
            )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                COORDINATE_TOKENIZER,
                revision=COORDINATE_REVISION,
            )
        self.generator, self.tokenizer = generator, tokenizer
        self.device = generator.device
        self.max_new_tokens = max_new_tokens
        self._started = False
        self._response: str | None = None

    def tokenize_response(self, response: str):
        tokenized = tokenize_response(self.tokenizer, response)
        self._response, self._tokenized = response, tokenized
        return tokenized

    def start(self, prompt: str, *, trace_id: str = "", query_ground_truth: str = ""):
        self._prompt = prompt
        self._index = self._end = 0
        self._started = True
        started = time.perf_counter()
        label, raw = self.score_prompt(prompt)
        return PromptDecision(
            trace_id,
            query_ground_truth,
            label,
            (),
            None,
            (time.perf_counter() - started) * 1000,
            raw_output=raw,
        )

    def score_prompt(self, prompt: str) -> tuple[str, str | None]:
        return "not_evaluated", None

    def score_token(self, *, token_id: int, token_index: int, end_character: int):
        if not self._started or self._response is None:
            raise RuntimeError("tokenize_response and start must precede score_token")
        if token_index != self._index + 1:
            raise ValueError("Tokens must be scored exactly once in order")
        expected = self._tokenized
        if (token_id, end_character) != (
            expected.token_ids[self._index],
            expected.end_characters[self._index],
        ):
            raise ValueError("Token does not match the prepared response")
        final = token_index == len(expected.token_ids)
        evaluated = final or bool(BOUNDARY.search(self._response[self._end : end_character]))
        self._index, self._end = token_index, end_character
        label, raw = "safe", None
        elapsed = 0.0
        if evaluated:
            started = time.perf_counter()
            label, raw = self.score_prefix(self._prompt, self._response[:end_character])
            elapsed = (time.perf_counter() - started) * 1000
        return ResponseTokenDecision(
            token_index,
            token_id,
            end_character,
            label,
            latency_ms=elapsed,
            evaluated=evaluated,
            raw_output=raw,
        )

    def score_prefix(self, prompt: str, response: str):
        raise NotImplementedError

    def close(self):
        self._started = False
        self._response = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class HFGenerator:
    """Lazy Transformers runtime; no remote Python code or silent truncation."""

    def __init__(self, model_id, *, revision, tokenizer_revision, device, quantization=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision)
        placement = device or "auto"
        kwargs = {}
        if quantization == "nf4":
            from transformers import BitsAndBytesConfig

            if not torch.cuda.is_available() or (device and not device.startswith("cuda")):
                raise ValueError("NF4 benchmark requires a CUDA GPU")
            placement = device or "cuda:0"
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            device_map=placement,
            torch_dtype=torch.float32 if placement == "cpu" else "auto",
            **kwargs,
        ).eval()
        self.device = str(self.model.device)

    def generate(self, instruction: str, *, chat: bool, max_new_tokens: int) -> str:
        import torch

        if chat:
            instruction = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(
            instruction,
            return_tensors="pt",
            add_special_tokens=False,
            return_token_type_ids=False,
        ).to(self.model.device)
        length = inputs["input_ids"].shape[-1]
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and length + max_new_tokens > limit:
            raise ValueError("Guard context limit exceeded; refusing to truncate the prefix")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, length:], skip_special_tokens=True)
