from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .sampling import SparseDistribution, TargetVerification
from .u280_config import VOCAB_SIZE


@dataclass(frozen=True)
class TargetSamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int | None = None
    min_tokens_to_keep: int = 1


@dataclass(frozen=True)
class VllmTargetConfig:
    model: str = "shuyuej/Llama-3.3-70B-Instruct-GPTQ"
    quantization: str | None = None
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    trust_remote_code: bool = False
    logprobs: int = VOCAB_SIZE
    require_full_logprobs: bool = True


class VllmTarget:
    """Target model wrapper that scores draft continuations with vLLM."""

    def __init__(
        self,
        config: VllmTargetConfig | None = None,
        sampling: TargetSamplingConfig | None = None,
    ) -> None:
        self.config = config or VllmTargetConfig()
        self.sampling = sampling or TargetSamplingConfig()

        from vllm import LLM, SamplingParams

        llm_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "quantization": self.config.quantization,
            "dtype": self.config.dtype,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "max_model_len": self.config.max_model_len,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "trust_remote_code": self.config.trust_remote_code,
            "max_logprobs": self.config.logprobs,
        }
        try:
            self.llm = LLM(**llm_kwargs)
        except TypeError:
            llm_kwargs.pop("max_logprobs", None)
            self.llm = LLM(**llm_kwargs)

        self._sampling_params_cls = SamplingParams
        self._score_params = self._make_prompt_score_params(self.config.logprobs)

    def tokenizer(self) -> Any:
        return self.llm.get_tokenizer()

    def generate(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        stop_token_ids: Sequence[int] = (),
    ) -> list[int]:
        if max_new_tokens <= 0:
            return []

        stops = [int(token_id) for token_id in stop_token_ids if token_id is not None]
        params_kwargs: dict[str, Any] = {
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "max_tokens": int(max_new_tokens),
            "detokenize": False,
        }
        if self.sampling.top_k is not None:
            params_kwargs["top_k"] = self.sampling.top_k
        if stops:
            params_kwargs["stop_token_ids"] = stops
        try:
            params = self._sampling_params_cls(**params_kwargs)
        except TypeError:
            params_kwargs.pop("stop_token_ids", None)
            params = self._sampling_params_cls(**params_kwargs)
        output = self._generate_from_token_ids(prompt_token_ids, params)
        generated_outputs = getattr(output, "outputs", None)
        if not generated_outputs:
            raise RuntimeError("vLLM did not return a generated output")
        token_ids = getattr(generated_outputs[0], "token_ids", None)
        if token_ids is None:
            raise RuntimeError("vLLM output did not contain token_ids")
        return [int(token_id) for token_id in token_ids]

    def verify(self, prefix_ids: Sequence[int], candidate_ids: Sequence[int]) -> TargetVerification:
        if not candidate_ids:
            raise ValueError("candidate_ids must not be empty")

        prompt_token_ids = list(prefix_ids) + [int(token_id) for token_id in candidate_ids]
        output = self._score_prompt_token_ids(prompt_token_ids)
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        if prompt_logprobs is None:
            raise RuntimeError("vLLM did not return prompt_logprobs")

        prefix_len = len(prefix_ids)
        candidate_distributions: list[SparseDistribution] = []
        for index in range(len(candidate_ids)):
            raw = prompt_logprobs[prefix_len + index]
            logprobs = self._parse_logprobs(raw)
            self._check_logprob_coverage(logprobs, "target candidate distribution")
            candidate_distributions.append(self._to_distribution(logprobs))

        return TargetVerification(candidate_distributions)

    def _generate_from_token_ids(self, token_ids: Sequence[int], sampling_params: Any) -> Any:
        return self._run_vllm_token_ids(token_ids, sampling_params)

    def _score_prompt_token_ids(self, token_ids: Sequence[int]) -> Any:
        # This vLLM build rejects max_tokens=0, so verification requests one
        # generated token and ignores it. No generated-token logprobs are
        # requested; prompt_logprobs score the draft block in parallel.
        return self._run_vllm_token_ids(token_ids, self._score_params)

    def _run_vllm_token_ids(self, token_ids: Sequence[int], sampling_params: Any) -> Any:
        try:
            from vllm.inputs import TokensPrompt

            prompts = [TokensPrompt(prompt_token_ids=list(token_ids))]
            return self.llm.generate(prompts, sampling_params, use_tqdm=False)[0]
        except Exception as exc:
            # Older vLLM releases used a prompt_token_ids keyword. Retry that API
            # only for signature/API errors, then let real generation failures out.
            if exc.__class__.__name__ not in {"TypeError", "ImportError"}:
                raise
            return self.llm.generate(
                prompts=None,
                sampling_params=sampling_params,
                prompt_token_ids=[list(token_ids)],
                use_tqdm=False,
            )[0]

    def _make_prompt_score_params(self, prompt_logprobs: int) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1,
            "prompt_logprobs": int(prompt_logprobs),
            "detokenize": False,
        }
        return self._sampling_params_cls(**kwargs)

    def _to_distribution(self, logprobs: Mapping[int, float]) -> SparseDistribution:
        return SparseDistribution.from_logprobs(
            logprobs,
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            top_k=self.sampling.top_k,
            min_tokens_to_keep=self.sampling.min_tokens_to_keep,
        )

    def _check_logprob_coverage(self, logprobs: Mapping[int, float], name: str) -> None:
        if self.config.require_full_logprobs and len(logprobs) < VOCAB_SIZE:
            raise RuntimeError(
                f"{name} has {len(logprobs)} logprobs, but exact speculative "
                f"decoding needs the full {VOCAB_SIZE}-token distribution. "
                "Increase VllmTargetConfig.logprobs and ensure vLLM accepts "
                "max_logprobs at model construction."
            )

    @staticmethod
    def _parse_logprobs(raw: Any) -> dict[int, float]:
        if raw is None:
            raise RuntimeError("missing logprob entry")
        parsed: dict[int, float] = {}
        for token_id, value in raw.items():
            if hasattr(value, "logprob"):
                parsed[int(token_id)] = float(value.logprob)
            elif isinstance(value, Mapping) and "logprob" in value:
                parsed[int(token_id)] = float(value["logprob"])
            else:
                parsed[int(token_id)] = float(value)
        return parsed
