from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class SparseDistribution:
    """A normalized discrete distribution over token IDs."""

    probs: Mapping[int, float]

    def __post_init__(self) -> None:
        total = sum(self.probs.values())
        if total <= 0.0:
            raise ValueError("distribution has no positive mass")
        if abs(total - 1.0) > 1e-4:
            object.__setattr__(
                self, "probs", {int(k): float(v) / total for k, v in self.probs.items() if v > 0.0}
            )

    @classmethod
    def dirac(cls, token_id: int) -> "SparseDistribution":
        return cls({int(token_id): 1.0})

    @classmethod
    def from_logprobs(
        cls,
        logprobs: Mapping[int, float],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        min_tokens_to_keep: int = 1,
    ) -> "SparseDistribution":
        if not logprobs:
            raise ValueError("empty logprob dictionary")

        items = [(int(tok), float(lp)) for tok, lp in logprobs.items() if math.isfinite(float(lp))]
        if not items:
            raise ValueError("logprob dictionary contains no finite entries")

        if temperature == 0.0:
            token_id = max(items, key=lambda item: item[1])[0]
            return cls.dirac(token_id)
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")

        scaled = [(tok, lp / temperature) for tok, lp in items]
        scaled.sort(key=lambda item: item[1], reverse=True)
        if top_k is not None and top_k > 0:
            scaled = scaled[: max(top_k, min_tokens_to_keep)]

        max_lp = scaled[0][1]
        probs = [(tok, math.exp(lp - max_lp)) for tok, lp in scaled]
        total = sum(prob for _, prob in probs)
        probs = [(tok, prob / total) for tok, prob in probs]

        if top_p < 1.0:
            if top_p <= 0.0:
                raise ValueError("top_p must be > 0")
            kept: list[tuple[int, float]] = []
            cumulative = 0.0
            for tok, prob in probs:
                kept.append((tok, prob))
                cumulative += prob
                if cumulative >= top_p and len(kept) >= min_tokens_to_keep:
                    break
            probs = kept

        total = sum(prob for _, prob in probs)
        return cls({tok: prob / total for tok, prob in probs if prob > 0.0})

    def prob(self, token_id: int) -> float:
        return float(self.probs.get(int(token_id), 0.0))

    def sample(self, rng: random.Random) -> int:
        threshold = rng.random()
        cumulative = 0.0
        last_token = None
        for token_id, prob in self.probs.items():
            cumulative += prob
            last_token = token_id
            if threshold < cumulative:
                return int(token_id)
        if last_token is None:
            raise ValueError("cannot sample from empty distribution")
        return int(last_token)

    def positive_difference(self, other: "SparseDistribution") -> "SparseDistribution":
        diff: dict[int, float] = {}
        for token_id, prob in self.probs.items():
            value = prob - other.prob(token_id)
            if value > 0.0:
                diff[token_id] = value
        if not diff:
            # This can only happen when distributions are numerically identical or
            # when `self` was too aggressively truncated. Falling back to `self`
            # keeps generation moving but the caller should avoid this in exact mode.
            diff = dict(self.probs)
        return SparseDistribution(diff)


@dataclass(frozen=True)
class DraftToken:
    token_id: int
    distribution: SparseDistribution

    @classmethod
    def greedy(cls, token_id: int) -> "DraftToken":
        return cls(int(token_id), SparseDistribution.dirac(int(token_id)))

    @property
    def probability(self) -> float:
        return self.distribution.prob(self.token_id)


@dataclass(frozen=True)
class TargetVerification:
    candidate_distributions: Sequence[SparseDistribution]
    bonus_distribution: SparseDistribution | None = None


class DraftModel(Protocol):
    def propose(self, prefix_ids: Sequence[int], num_tokens: int) -> Sequence[DraftToken]:
        ...


class TargetModel(Protocol):
    def verify(self, prefix_ids: Sequence[int], candidate_ids: Sequence[int]) -> TargetVerification:
        ...


@dataclass
class GenerationStats:
    target_calls: int = 0
    draft_calls: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    draft_profile_seconds: float = 0.0
    target_verify_seconds: float = 0.0
    fpga_prefill_seconds: float = 0.0
    fpga_decode_seconds: float = 0.0
    excluded_fpga_prefill_seconds: float = 0.0
    profiled_seconds: float = 0.0
    profiled_tokens: int = 0

    @property
    def tokens_per_second(self) -> float:
        if self.profiled_seconds <= 0.0:
            return 0.0
        return self.profiled_tokens / self.profiled_seconds


@dataclass
class GenerationResult:
    token_ids: list[int]
    stats: GenerationStats


class SpeculativeDecoder:
    """Vanilla speculative decoding over pluggable draft and target backends."""

    def __init__(
        self,
        *,
        draft: DraftModel,
        target: TargetModel,
        gamma: int = 4,
        rng: random.Random | None = None,
    ) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.draft = draft
        self.target = target
        self.gamma = gamma
        self.rng = rng or random.Random()

    def generate(
        self,
        prefix_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: Iterable[int] = (),
    ) -> GenerationResult:
        if max_new_tokens <= 0:
            stats = GenerationStats()
            return self._finish([], stats, max_new_tokens)

        eos_ids = {int(token_id) for token_id in eos_token_ids if token_id is not None}
        generated: list[int] = []
        stats = GenerationStats()

        while len(generated) < max_new_tokens:
            current_prefix = list(prefix_ids) + generated
            draft_budget = min(self.gamma, max_new_tokens - len(generated))
            draft_start = time.perf_counter()
            draft_tokens = list(self.draft.propose(current_prefix, draft_budget))
            draft_wall_seconds = time.perf_counter() - draft_start
            stats.draft_calls += 1
            self._record_draft_profile(stats, draft_wall_seconds)
            if not draft_tokens:
                break

            candidate_ids = [token.token_id for token in draft_tokens]
            verify_start = time.perf_counter()
            verification = self.target.verify(current_prefix, candidate_ids)
            stats.target_verify_seconds += time.perf_counter() - verify_start
            stats.target_calls += 1
            stats.draft_tokens += len(candidate_ids)

            accepted_all = True
            for index, draft_token in enumerate(draft_tokens):
                target_dist = verification.candidate_distributions[index]
                target_prob = target_dist.prob(draft_token.token_id)
                draft_prob = draft_token.probability
                if draft_prob <= 0.0:
                    accept_prob = 1.0
                else:
                    accept_prob = min(1.0, target_prob / draft_prob)

                if self.rng.random() <= accept_prob:
                    generated.append(draft_token.token_id)
                    stats.accepted_tokens += 1
                    if draft_token.token_id in eos_ids or len(generated) >= max_new_tokens:
                        return self._finish(generated, stats, max_new_tokens)
                    continue

                accepted_all = False
                stats.rejected_tokens += 1
                residual = target_dist.positive_difference(draft_token.distribution)
                sampled = residual.sample(self.rng)
                generated.append(sampled)
                if sampled in eos_ids:
                    return self._finish(generated, stats, max_new_tokens)
                break

            if accepted_all and verification.bonus_distribution is not None and len(generated) < max_new_tokens:
                sampled = verification.bonus_distribution.sample(self.rng)
                generated.append(sampled)
                if sampled in eos_ids:
                    break

        return self._finish(generated, stats, max_new_tokens)

    def _record_draft_profile(self, stats: GenerationStats, wall_seconds: float) -> None:
        profile = getattr(self.draft, "last_profile", None)
        if profile is None:
            stats.draft_profile_seconds += wall_seconds
            return

        profiled_seconds = getattr(profile, "profiled_seconds", None)
        if profiled_seconds is None:
            stats.draft_profile_seconds += wall_seconds
        else:
            stats.draft_profile_seconds += float(profiled_seconds)

        stats.fpga_prefill_seconds += float(getattr(profile, "prefill_seconds", 0.0))
        stats.fpga_decode_seconds += float(getattr(profile, "decode_seconds", 0.0))
        stats.excluded_fpga_prefill_seconds += float(getattr(profile, "excluded_prefill_seconds", 0.0))

    @staticmethod
    def _finish(generated: Sequence[int], stats: GenerationStats, max_new_tokens: int) -> GenerationResult:
        token_ids = list(generated[:max_new_tokens])
        stats.profiled_tokens = len(token_ids)
        stats.profiled_seconds = stats.draft_profile_seconds + stats.target_verify_seconds
        return GenerationResult(token_ids, stats)
