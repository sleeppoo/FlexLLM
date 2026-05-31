from __future__ import annotations

import argparse
from pathlib import Path
import random
import time
from typing import Sequence

from .sampling import SpeculativeDecoder
from .target_vllm import TargetSamplingConfig, VllmTarget, VllmTargetConfig
from .u280_config import U280Paths, VOCAB_SIZE
from .u280_pyxrt import U280DraftModel


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vanilla speculative decoding: Llama-3.3-70B target + U280 Llama-3.2-1B draft")
    parser.add_argument("--prompt", help="single user prompt; omit for interactive chat")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gamma", type=int, default=4, help="number of greedy U280 draft tokens per verification round")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fpga-only", action="store_true", help="run only the U280 draft backend and print drafted token IDs")
    parser.add_argument("--gpu-only", action="store_true", help="run direct vLLM target decoding without the FPGA draft")
    parser.add_argument(
        "--prompt-token-ids",
        help="comma or space separated token IDs for FPGA-only debugging; bypasses all tokenizer/model loading",
    )
    parser.add_argument(
        "--tokenizer-model",
        help="Hugging Face tokenizer to use with --fpga-only --prompt; defaults to --target-model",
    )

    parser.add_argument("--target-model", default="shuyuej/Llama-3.3-70B-Instruct-GPTQ")
    parser.add_argument("--target-quantization", default=None)
    parser.add_argument("--target-dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0, help="0 disables target top-k truncation")
    parser.add_argument("--target-logprobs", type=int, default=VOCAB_SIZE)
    parser.add_argument("--allow-truncated-target", action="store_true")

    parser.add_argument("--device-index", type=int, default=None, help="XRT device index for the U280; omit to probe indices")
    parser.add_argument("--xrt-probe-limit", type=int, default=16, help="number of XRT device indices to probe when --device-index is omitted")
    parser.add_argument("--parameters-dir")
    parser.add_argument("--prefill-xclbin")
    parser.add_argument("--decode-xclbin")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.fpga_only and args.gpu_only:
        raise SystemExit("--fpga-only and --gpu-only cannot be used together")

    rng = random.Random(args.seed)

    paths = _u280_paths_from_args(args)

    if args.fpga_only:
        draft = U280DraftModel(paths=paths, device_index=args.device_index, xrt_probe_limit=args.xrt_probe_limit, greedy=True)
        selected_device = draft.initialize_runtime()
        _run_fpga_only(args, draft, selected_device)
        return

    target = VllmTarget(
        VllmTargetConfig(
            model=args.target_model,
            quantization=args.target_quantization,
            dtype=args.target_dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            logprobs=args.target_logprobs,
            require_full_logprobs=not args.allow_truncated_target,
        ),
        TargetSamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k or None,
        ),
    )

    tokenizer = target.tokenizer()
    eos_ids = _eos_ids(tokenizer)

    if args.prompt is not None:
        token_ids = _chat_token_ids(tokenizer, args.system_prompt, args.prompt)
        if args.gpu_only:
            generated_ids, elapsed = _run_gpu_only(target, token_ids, args.max_new_tokens, eos_ids)
            print(tokenizer.decode(generated_ids, skip_special_tokens=True))
            _print_gpu_profile(len(generated_ids), elapsed)
            return

        draft = U280DraftModel(paths=paths, device_index=args.device_index, xrt_probe_limit=args.xrt_probe_limit, greedy=True)
        decoder = SpeculativeDecoder(draft=draft, target=target, gamma=args.gamma, rng=rng)
        result = decoder.generate(token_ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos_ids)
        print(tokenizer.decode(result.token_ids, skip_special_tokens=True))
        _print_spec_profile(result.stats)
        return

    if not args.gpu_only:
        draft = U280DraftModel(paths=paths, device_index=args.device_index, xrt_probe_limit=args.xrt_probe_limit, greedy=True)
        decoder = SpeculativeDecoder(draft=draft, target=target, gamma=args.gamma, rng=rng)

    conversation: list[dict[str, str]] = [{"role": "system", "content": args.system_prompt}]
    mode = "GPU-only Chat" if args.gpu_only else "Speculative Chat"
    print(f"=== {mode} (type 'quit' or 'exit' to stop) ===")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break

        conversation.append({"role": "user", "content": user_input})
        token_ids = _apply_chat_template(tokenizer, conversation)
        if args.gpu_only:
            generated_ids, elapsed = _run_gpu_only(target, token_ids, args.max_new_tokens, eos_ids)
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            result = decoder.generate(token_ids, max_new_tokens=args.max_new_tokens, eos_token_ids=eos_ids)
            generated_ids = result.token_ids
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": text})
        print(f"\nAssistant: {text}\n")
        if args.gpu_only:
            _print_gpu_profile(len(generated_ids), elapsed)
        else:
            _print_spec_profile(result.stats)
        print()


def _u280_paths_from_args(args: argparse.Namespace) -> U280Paths:
    paths = U280Paths.default()
    if args.parameters_dir:
        paths = U280Paths(paths.model_dir, Path(args.parameters_dir), paths.prefill_xclbin, paths.decode_xclbin)
    if args.prefill_xclbin:
        paths = U280Paths(paths.model_dir, paths.parameters_dir, Path(args.prefill_xclbin), paths.decode_xclbin)
    if args.decode_xclbin:
        paths = U280Paths(paths.model_dir, paths.parameters_dir, paths.prefill_xclbin, Path(args.decode_xclbin))
    return paths


def _run_fpga_only(args: argparse.Namespace, draft: U280DraftModel, selected_device: int) -> None:
    tokenizer = None
    if args.prompt_token_ids:
        token_ids = _parse_token_ids(args.prompt_token_ids)
    elif args.prompt is not None:
        tokenizer_model = args.tokenizer_model or args.target_model
        tokenizer = _load_hf_tokenizer(tokenizer_model)
        token_ids = _chat_token_ids(tokenizer, args.system_prompt, args.prompt)
    else:
        raise SystemExit("--fpga-only requires either --prompt-token-ids or --prompt")

    if len(token_ids) < 2:
        raise SystemExit("FPGA-only mode needs at least two prompt token IDs")

    draft_tokens = list(draft.propose(token_ids, args.max_new_tokens))
    draft_ids = [token.token_id for token in draft_tokens]
    print(f"Using XRT device index: {selected_device}")
    print("Draft token IDs:", " ".join(str(token_id) for token_id in draft_ids))
    if tokenizer is not None:
        print("Draft text:", tokenizer.decode(draft_ids, skip_special_tokens=True))
    profile = draft.last_profile
    _print_fpga_profile(len(draft_ids), profile.profiled_seconds, profile)


def _run_gpu_only(
    target: VllmTarget,
    token_ids: Sequence[int],
    max_new_tokens: int,
    stop_token_ids: Sequence[int],
) -> tuple[list[int], float]:
    start = time.perf_counter()
    generated_ids = target.generate(token_ids, max_new_tokens=max_new_tokens, stop_token_ids=stop_token_ids)
    return generated_ids, time.perf_counter() - start


def _tokens_per_second(token_count: int, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return token_count / seconds


def _print_gpu_profile(token_count: int, seconds: float) -> None:
    print(
        f"\n[profile] generated tokens: {token_count}, profiled decode time: "
        f"{seconds:.6f}s, end-to-end decode: {_tokens_per_second(token_count, seconds):.2f} tok/s"
    )


def _print_spec_profile(stats: object) -> None:
    print(
        f"\n[profile] generated tokens: {stats.profiled_tokens}, profiled end-to-end time: "
        f"{stats.profiled_seconds:.6f}s, end-to-end decode: {stats.tokens_per_second:.2f} tok/s"
    )
    print(
        f"[profile] target parallel verify + 1 decode overhead: {stats.target_verify_seconds:.6f}s, "
        f"FPGA draft counted: {stats.draft_profile_seconds:.6f}s, "
        f"FPGA prefill kernel: {stats.fpga_prefill_seconds:.6f}s, "
        f"FPGA decode kernel: {stats.fpga_decode_seconds:.6f}s, "
        f"excluded post-first FPGA prefill: {stats.excluded_fpga_prefill_seconds:.6f}s"
    )
    print(
        f"[stats] target calls: {stats.target_calls}, draft calls: {stats.draft_calls}, "
        f"accepted: {stats.accepted_tokens}, rejected: {stats.rejected_tokens}"
    )


def _print_fpga_profile(token_count: int, seconds: float, profile: object) -> None:
    print(
        f"\n[profile] drafted tokens: {token_count}, profiled FPGA time: "
        f"{seconds:.6f}s, FPGA draft: {_tokens_per_second(token_count, seconds):.2f} tok/s"
    )
    print(
        f"[profile] prefill kernel: {profile.prefill_seconds:.6f}s, "
        f"decode kernel: {profile.decode_seconds:.6f}s, "
        f"excluded prefill: {profile.excluded_prefill_seconds:.6f}s"
    )


def _parse_token_ids(raw: str) -> list[int]:
    cleaned = raw.replace("[", " ").replace("]", " ").replace(",", " ")
    token_ids = [int(part) for part in cleaned.split()]
    if not token_ids:
        raise SystemExit("--prompt-token-ids did not contain any token IDs")
    return token_ids


def _load_hf_tokenizer(model_name: str) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _chat_token_ids(tokenizer: object, system_prompt: str, user_prompt: str) -> list[int]:
    return _apply_chat_template(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )


def _apply_chat_template(tokenizer: object, messages: Sequence[dict[str, str]]) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                list(messages),
                add_generation_prompt=True,
                tokenize=True,
            )
        )
    text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\nassistant:"
    return list(tokenizer.encode(text, add_special_tokens=True))


def _eos_ids(tokenizer: object) -> set[int]:
    ids: set[int] = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        if isinstance(eos_token_id, int):
            ids.add(eos_token_id)
        else:
            ids.update(int(token_id) for token_id in eos_token_id)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        for token in ("<|eot_id|>", "<|end_of_text|>"):
            token_id = convert(token)
            if isinstance(token_id, int) and token_id >= 0:
                ids.add(token_id)
    return ids


if __name__ == "__main__":
    main()
