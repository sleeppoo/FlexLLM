from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from .c_header_parser import load_float_array, load_scale_sum
from .sampling import DraftToken
from .u280_config import (
    DEC_HEAD_PARALLEL,
    DEC_K_CACHE_ELEMS,
    DEC_K_PARALLEL,
    DEC_QKVO_FFN_W_PARALLEL,
    DEC_V_CACHE_ELEMS,
    DEC_V_PARALLEL,
    DECODER_LAYER_NUM,
    HEAD_DIM,
    HIDDEN_DIM,
    INTER_DIM,
    KV_HEAD_NUM,
    KV_HIDDEN_DIM,
    KV_HIDDEN_DIM_PAD,
    MAX_DEC_SEQ_LEN,
    MAX_PRE_SEQ_LEN,
    MAX_SUM_SEQ_LEN,
    PREF_FFN_DOWN_ELEMS,
    PREF_FFN_INTER_ELEMS,
    PREF_FFN_W_BLOCK_NUM,
    PREF_FFN_W_PARALLEL,
    PREF_FFN_W_PARALLEL_READ,
    PREF_IO_ELEMS,
    PREF_K_CACHE_ELEMS,
    PREF_K_PARALLEL,
    PREF_QKVO_ELEMS,
    PREF_QKVO_S_ELEMS,
    PREF_QKVO_W_PARALLEL,
    PREF_QKVO_W_PARALLEL_READ,
    PREF_V_CACHE_ELEMS,
    PREF_V_PARALLEL,
    T_BLOCK_PARALLEL,
    T_QKVO_FFN_BLOCK_PARALLEL,
    TOKEN_PARALLEL,
    U280Paths,
    VOCAB_SIZE,
    VOCAB_SIZE_PAD,
    W_FFN_DOWN_ADDR_BIAS,
    W_FFN_GATE_ADDR_BIAS,
    W_FFN_UP_ADDR_BIAS,
    W_KV_ADDR_BIAS,
    W_O_ADDR_BIAS,
    W_Q_ADDR_BIAS,
    W_QKVO_FFN_SIZE,
    W_S_FFN_DOWN_ADDR_BIAS,
    W_S_FFN_GATE_ADDR_BIAS,
    W_S_FFN_UP_ADDR_BIAS,
    W_S_KV_ADDR_BIAS,
    W_S_O_ADDR_BIAS,
    W_S_Q_ADDR_BIAS,
    W_S_QKVO_FFN_SIZE,
    W_S_VOCAB_ADDR_BIAS,
    W_VOCAB_ADDR_BIAS,
)


@dataclass
class _XrtArray:
    name: str
    bo: object
    array: np.ndarray
    sync_direction: object

    def sync_to_device(self, size: int | None = None, offset: int = 0) -> None:
        self._sync(self.sync_direction.XCL_BO_SYNC_BO_TO_DEVICE, size, offset)

    def sync_from_device(self, size: int | None = None, offset: int = 0) -> None:
        self._sync(self.sync_direction.XCL_BO_SYNC_BO_FROM_DEVICE, size, offset)

    def _sync(self, direction: object, size: int | None, offset: int) -> None:
        if size is None:
            size = self.bo.size()
        self.bo.sync(direction, int(size), int(offset))


@dataclass(frozen=True)
class U280DraftProfile:
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    profiled_seconds: float = 0.0
    excluded_prefill_seconds: float = 0.0


class U280DraftModel:
    """pyxrt host for the Llama-3.2-1B U280 draft bitstreams.

    The shipped decode bitstream stores token IDs only. For exact vanilla
    speculative decoding, this host drives all decode random seeds to zero,
    making each proposal greedy and giving the draft distribution q(x)=1 for
    the proposed token.
    """

    def __init__(
        self,
        paths: U280Paths | None = None,
        *,
        device_index: int | None = None,
        xrt_probe_limit: int = 16,
        max_dec_seq_len: int = MAX_DEC_SEQ_LEN,
        greedy: bool = True,
    ) -> None:
        if max_dec_seq_len > MAX_DEC_SEQ_LEN:
            raise ValueError(f"max_dec_seq_len cannot exceed {MAX_DEC_SEQ_LEN}")
        if xrt_probe_limit <= 0:
            raise ValueError("xrt_probe_limit must be positive")
        if not greedy:
            raise ValueError(
                "The current U280 bitstream does not expose draft probabilities. "
                "Use greedy=True for exact speculative decoding."
            )
        self.paths = paths or U280Paths.default()
        self.device_index = device_index
        self.xrt_probe_limit = xrt_probe_limit
        self.max_dec_seq_len = max_dec_seq_len
        self.greedy = greedy

        self._xrt = None
        self._device = None
        self._pref_uuid = None
        self._dec_uuid = None
        self._pref_kernel = None
        self._dec_kernel = None
        self._loaded_xclbin_kind: str | None = None

        self._pref_static_loaded = False
        self._dec_static_loaded = False
        self._pref_static_synced = False
        self._dec_static_synced = False
        self._pref_k_cache_host: np.ndarray | None = None
        self._pref_v_cache_host: np.ndarray | None = None
        self._proposal_count = 0
        self.last_profile = U280DraftProfile()

        self._embedding = np.memmap(
            self.paths.parameters_dir / "model_embed_tokens_fp32.bin",
            dtype=np.float32,
            mode="r",
            shape=(VOCAB_SIZE, HIDDEN_DIM),
        )

    def propose(self, prefix_ids: Sequence[int], num_tokens: int) -> Sequence[DraftToken]:
        if num_tokens <= 0:
            return []
        if num_tokens > self.max_dec_seq_len:
            raise ValueError(f"num_tokens cannot exceed {self.max_dec_seq_len}")
        if len(prefix_ids) < 2:
            raise ValueError("U280 draft prefill needs at least two prefix tokens")
        if len(prefix_ids) - 1 > MAX_PRE_SEQ_LEN:
            raise ValueError(
                f"U280 prefill supports at most {MAX_PRE_SEQ_LEN + 1} prefix tokens, "
                f"got {len(prefix_ids)}"
            )
        if len(prefix_ids) - 1 + num_tokens > MAX_SUM_SEQ_LEN:
            raise ValueError("U280 decode would exceed MAX_SUM_SEQ_LEN")

        self._ensure_xrt()
        prefill_seconds = self._run_prefill(prefix_ids)
        sampled, decode_seconds = self._run_decode(prefix_ids, num_tokens)
        include_prefill = self._proposal_count == 0
        excluded_prefill_seconds = 0.0 if include_prefill else prefill_seconds
        profiled_seconds = decode_seconds + (prefill_seconds if include_prefill else 0.0)
        self.last_profile = U280DraftProfile(
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            profiled_seconds=profiled_seconds,
            excluded_prefill_seconds=excluded_prefill_seconds,
        )
        self._proposal_count += 1
        return [DraftToken.greedy(token_id) for token_id in sampled]

    def initialize_runtime(self) -> int:
        """Open the XRT device so FPGA failures can be debugged before vLLM load."""
        self._ensure_xrt()
        if self.device_index is None:
            raise RuntimeError("XRT device opened but selected device index was not recorded")
        return self.device_index

    def _ensure_xrt(self) -> None:
        if self._xrt is not None:
            return
        try:
            import pyxrt as xrt
        except ImportError as exc:
            if "GLIBCXX_3.4.30" in str(exc) and "libstdc++.so.6" in str(exc):
                raise ImportError(
                    "pyxrt could not load XRT because the active conda libstdc++.so.6 "
                    "does not provide GLIBCXX_3.4.30. Launch Python with "
                    "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 or update "
                    "the conda libstdcxx-ng package."
                ) from exc
            raise

        device = self._open_xrt_device(xrt)
        self._xrt = xrt
        self._device = device

    def _open_xrt_device(self, xrt: object) -> object:
        if self.device_index is not None:
            try:
                return xrt.device(self.device_index)
            except Exception as exc:
                raise RuntimeError(f"Could not open requested XRT device index {self.device_index}") from exc

        failures: list[str] = []
        for index in range(self.xrt_probe_limit):
            try:
                device = xrt.device(index)
            except Exception as exc:
                failures.append(f"{index}: {exc}")
                continue
            self.device_index = index
            return device

        details = "; ".join(failures[:8])
        if len(failures) > 8:
            details += "; ..."
        raise RuntimeError(
            f"Could not open any XRT device in indices 0..{self.xrt_probe_limit - 1}. "
            f"Probe failures: {details}"
        )

    def _load_prefill_xclbin(self) -> None:
        if self._loaded_xclbin_kind != "prefill":
            if self._loaded_xclbin_kind == "decode":
                self._release_decode_xrt_objects()
            self._pref_uuid = self._device.load_xclbin(str(self.paths.prefill_xclbin))
            self._pref_kernel = self._xrt.kernel(self._device, self._pref_uuid, "SpinQuant_Prefilling")
            self._pref_static_synced = False
            self._loaded_xclbin_kind = "prefill"
        if not self._pref_static_loaded:
            self._alloc_prefill_buffers()
            self._load_prefill_static_buffers()
            self._pref_static_loaded = True
        if not self._pref_static_synced:
            self._sync_prefill_static()
            self._pref_static_synced = True

    def _load_decode_xclbin(self) -> None:
        if self._loaded_xclbin_kind != "decode":
            if self._loaded_xclbin_kind == "prefill":
                self._release_prefill_xrt_objects()
            self._dec_uuid = self._device.load_xclbin(str(self.paths.decode_xclbin))
            self._dec_kernel = self._xrt.kernel(self._device, self._dec_uuid, "SpinQuant_Decoding")
            self._dec_static_synced = False
            self._loaded_xclbin_kind = "decode"
        if not self._dec_static_loaded:
            self._alloc_decode_buffers()
            self._load_decode_static_buffers()
            self._dec_static_loaded = True
        if not self._dec_static_synced:
            self._sync_decode_static()
            self._dec_static_synced = True

    def _release_prefill_xrt_objects(self) -> None:
        for name in (
            "pref_io",
            "pref_wk_wq",
            "pref_wk_wq_s",
            "pref_wv_wo",
            "pref_wv_wo_s",
            "pref_k_cache",
            "pref_v_cache",
            "pref_w_gate",
            "pref_w_gate_s",
            "pref_w_up",
            "pref_w_up_s",
            "pref_w_down",
            "pref_w_down_s",
            "pref_gamma_0",
            "pref_gamma_1",
        ):
            if hasattr(self, name):
                setattr(self, name, None)
        self._pref_kernel = None
        self._pref_uuid = None
        self._pref_static_loaded = False
        self._pref_static_synced = False
        self._loaded_xclbin_kind = None
        gc.collect()

    def _release_decode_xrt_objects(self) -> None:
        for name in (
            "dec_vocab",
            "dec_io",
            "dec_w_half0",
            "dec_w_half1",
            "dec_w_s",
            "dec_gamma",
            "dec_rand",
            "dec_sampled",
        ):
            if hasattr(self, name):
                setattr(self, name, None)
        self._dec_kernel = None
        self._dec_uuid = None
        self._dec_static_loaded = False
        self._dec_static_synced = False
        self._loaded_xclbin_kind = None
        gc.collect()

    def _alloc(self, kernel: object, arg_index: int, name: str, shape: tuple[int, ...], dtype: np.dtype) -> _XrtArray:
        dtype = np.dtype(dtype)
        size = int(np.prod(shape)) * dtype.itemsize
        group_id = kernel.group_id(arg_index)
        bo = self._xrt.bo(self._device, size, self._xrt.bo.flags.normal, group_id)
        array = np.ndarray(shape=shape, dtype=dtype, buffer=bo.map())
        return _XrtArray(name=name, bo=bo, array=array, sync_direction=self._xrt.xclBOSyncDirection)

    def _alloc_prefill_buffers(self) -> None:
        k = self._pref_kernel
        self.pref_io = self._alloc(k, 0, "pref_io", (PREF_IO_ELEMS, TOKEN_PARALLEL), np.float32)
        self.pref_wk_wq = self._alloc(k, 1, "pref_wk_wq", (PREF_QKVO_ELEMS, PREF_QKVO_W_PARALLEL_READ // 2), np.uint8)
        self.pref_wk_wq_s = self._alloc(k, 2, "pref_wk_wq_s", (PREF_QKVO_S_ELEMS, 2), np.float32)
        self.pref_wv_wo = self._alloc(k, 3, "pref_wv_wo", (PREF_QKVO_ELEMS, PREF_QKVO_W_PARALLEL_READ // 2), np.uint8)
        self.pref_wv_wo_s = self._alloc(k, 4, "pref_wv_wo_s", (PREF_QKVO_S_ELEMS, 2), np.float32)
        self.pref_k_cache = self._alloc(k, 5, "pref_k_cache", (PREF_K_CACHE_ELEMS, PREF_K_PARALLEL), np.uint8)
        self.pref_v_cache = self._alloc(k, 6, "pref_v_cache", (PREF_V_CACHE_ELEMS, PREF_V_PARALLEL), np.uint8)
        lane = PREF_FFN_W_PARALLEL_READ // PREF_FFN_W_BLOCK_NUM // 2
        self.pref_w_gate = [
            self._alloc(k, 7 + i, f"pref_w_gate_{i}", (PREF_FFN_INTER_ELEMS, lane), np.uint8)
            for i in range(PREF_FFN_W_BLOCK_NUM)
        ]
        self.pref_w_gate_s = self._alloc(k, 10, "pref_w_gate_s", (DECODER_LAYER_NUM * INTER_DIM, 2), np.float32)
        self.pref_w_up = [
            self._alloc(k, 11 + i, f"pref_w_up_{i}", (PREF_FFN_INTER_ELEMS, lane), np.uint8)
            for i in range(PREF_FFN_W_BLOCK_NUM)
        ]
        self.pref_w_up_s = self._alloc(k, 14, "pref_w_up_s", (DECODER_LAYER_NUM * INTER_DIM, 2), np.float32)
        self.pref_w_down = [
            self._alloc(k, 15 + i, f"pref_w_down_{i}", (PREF_FFN_DOWN_ELEMS, lane), np.uint8)
            for i in range(PREF_FFN_W_BLOCK_NUM)
        ]
        self.pref_w_down_s = self._alloc(k, 18, "pref_w_down_s", (DECODER_LAYER_NUM * HIDDEN_DIM, 2), np.float32)
        self.pref_gamma_0 = self._alloc(k, 19, "pref_gamma_0", (DECODER_LAYER_NUM * HIDDEN_DIM,), np.float32)
        self.pref_gamma_1 = self._alloc(k, 20, "pref_gamma_1", (DECODER_LAYER_NUM * HIDDEN_DIM,), np.float32)

    def _alloc_decode_buffers(self) -> None:
        k = self._dec_kernel
        self.dec_vocab = self._alloc(k, 0, "dec_vocab", (VOCAB_SIZE * HIDDEN_DIM // T_BLOCK_PARALLEL, T_BLOCK_PARALLEL), np.float32)
        self.dec_io = self._alloc(
            k,
            1,
            "dec_io",
            (MAX_DEC_SEQ_LEN * (DECODER_LAYER_NUM + 1) * HIDDEN_DIM // T_BLOCK_PARALLEL, T_BLOCK_PARALLEL),
            np.float32,
        )
        weight_shape = (W_QKVO_FFN_SIZE + DEC_K_CACHE_ELEMS, DEC_QKVO_FFN_W_PARALLEL // 2)
        self.dec_w_half0 = [
            self._alloc(k, 2 + i, f"dec_w_half0_{i}", weight_shape, np.uint8)
            for i in range(T_QKVO_FFN_BLOCK_PARALLEL // 2)
        ]
        weight_shape = (W_QKVO_FFN_SIZE + DEC_V_CACHE_ELEMS, DEC_QKVO_FFN_W_PARALLEL // 2)
        self.dec_w_half1 = [
            self._alloc(k, 10 + i, f"dec_w_half1_{i}", weight_shape, np.uint8)
            for i in range(T_QKVO_FFN_BLOCK_PARALLEL // 2)
        ]
        self.dec_w_s = self._alloc(k, 18, "dec_w_s", (W_S_QKVO_FFN_SIZE, 2), np.float32)
        self.dec_gamma = self._alloc(
            k,
            19,
            "dec_gamma",
            ((2 * DECODER_LAYER_NUM + 1) * HIDDEN_DIM // T_BLOCK_PARALLEL, T_BLOCK_PARALLEL),
            np.float32,
        )
        self.dec_rand = self._alloc(k, 20, "dec_rand", (MAX_DEC_SEQ_LEN,), np.float32)
        self.dec_sampled = self._alloc(k, 21, "dec_sampled", (MAX_DEC_SEQ_LEN,), np.int32)

    def _load_prefill_static_buffers(self) -> None:
        for arr in (
            self.pref_wk_wq,
            self.pref_wk_wq_s,
            self.pref_wv_wo,
            self.pref_wv_wo_s,
            self.pref_w_gate_s,
            self.pref_w_up_s,
            self.pref_w_down_s,
            self.pref_gamma_0,
            self.pref_gamma_1,
        ):
            arr.array.fill(0)
        for arr in self.pref_w_gate + self.pref_w_up + self.pref_w_down:
            arr.array.fill(0)

        for layer in range(DECODER_LAYER_NUM):
            self._pack_pref_weight("k_proj", layer, HIDDEN_DIM, KV_HIDDEN_DIM, self.pref_wk_wq.array, layer * self._pref_kv_stride())
            self._pack_pref_weight(
                "q_proj",
                layer,
                HIDDEN_DIM,
                HIDDEN_DIM,
                self.pref_wk_wq.array,
                DECODER_LAYER_NUM * self._pref_kv_stride() + layer * self._pref_hidden_stride(),
            )
            self._pack_pref_weight("v_proj", layer, HIDDEN_DIM, KV_HIDDEN_DIM, self.pref_wv_wo.array, layer * self._pref_kv_stride())
            self._pack_pref_weight(
                "o_proj",
                layer,
                HIDDEN_DIM,
                HIDDEN_DIM,
                self.pref_wv_wo.array,
                DECODER_LAYER_NUM * self._pref_kv_stride() + layer * self._pref_hidden_stride(),
            )
            self._pack_pref_blocked_weight("gate_proj", layer, HIDDEN_DIM, INTER_DIM, [x.array for x in self.pref_w_gate], layer * self._pref_inter_stride())
            self._pack_pref_blocked_weight("up_proj", layer, HIDDEN_DIM, INTER_DIM, [x.array for x in self.pref_w_up], layer * self._pref_inter_stride())
            self._pack_pref_blocked_weight("down_proj", layer, INTER_DIM, HIDDEN_DIM, [x.array for x in self.pref_w_down], layer * self._pref_down_stride())

        self._load_prefill_scales()

    def _load_decode_static_buffers(self) -> None:
        self.dec_vocab.array.fill(0)
        self.dec_w_s.array.fill(0)
        self.dec_gamma.array.fill(0)
        for arr in self.dec_w_half0 + self.dec_w_half1:
            arr.array.fill(0)

        self.dec_vocab.array.reshape(VOCAB_SIZE, HIDDEN_DIM // T_BLOCK_PARALLEL, T_BLOCK_PARALLEL)[:] = (
            self._embedding.reshape(VOCAB_SIZE, T_BLOCK_PARALLEL, HIDDEN_DIM // T_BLOCK_PARALLEL).transpose(0, 2, 1)
        )

        for layer in range(DECODER_LAYER_NUM):
            self._pack_dec_weight("k_proj", layer, HIDDEN_DIM, KV_HIDDEN_DIM, self.dec_w_half0, W_KV_ADDR_BIAS + layer * self._dec_kv_stride(), 0)
            self._pack_dec_weight("k_proj", layer, HIDDEN_DIM, KV_HIDDEN_DIM, self.dec_w_half1, W_KV_ADDR_BIAS + layer * self._dec_kv_stride(), 1)
            self._pack_dec_weight(
                "v_proj",
                layer,
                HIDDEN_DIM,
                KV_HIDDEN_DIM,
                self.dec_w_half0,
                W_KV_ADDR_BIAS + layer * self._dec_kv_stride(),
                0,
                KV_HIDDEN_DIM // T_QKVO_FFN_BLOCK_PARALLEL,
            )
            self._pack_dec_weight(
                "v_proj",
                layer,
                HIDDEN_DIM,
                KV_HIDDEN_DIM,
                self.dec_w_half1,
                W_KV_ADDR_BIAS + layer * self._dec_kv_stride(),
                1,
                KV_HIDDEN_DIM // T_QKVO_FFN_BLOCK_PARALLEL,
            )
            self._pack_dec_weight("q_proj", layer, HIDDEN_DIM, HIDDEN_DIM, self.dec_w_half0, W_Q_ADDR_BIAS + layer * self._dec_hidden_stride(), 0)
            self._pack_dec_weight("q_proj", layer, HIDDEN_DIM, HIDDEN_DIM, self.dec_w_half1, W_Q_ADDR_BIAS + layer * self._dec_hidden_stride(), 1)
            self._pack_dec_weight("o_proj", layer, HIDDEN_DIM, HIDDEN_DIM, self.dec_w_half0, W_O_ADDR_BIAS + layer * self._dec_hidden_stride(), 0)
            self._pack_dec_weight("o_proj", layer, HIDDEN_DIM, HIDDEN_DIM, self.dec_w_half1, W_O_ADDR_BIAS + layer * self._dec_hidden_stride(), 1)
            self._pack_dec_weight("up_proj", layer, HIDDEN_DIM, INTER_DIM, self.dec_w_half0, W_FFN_UP_ADDR_BIAS + layer * self._dec_inter_stride(), 0)
            self._pack_dec_weight("up_proj", layer, HIDDEN_DIM, INTER_DIM, self.dec_w_half1, W_FFN_UP_ADDR_BIAS + layer * self._dec_inter_stride(), 1)
            self._pack_dec_weight("gate_proj", layer, HIDDEN_DIM, INTER_DIM, self.dec_w_half0, W_FFN_GATE_ADDR_BIAS + layer * self._dec_inter_stride(), 0)
            self._pack_dec_weight("gate_proj", layer, HIDDEN_DIM, INTER_DIM, self.dec_w_half1, W_FFN_GATE_ADDR_BIAS + layer * self._dec_inter_stride(), 1)
            self._pack_dec_weight("down_proj", layer, INTER_DIM, HIDDEN_DIM, self.dec_w_half0, W_FFN_DOWN_ADDR_BIAS + layer * self._dec_down_stride(), 0)
            self._pack_dec_weight("down_proj", layer, INTER_DIM, HIDDEN_DIM, self.dec_w_half1, W_FFN_DOWN_ADDR_BIAS + layer * self._dec_down_stride(), 1)

        self._pack_dec_weight("lm_head", 0, HIDDEN_DIM, VOCAB_SIZE_PAD, self.dec_w_half0, W_VOCAB_ADDR_BIAS, 0)
        self._pack_dec_weight("lm_head", 0, HIDDEN_DIM, VOCAB_SIZE_PAD, self.dec_w_half1, W_VOCAB_ADDR_BIAS, 1)
        self._load_decode_scales()

    def _sync_prefill_static(self) -> None:
        for arr in [
            self.pref_wk_wq,
            self.pref_wk_wq_s,
            self.pref_wv_wo,
            self.pref_wv_wo_s,
            *self.pref_w_gate,
            self.pref_w_gate_s,
            *self.pref_w_up,
            self.pref_w_up_s,
            *self.pref_w_down,
            self.pref_w_down_s,
            self.pref_gamma_0,
            self.pref_gamma_1,
        ]:
            arr.sync_to_device()

    def _sync_decode_static(self) -> None:
        self.dec_vocab.sync_to_device()
        static_weight_bytes = W_QKVO_FFN_SIZE * (DEC_QKVO_FFN_W_PARALLEL // 2)
        for arr in self.dec_w_half0 + self.dec_w_half1:
            arr.sync_to_device(size=static_weight_bytes)
        self.dec_w_s.sync_to_device()
        self.dec_gamma.sync_to_device()

    def _run_prefill(self, prefix_ids: Sequence[int]) -> float:
        self._load_prefill_xclbin()
        pre_seq_len = len(prefix_ids) - 1
        pad_factor = max(TOKEN_PARALLEL, PREF_K_PARALLEL)
        pre_seq_len_pad = (pre_seq_len + pad_factor - 1) // pad_factor * pad_factor

        self.pref_io.array.fill(0)
        self._write_prefill_embeddings(prefix_ids[:-1])
        self.pref_k_cache.array.fill(0)
        self.pref_v_cache.array.fill(0)

        self.pref_io.sync_to_device()
        self.pref_k_cache.sync_to_device()
        self.pref_v_cache.sync_to_device()

        kernel_seconds = self._run_kernel(
            self._pref_kernel,
            [
                self.pref_io.bo,
                self.pref_wk_wq.bo,
                self.pref_wk_wq_s.bo,
                self.pref_wv_wo.bo,
                self.pref_wv_wo_s.bo,
                self.pref_k_cache.bo,
                self.pref_v_cache.bo,
                *(arr.bo for arr in self.pref_w_gate),
                self.pref_w_gate_s.bo,
                *(arr.bo for arr in self.pref_w_up),
                self.pref_w_up_s.bo,
                *(arr.bo for arr in self.pref_w_down),
                self.pref_w_down_s.bo,
                self.pref_gamma_0.bo,
                self.pref_gamma_1.bo,
                int(pre_seq_len_pad),
            ],
        )
        self.pref_k_cache.sync_from_device()
        self.pref_v_cache.sync_from_device()
        self._pref_k_cache_host = self.pref_k_cache.array.copy()
        self._pref_v_cache_host = self.pref_v_cache.array.copy()
        return kernel_seconds

    def _run_decode(self, prefix_ids: Sequence[int], num_tokens: int) -> tuple[list[int], float]:
        self._load_decode_xclbin()
        pre_seq_len = len(prefix_ids) - 1

        self.dec_io.array.fill(0)
        last_token = int(prefix_ids[-1])
        self.dec_io.array[: HIDDEN_DIM // T_BLOCK_PARALLEL] = self.dec_vocab.array[
            last_token * (HIDDEN_DIM // T_BLOCK_PARALLEL) : (last_token + 1) * (HIDDEN_DIM // T_BLOCK_PARALLEL)
        ]
        self.dec_rand.array.fill(0.0)
        self.dec_sampled.array.fill(0)
        self._copy_prefill_cache_to_decode(pre_seq_len)

        self.dec_io.sync_to_device()
        self.dec_rand.sync_to_device(size=num_tokens * self.dec_rand.array.dtype.itemsize)
        self.dec_sampled.sync_to_device(size=num_tokens * self.dec_sampled.array.dtype.itemsize)
        self._sync_decode_cache_regions()

        kernel_seconds = self._run_kernel(
            self._dec_kernel,
            [
                self.dec_vocab.bo,
                self.dec_io.bo,
                *(arr.bo for arr in self.dec_w_half0),
                *(arr.bo for arr in self.dec_w_half1),
                self.dec_w_s.bo,
                self.dec_gamma.bo,
                self.dec_rand.bo,
                self.dec_sampled.bo,
                int(pre_seq_len),
                int(num_tokens),
            ],
        )
        self.dec_sampled.sync_from_device(size=num_tokens * self.dec_sampled.array.dtype.itemsize)
        return [int(token_id) for token_id in self.dec_sampled.array[:num_tokens]], kernel_seconds

    def _run_kernel(self, kernel: object, args: Sequence[object]) -> float:
        run = self._xrt.run(kernel)
        for index, arg in enumerate(args):
            run.set_arg(index, arg)
        run.start()
        wait_start = time.perf_counter()
        run.wait()
        return time.perf_counter() - wait_start

    def _write_prefill_embeddings(self, token_ids: Sequence[int]) -> None:
        if len(token_ids) > MAX_PRE_SEQ_LEN:
            raise ValueError(f"prefix prefill length exceeds {MAX_PRE_SEQ_LEN}")
        view = self.pref_io.array.reshape(
            DECODER_LAYER_NUM + 1, MAX_PRE_SEQ_LEN // TOKEN_PARALLEL, HIDDEN_DIM, TOKEN_PARALLEL
        )[0]
        for n, token_id in enumerate(token_ids):
            tile = n // TOKEN_PARALLEL
            lane = n % TOKEN_PARALLEL
            view[tile, :, lane] = self._embedding[int(token_id)]

    def _copy_prefill_cache_to_decode(self, pre_seq_len: int) -> None:
        if self._pref_k_cache_host is None or self._pref_v_cache_host is None:
            raise RuntimeError("prefill KV cache host copy is missing; run prefill before decode")
        for arr in self.dec_w_half0:
            arr.array[W_QKVO_FFN_SIZE:].fill(0)
        for arr in self.dec_w_half1:
            arr.array[W_QKVO_FFN_SIZE:].fill(0)

        for layer in range(DECODER_LAYER_NUM):
            for seq_idx in range(pre_seq_len):
                for head in range(KV_HEAD_NUM):
                    target_bank = head // (KV_HEAD_NUM // DEC_HEAD_PARALLEL)
                    local_head = head % (KV_HEAD_NUM // DEC_HEAD_PARALLEL)
                    for dim in range(HEAD_DIM):
                        write_idx = (
                            ((layer * KV_HEAD_NUM // DEC_HEAD_PARALLEL + local_head) * MAX_SUM_SEQ_LEN + seq_idx)
                            // DEC_K_PARALLEL
                            * HEAD_DIM
                            + dim
                        )
                        write_lane = seq_idx % DEC_K_PARALLEL
                        read_idx = ((layer * KV_HEAD_NUM + head) * MAX_PRE_SEQ_LEN + seq_idx) // PREF_K_PARALLEL * HEAD_DIM + dim
                        read_lane = seq_idx % PREF_K_PARALLEL
                        self.dec_w_half0[target_bank].array[W_QKVO_FFN_SIZE + write_idx, write_lane] = self._pref_k_cache_host[
                            read_idx, read_lane
                        ]

                        write_idx = (
                            ((layer * (KV_HEAD_NUM // DEC_HEAD_PARALLEL) + local_head) * HEAD_DIM + dim)
                            // DEC_V_PARALLEL
                            * MAX_SUM_SEQ_LEN
                            + seq_idx
                        )
                        write_lane = dim % DEC_V_PARALLEL
                        read_idx = ((layer * KV_HIDDEN_DIM + head * HEAD_DIM + dim) // PREF_V_PARALLEL * MAX_PRE_SEQ_LEN) + seq_idx
                        read_lane = (head * HEAD_DIM + dim) % PREF_V_PARALLEL
                        self.dec_w_half1[target_bank].array[W_QKVO_FFN_SIZE + write_idx, write_lane] = self._pref_v_cache_host[
                            read_idx, read_lane
                        ]

    def _sync_decode_cache_regions(self) -> None:
        elem_bytes = DEC_QKVO_FFN_W_PARALLEL // 2
        offset = W_QKVO_FFN_SIZE * elem_bytes
        for arr in self.dec_w_half0:
            arr.sync_to_device(size=DEC_K_CACHE_ELEMS * elem_bytes, offset=offset)
        for arr in self.dec_w_half1:
            arr.sync_to_device(size=DEC_V_CACHE_ELEMS * elem_bytes, offset=offset)

    def _pack_pref_weight(
        self,
        layer_name: str,
        layer: int,
        input_dim: int,
        output_dim: int,
        out: np.ndarray,
        mmap_offset: int,
    ) -> None:
        data = self._weight_matrix(layer_name, layer, input_dim, output_dim)
        lanes = PREF_QKVO_W_PARALLEL // 2
        for n in range(output_dim // 2):
            tile = n // lanes
            lane = n % lanes
            out[mmap_offset + tile * input_dim : mmap_offset + (tile + 1) * input_dim, lane] = self._pack_rows(data[2 * n], data[2 * n + 1])

    def _pack_pref_blocked_weight(
        self,
        layer_name: str,
        layer: int,
        input_dim: int,
        output_dim: int,
        outs: Sequence[np.ndarray],
        mmap_offset: int,
    ) -> None:
        data = self._weight_matrix(layer_name, layer, input_dim, output_dim)
        lanes = PREF_FFN_W_PARALLEL // PREF_FFN_W_BLOCK_NUM // 2
        for n in range(output_dim // 2):
            block_id = n % PREF_FFN_W_BLOCK_NUM
            tile = (n // PREF_FFN_W_BLOCK_NUM) // lanes
            lane = (n // PREF_FFN_W_BLOCK_NUM) % lanes
            outs[block_id][mmap_offset + tile * input_dim : mmap_offset + (tile + 1) * input_dim, lane] = self._pack_rows(
                data[2 * n], data[2 * n + 1]
            )

    def _pack_dec_weight(
        self,
        layer_name: str,
        layer: int,
        input_dim: int,
        output_dim: int,
        outs: Sequence[_XrtArray],
        mmap_offset: int,
        unroll_id: int,
        out_dim_offset: int = 0,
    ) -> None:
        data = self._weight_matrix(layer_name, layer, input_dim, output_dim)
        lanes = DEC_QKVO_FFN_W_PARALLEL // 2
        blocks = T_QKVO_FFN_BLOCK_PARALLEL // 2
        rows_per_block = output_dim // T_QKVO_FFN_BLOCK_PARALLEL
        for block_id in range(blocks):
            actual_block_id = unroll_id * blocks + block_id
            base_row = actual_block_id * rows_per_block
            out = outs[block_id].array
            for n in range(rows_per_block // 2):
                packed_n = n + out_dim_offset // 2
                tile = packed_n // lanes
                lane = packed_n % lanes
                row0 = base_row + 2 * n
                row1 = base_row + 2 * n + 1
                row0_data = self._weight_row_or_zero(data, row0, input_dim)
                row1_data = self._weight_row_or_zero(data, row1, input_dim)
                out[mmap_offset + tile * input_dim : mmap_offset + (tile + 1) * input_dim, lane] = self._pack_rows(row0_data, row1_data)

    def _weight_matrix(self, layer_name: str, layer: int, input_dim: int, output_dim: int) -> np.memmap:
        filename = "lm_head.bin" if layer_name == "lm_head" else f"{layer_name}_L{layer:02d}.bin"
        path = self.paths.parameters_dir / filename
        actual_output_dim = path.stat().st_size // input_dim
        if path.stat().st_size % input_dim != 0:
            raise ValueError(f"{path} size is not divisible by input_dim={input_dim}")
        return np.memmap(
            path,
            dtype=np.int8,
            mode="r",
            shape=(min(output_dim, actual_output_dim), input_dim),
        )

    @staticmethod
    def _pack_rows(row0: np.ndarray, row1: np.ndarray) -> np.ndarray:
        return ((row0.astype(np.int16) & 0x0F) | ((row1.astype(np.int16) & 0x0F) << 4)).astype(np.uint8)

    @staticmethod
    def _weight_row_or_zero(data: np.ndarray, row: int, width: int) -> np.ndarray:
        if row < data.shape[0]:
            return data[row]
        return np.zeros(width, dtype=np.int8)

    def _load_prefill_scales(self) -> None:
        p = self.paths.parameters_dir
        w_k_s, w_k_sum = load_scale_sum(p, "w_k_proj_s_sum.h", "w_k_proj_s", "w_k_proj_sum", (DECODER_LAYER_NUM, KV_HIDDEN_DIM))
        w_v_s, w_v_sum = load_scale_sum(p, "w_v_proj_s_sum.h", "w_v_proj_s", "w_v_proj_sum", (DECODER_LAYER_NUM, KV_HIDDEN_DIM))
        w_q_s, w_q_sum = load_scale_sum(p, "w_q_proj_s_sum.h", "w_q_proj_s", "w_q_proj_sum", (DECODER_LAYER_NUM, HIDDEN_DIM))
        w_o_s, w_o_sum = load_scale_sum(p, "w_o_proj_s_sum.h", "w_o_proj_s", "w_o_proj_sum", (DECODER_LAYER_NUM, HIDDEN_DIM))
        w_gate_s, w_gate_sum = load_scale_sum(p, "w_gate_proj_s_sum.h", "w_gate_proj_s", "w_gate_proj_sum", (DECODER_LAYER_NUM, INTER_DIM))
        w_up_s, w_up_sum = load_scale_sum(p, "w_up_proj_s_sum.h", "w_up_proj_s", "w_up_proj_sum", (DECODER_LAYER_NUM, INTER_DIM))
        w_down_s, w_down_sum = load_scale_sum(p, "w_down_proj_s_sum.h", "w_down_proj_s", "w_down_proj_sum", (DECODER_LAYER_NUM, HIDDEN_DIM))
        rms = load_float_array(str(p / "w_rmsnorm.h"), "RMSNorm_weight", (2 * DECODER_LAYER_NUM + 1, HIDDEN_DIM))

        for layer in range(DECODER_LAYER_NUM):
            bias = layer * KV_HIDDEN_DIM
            self.pref_wk_wq_s.array[bias : bias + KV_HIDDEN_DIM, 0] = w_k_s[layer]
            self.pref_wk_wq_s.array[bias : bias + KV_HIDDEN_DIM, 1] = w_k_sum[layer]
            self.pref_wv_wo_s.array[bias : bias + KV_HIDDEN_DIM, 0] = w_v_s[layer]
            self.pref_wv_wo_s.array[bias : bias + KV_HIDDEN_DIM, 1] = w_v_sum[layer]

            bias = DECODER_LAYER_NUM * KV_HIDDEN_DIM + layer * HIDDEN_DIM
            self.pref_wk_wq_s.array[bias : bias + HIDDEN_DIM, 0] = w_q_s[layer]
            self.pref_wk_wq_s.array[bias : bias + HIDDEN_DIM, 1] = w_q_sum[layer]
            self.pref_wv_wo_s.array[bias : bias + HIDDEN_DIM, 0] = w_o_s[layer]
            self.pref_wv_wo_s.array[bias : bias + HIDDEN_DIM, 1] = w_o_sum[layer]

            self.pref_w_gate_s.array[layer * INTER_DIM : (layer + 1) * INTER_DIM, 0] = w_gate_s[layer]
            self.pref_w_gate_s.array[layer * INTER_DIM : (layer + 1) * INTER_DIM, 1] = w_gate_sum[layer]
            self.pref_w_up_s.array[layer * INTER_DIM : (layer + 1) * INTER_DIM, 0] = w_up_s[layer]
            self.pref_w_up_s.array[layer * INTER_DIM : (layer + 1) * INTER_DIM, 1] = w_up_sum[layer]
            self.pref_w_down_s.array[layer * HIDDEN_DIM : (layer + 1) * HIDDEN_DIM, 0] = w_down_s[layer]
            self.pref_w_down_s.array[layer * HIDDEN_DIM : (layer + 1) * HIDDEN_DIM, 1] = w_down_sum[layer]
            self.pref_gamma_0.array[layer * HIDDEN_DIM : (layer + 1) * HIDDEN_DIM] = rms[2 * layer]
            self.pref_gamma_1.array[layer * HIDDEN_DIM : (layer + 1) * HIDDEN_DIM] = rms[2 * layer + 1]

    def _load_decode_scales(self) -> None:
        p = self.paths.parameters_dir
        w_k_s, w_k_sum = load_scale_sum(p, "w_k_proj_s_sum.h", "w_k_proj_s", "w_k_proj_sum", (DECODER_LAYER_NUM, KV_HIDDEN_DIM))
        w_v_s, w_v_sum = load_scale_sum(p, "w_v_proj_s_sum.h", "w_v_proj_s", "w_v_proj_sum", (DECODER_LAYER_NUM, KV_HIDDEN_DIM))
        w_q_s, w_q_sum = load_scale_sum(p, "w_q_proj_s_sum.h", "w_q_proj_s", "w_q_proj_sum", (DECODER_LAYER_NUM, HIDDEN_DIM))
        w_o_s, w_o_sum = load_scale_sum(p, "w_o_proj_s_sum.h", "w_o_proj_s", "w_o_proj_sum", (DECODER_LAYER_NUM, HIDDEN_DIM))
        w_gate_s, w_gate_sum = load_scale_sum(p, "w_gate_proj_s_sum.h", "w_gate_proj_s", "w_gate_proj_sum", (DECODER_LAYER_NUM, INTER_DIM))
        w_up_s, w_up_sum = load_scale_sum(p, "w_up_proj_s_sum.h", "w_up_proj_s", "w_up_proj_sum", (DECODER_LAYER_NUM, INTER_DIM))
        w_down_s, w_down_sum = load_scale_sum(p, "w_down_proj_s_sum.h", "w_down_proj_s", "w_down_proj_sum", (DECODER_LAYER_NUM, HIDDEN_DIM))
        w_lm_s, w_lm_sum = load_scale_sum(p, "w_lm_head_lm_head.h", "w_lm_head_lm_head_s", "w_lm_head_lm_head_sum", (VOCAB_SIZE,))
        rms = load_float_array(str(p / "w_rmsnorm.h"), "RMSNorm_weight", (2 * DECODER_LAYER_NUM + 1, HIDDEN_DIM))

        for layer in range(DECODER_LAYER_NUM):
            for t in range(T_QKVO_FFN_BLOCK_PARALLEL):
                bias_k = W_S_KV_ADDR_BIAS + layer * KV_HIDDEN_DIM_PAD + t * KV_HIDDEN_DIM_PAD // T_QKVO_FFN_BLOCK_PARALLEL
                bias_v = bias_k + KV_HIDDEN_DIM // T_QKVO_FFN_BLOCK_PARALLEL
                sl = slice(t * KV_HIDDEN_DIM // T_QKVO_FFN_BLOCK_PARALLEL, (t + 1) * KV_HIDDEN_DIM // T_QKVO_FFN_BLOCK_PARALLEL)
                self.dec_w_s.array[bias_k : bias_k + KV_HIDDEN_DIM // T_QKVO_FFN_BLOCK_PARALLEL, 0] = w_k_s[layer, sl]
                self.dec_w_s.array[bias_k : bias_k + KV_HIDDEN_DIM // T_QKVO_FFN_BLOCK_PARALLEL, 1] = w_k_sum[layer, sl]
                self.dec_w_s.array[bias_v : bias_v + KV_HIDDEN_DIM // T_QKVO_FFN_BLOCK_PARALLEL, 0] = w_v_s[layer, sl]
                self.dec_w_s.array[bias_v : bias_v + KV_HIDDEN_DIM // T_QKVO_FFN_BLOCK_PARALLEL, 1] = w_v_sum[layer, sl]

            self._fill_scale_pair(W_S_Q_ADDR_BIAS + layer * HIDDEN_DIM, w_q_s[layer], w_q_sum[layer])
            self._fill_scale_pair(W_S_O_ADDR_BIAS + layer * HIDDEN_DIM, w_o_s[layer], w_o_sum[layer])
            self._fill_scale_pair(W_S_FFN_UP_ADDR_BIAS + layer * INTER_DIM, w_up_s[layer], w_up_sum[layer])
            self._fill_scale_pair(W_S_FFN_GATE_ADDR_BIAS + layer * INTER_DIM, w_gate_s[layer], w_gate_sum[layer])
            self._fill_scale_pair(W_S_FFN_DOWN_ADDR_BIAS + layer * HIDDEN_DIM, w_down_s[layer], w_down_sum[layer])

        self._fill_scale_pair(W_S_VOCAB_ADDR_BIAS, w_lm_s, w_lm_sum)
        gamma_view = self.dec_gamma.array.reshape(2 * DECODER_LAYER_NUM + 1, HIDDEN_DIM // T_BLOCK_PARALLEL, T_BLOCK_PARALLEL)
        gamma_view[:] = rms.reshape(2 * DECODER_LAYER_NUM + 1, T_BLOCK_PARALLEL, HIDDEN_DIM // T_BLOCK_PARALLEL).transpose(0, 2, 1)

    def _fill_scale_pair(self, offset: int, scale: np.ndarray, row_sum: np.ndarray) -> None:
        self.dec_w_s.array[offset : offset + scale.size, 0] = scale
        self.dec_w_s.array[offset : offset + row_sum.size, 1] = row_sum

    @staticmethod
    def _pref_kv_stride() -> int:
        return ((KV_HIDDEN_DIM + PREF_QKVO_W_PARALLEL - 1) // PREF_QKVO_W_PARALLEL) * HIDDEN_DIM

    @staticmethod
    def _pref_hidden_stride() -> int:
        return ((HIDDEN_DIM + PREF_QKVO_W_PARALLEL - 1) // PREF_QKVO_W_PARALLEL) * HIDDEN_DIM

    @staticmethod
    def _pref_inter_stride() -> int:
        return ((INTER_DIM + PREF_FFN_W_PARALLEL - 1) // PREF_FFN_W_PARALLEL) * HIDDEN_DIM

    @staticmethod
    def _pref_down_stride() -> int:
        return ((HIDDEN_DIM + PREF_FFN_W_PARALLEL - 1) // PREF_FFN_W_PARALLEL) * INTER_DIM

    @staticmethod
    def _dec_kv_stride() -> int:
        return KV_HIDDEN_DIM * 2 // (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM

    @staticmethod
    def _dec_hidden_stride() -> int:
        return HIDDEN_DIM // (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM

    @staticmethod
    def _dec_inter_stride() -> int:
        return INTER_DIM // (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * HIDDEN_DIM

    @staticmethod
    def _dec_down_stride() -> int:
        return HIDDEN_DIM // (T_QKVO_FFN_BLOCK_PARALLEL * DEC_QKVO_FFN_W_PARALLEL) * INTER_DIM
