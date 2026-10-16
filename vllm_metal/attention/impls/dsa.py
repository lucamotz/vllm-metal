# SPDX-License-Identifier: Apache-2.0
"""GLM-5-Next sparse MLA (NoPE latent + pooled lightning indexer) on paged rows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import vllm.envs as envs
from mlx_vlm.models.cache import KVCache, PoolingCache
from mlx_vlm.models.glm5_next.language import (
    _MAX_PROJECTED_PREFILL_TOKENS,
    _exact_pool_select,
    _sparse_prefill_attention,
)
from mlx_vlm.models.linear import DECODE_BLOCK_SIZE, linear

from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.context import PagedAttentionContext, get_context
from vllm_metal.attention.impls.mla import _block_table_rows, _mla_metadata

# Past this context a prefill chunk attends over the latent rows, as decode
# does, instead of projecting every past row to per-head keys and values. It is
# mlx-vlm's own bound on its projected prefill, whose memory grows by every head.
_LATENT_PREFILL_MIN_CONTEXT = _MAX_PROJECTED_PREFILL_TOKENS

# vLLM splits a prefill's sparse-indexer query rows so their float32 logits stay
# within VLLM_SPARSE_INDEXER_MAX_LOGITS_MB. mlx-vlm sums the index heads only
# after it has built per-head float32 scores, so here the cap bounds those.
_INDEXER_SCORE_BYTES = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 2**20

# Peak bytes one sparse layer's step allocates per past row it gathers, in
# multiples of the row itself: the gathered row, the recompressed pooled keys,
# the latent concatenated onto it and the attention's copies. The projected path
# adds its per-head keys and values. Measured on one GLM-5.3-Flash layer at
# 1024-token chunks over 16k-256k of context: 3.5 KB per context token on the
# latent path (2.3 bf16 rows of 768 columns) and 68.0 KB on the projected path.
_LATENT_ROW_COPIES = 3
_PROJECTED_KV_COPIES = 1


def gathered_row_bytes(latent_dim: int, itemsize: int, projected_dim: int = 0) -> int:
    """Peak bytes a sparse layer's step allocates per past row it gathers.

    ``projected_dim`` is the per-head key and value width the step projects
    each row to; zero on the latent path and for decode.
    """
    copies = _LATENT_ROW_COPIES * latent_dim + _PROJECTED_KV_COPIES * projected_dim
    return copies * itemsize


def gather_workspace_bytes(
    *, max_model_len: int, latent_dim: int, projected_dim: int, itemsize: int
) -> int:
    """The allowance all gathered rows of one step share, reserved before KV sizing.

    One request of any length fits it: on the latent path past the threshold,
    or projected below it.
    """
    return max(
        max_model_len * gathered_row_bytes(latent_dim, itemsize),
        min(max_model_len, _LATENT_PREFILL_MIN_CONTEXT)
        * gathered_row_bytes(latent_dim, itemsize, projected_dim),
    )


def indexer_cache_dim(model_args: Mapping[str, Any]) -> int:
    """Columns a pooled indexer appends to each cached MLA latent row."""
    if "index_kpool" not in model_args:
        return 0
    return 2 * int(model_args["index_head_dim"])


def is_dsa_attention(module: nn.Module) -> bool:
    """Return whether a module is an MLA layer with its own pooled indexer."""
    indexer = getattr(module, "indexer", None)
    return hasattr(indexer, "index_kpool") and hasattr(module, "kv_lora_rank")


class _RecordingPoolingCache(PoolingCache):
    """mlx-vlm pooling cache that keeps the raw keys and gates it last received."""

    def accumulate_windows(self, kv: mx.array, gate: mx.array, offset: Any) -> Any:
        self.inputs = (kv, gate)
        return super().accumulate_windows(kv, gate, offset)


class DSAPagedAttentionWrapper(nn.Module):
    """Run a GLM-5-Next sparse attention layer against the paged latent cache.

    Each cached token row is ``[latent | index key | index gate]``. Per request,
    the wrapper rebuilds mlx-vlm's layer caches from the rows of earlier steps
    and runs the inner layer, so indexer pooling, top-k selection and sparse
    attention stay mlx-vlm's own; the step's new rows are then written back.
    Prefill chunks score the indexer's query rows within a fixed byte cap, and
    long-context ones attend in the latent space instead of the inner layer's
    per-head projection. Requests run one at a time, and each step recompresses
    the request's pooled keys from its rows: O(context) per layer, like the
    latent gather.

    All sparse layers and requests of a step share the gather allowance the
    worker reserved before KV sizing (``ctx.dsa_workspace_bytes``): before a
    request would push the step's pending rows past it, the step so far is
    evaluated, as vLLM bounds its sparse-attention workspace by splitting the
    prefill.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        latent_cache: MLAPagedLatentCache,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        self.rebind_cache(latent_cache, cache_idx=layer_idx)
        latent_dim, itemsize = latent_cache.latent_dim, latent_cache.dtype.size
        projected_dim = inner.num_heads * (inner.q_head_dim + inner.v_head_dim)
        object.__setattr__(
            self, "_latent_row_bytes", gathered_row_bytes(latent_dim, itemsize)
        )
        object.__setattr__(
            self,
            "_projected_row_bytes",
            gathered_row_bytes(latent_dim, itemsize, projected_dim),
        )

    def rebind_cache(
        self, latent_cache: MLAPagedLatentCache, *, cache_idx: int
    ) -> None:
        """Refresh the latent cache and compact layer index."""
        object.__setattr__(self, "_dsa_layer_idx", cache_idx)
        object.__setattr__(self, "_dsa_latent_cache", latent_cache)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any = None,
        prev_topk_indices: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask, cache, prev_topk_indices)

        latent_cache = self._dsa_latent_cache
        outputs: list[mx.array] = []
        rows: list[mx.array] = []
        for req_idx, context_len in enumerate(ctx.context_lens):
            start, end = ctx.cu_seqlens[req_idx], ctx.cu_seqlens[req_idx + 1]
            past = context_len - (end - start)
            prefill = end - start > DECODE_BLOCK_SIZE
            latent_prefill = prefill and context_len > _LATENT_PREFILL_MIN_CONTEXT
            projected = prefill and not latent_prefill
            pending = past * (
                self._projected_row_bytes if projected else self._latent_row_bytes
            )
            if (
                ctx.dsa_workspace_bytes
                and ctx.dsa_pending_bytes
                and ctx.dsa_pending_bytes + pending > ctx.dsa_workspace_bytes
            ):
                # Finish the step so far, earlier layers' cache writes included,
                # before this request gathers its rows. eval waits for the
                # result; synchronize lets Metal release the inputs too.
                mx.eval(x, outputs, rows, latent_cache.latent_caches)
                mx.synchronize()
                ctx.dsa_pending_bytes = 0
            ctx.dsa_pending_bytes += pending
            caches = self._request_caches(ctx, req_idx, past)
            if prefill:
                output = self._prefill(x[:, start:end], caches, latent_prefill)
            else:
                output, _ = self._inner(x[:, start:end], None, caches)
            outputs.append(output)
            latent, _, pool, _ = caches
            keys, gates = pool.inputs
            rows.append(
                mx.concatenate(
                    [latent.state[0][0, 0, past:], keys[0], gates[0]], axis=-1
                )
            )

        latent_cache.write_slots(
            self._dsa_layer_idx,
            _mla_metadata(ctx).slot_mapping,
            mx.concatenate(rows).astype(latent_cache.dtype),
        )
        return mx.concatenate(outputs, axis=1), None

    def _prefill(
        self, x: mx.array, caches: list[Any], latent_attention: bool
    ) -> mx.array:
        """The inner layer's forward with the bounded indexer. With
        ``latent_attention`` it attends in the latent space like decode, instead
        of projecting every past row to per-head keys and values."""
        inner = self._inner
        batch, length, _ = x.shape
        q_a, kv_a = mx.split(linear(inner.qkv_a_proj, x), (inner.q_lora_rank,), axis=-1)
        q_resid = inner.q_a_layernorm(q_a)
        q = (
            linear(inner.q_b_proj, q_resid)
            .reshape(batch, length, inner.num_heads, inner.q_head_dim)
            .transpose(0, 2, 1, 3)
        )
        new_latent = inner.kv_a_layernorm(kv_a)[:, None]
        latent_cache, _, pool, projected_cache = caches
        offset = latent_cache.offset
        latent, _ = latent_cache.update_and_fetch(
            new_latent, mx.zeros((batch, 1, length, 0), dtype=new_latent.dtype)
        )
        topk = self._index_topk(x, q_resid, pool, offset)
        if latent_attention:
            out = _sparse_prefill_attention(
                inner.embed_q(q), latent, latent, topk, inner.scale
            )
            out = inner.unembed_out(out).transpose(0, 2, 1, 3)
            out = out.reshape(batch, length, -1)
        else:
            out, _ = inner._attend(
                q, latent, new_latent, topk, latent_cache, projected_cache
            )
        return linear(inner.o_proj, out)

    def _index_topk(
        self, x: mx.array, q_resid: mx.array, pool: PoolingCache, offset: int
    ) -> mx.array:
        """The inner indexer's top-k for one paged request, every key valid.

        The query rows are scored a few at a time, each part evaluated before
        the next, so the float32 per-head scores stay within
        ``_INDEXER_SCORE_BYTES``, as vLLM splits its indexer's prefill queries.
        """
        indexer = self._inner.indexer
        kpool = indexer.index_kpool
        batch, length, _ = x.shape
        keys, gates = indexer._project_keys(x)
        ready_keys, ready_gates, _ = pool.accumulate_windows(
            keys, gates.astype(keys.dtype), offset
        )
        pool_keys = pool.update_and_fetch(
            indexer._compress_pools(ready_keys, ready_gates)
        )
        pool_count = pool_keys.shape[1]
        kv_length = offset + length
        pool_end = mx.clip(mx.arange(1, pool_count + 1) * kpool - 1, 0, kv_length - 1)
        pool_valid = mx.ones((batch, pool_count), dtype=mx.bool_)
        positions = offset + mx.arange(length)
        select_k = min(indexer.index_topk // kpool, pool_count)
        if select_k == pool_count:
            selected, selected_valid = _exact_pool_select(
                None,
                pool_keys,
                None,
                pool_end[None],
                pool_valid,
                positions,
                select_k,
                indexer.softmax_scale,
            )
        else:
            q, weights = indexer._project_queries(x, q_resid)
            step = max(1, _INDEXER_SCORE_BYTES // (indexer.n_heads * pool_count * 4))
            parts = []
            for start in range(0, length, step):
                part = _exact_pool_select(
                    q[:, start : start + step],
                    pool_keys,
                    weights[:, start : start + step],
                    pool_end[None],
                    pool_valid,
                    positions[start : start + step],
                    select_k,
                    indexer.softmax_scale,
                    chunk_size=step,
                )
                mx.eval(part)
                mx.synchronize()
                parts.append(part)
            selected = mx.concatenate([part[0] for part in parts], axis=1)
            selected_valid = mx.concatenate([part[1] for part in parts], axis=1)

        # mlx-vlm's own ordering, padding and always-selected tail.
        order = mx.argsort(mx.where(selected_valid, selected, pool_count), axis=-1)
        selected = mx.take_along_axis(selected, order, axis=-1)
        selected_valid = mx.take_along_axis(selected_valid, order, axis=-1)
        safe = mx.clip(selected, 0, pool_count - 1)
        topk = (safe[..., None] * kpool + mx.arange(kpool)).reshape(batch, length, -1)
        valid = mx.repeat(selected_valid[..., None], kpool, axis=-1)
        topk = mx.where(valid.reshape(batch, length, -1), topk, -1)
        width = indexer.index_topk
        if topk.shape[-1] < width:
            pad = [(0, 0), (0, 0), (0, width - topk.shape[-1])]
            topk = mx.pad(topk, pad, constant_values=-1)
        topk = topk[..., :width]
        if indexer.always_select_tail and kpool > 1:
            visible = positions + 1
            tail_count = visible % kpool
            tail = (visible - tail_count)[:, None] + mx.arange(kpool - 1)
            tail_valid = (mx.arange(kpool - 1) < tail_count[:, None]) & (
                tail < kv_length
            )
            tail = mx.where(tail_valid, tail, -1)[None]
            topk = mx.concatenate([topk, tail], axis=-1)
        return topk.astype(mx.int32)

    def _request_caches(
        self, ctx: PagedAttentionContext, req_idx: int, past: int
    ) -> list[Any]:
        """mlx-vlm's ``[latent, key validity, pooled keys, projected KV]`` caches."""
        inner = self._inner
        indexer = inner.indexer
        latent = KVCache()
        valid = KVCache()
        pool = _RecordingPoolingCache(indexer.index_kpool)
        if past:
            latent_cache = self._dsa_latent_cache
            n_blocks = math.ceil(past / latent_cache.block_size)
            blocks = _block_table_rows(ctx)[req_idx][:n_blocks]
            past_rows = latent_cache.latent_caches[self._dsa_layer_idx][blocks]
            past_rows = past_rows.reshape(1, -1, latent_cache.latent_dim)[:, :past]
            past_latent, keys, gates = mx.split(
                past_rows,
                [inner.kv_lora_rank, inner.kv_lora_rank + indexer.head_dim],
                axis=-1,
            )
            latent.state = (past_latent[:, None], past_latent[:, None, :, :0])
            valid.state = (
                mx.ones((1, 1, past, 1), dtype=mx.bool_),
                mx.zeros((1, 1, past, 0), dtype=mx.bool_),
            )
            pooled = past - past % indexer.index_kpool
            # The indexer's own window compression; mlx-vlm keeps no raw keys
            # for complete pools, so the paged rows are the source of truth.
            pooled_keys = indexer._compress_pools(keys[:, :pooled], gates[:, :pooled])
            if pooled < past:
                pool.state = (keys[:, pooled:], gates[:, pooled:], pooled_keys)
            else:
                pool.state = (None, None, pooled_keys)
        return [latent, valid, pool, None]
