# SPDX-License-Identifier: Apache-2.0
"""GLM-5-Next sparse MLA (NoPE latent + pooled lightning indexer) on paged rows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.cache import KVCache, PoolingCache
from mlx_vlm.models.glm5_next.language import _sparse_prefill_attention
from mlx_vlm.models.linear import DECODE_BLOCK_SIZE, linear

from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.context import PagedAttentionContext, get_context
from vllm_metal.attention.impls.mla import _block_table_rows, _mla_metadata

# Past this context a prefill chunk attends over the latent rows, as decode
# does, instead of projecting every past row to per-head keys and values,
# whose time and memory grow with the context.
_LATENT_PREFILL_MIN_CONTEXT = 8192


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
    and calls the inner layer, so indexer pooling, top-k selection and sparse
    attention stay mlx-vlm's own; the step's new rows are then written back.
    Long-context prefill chunks attend in the latent space instead of the
    inner layer's per-head projection. Requests run one at a time, and each
    step recompresses the request's pooled keys from its rows: O(context) per
    layer, like the latent gather.
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

        outputs = []
        rows = []
        for req_idx, context_len in enumerate(ctx.context_lens):
            start, end = ctx.cu_seqlens[req_idx], ctx.cu_seqlens[req_idx + 1]
            past = context_len - (end - start)
            caches = self._request_caches(ctx, req_idx, past)
            if (
                end - start > DECODE_BLOCK_SIZE
                and context_len > _LATENT_PREFILL_MIN_CONTEXT
            ):
                output = self._latent_prefill(x[:, start:end], caches)
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

        latent_cache = self._dsa_latent_cache
        latent_cache.write_slots(
            self._dsa_layer_idx,
            _mla_metadata(ctx).slot_mapping,
            mx.concatenate(rows).astype(latent_cache.dtype),
        )
        return mx.concatenate(outputs, axis=1), None

    def _latent_prefill(self, x: mx.array, caches: list[Any]) -> mx.array:
        """The inner layer's forward, attending in the latent space like decode."""
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
        latent_cache, valid, pool, _ = caches
        offset = latent_cache.offset
        latent, _ = latent_cache.update_and_fetch(
            new_latent, mx.zeros((batch, 1, length, 0), dtype=new_latent.dtype)
        )
        topk = inner.indexer(x, q_resid, None, valid, pool, offset)
        out = _sparse_prefill_attention(
            inner.embed_q(q), latent, latent, topk, inner.scale
        )
        out = inner.unembed_out(out).transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return linear(inner.o_proj, out)

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
