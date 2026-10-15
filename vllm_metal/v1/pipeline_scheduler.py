# SPDX-License-Identifier: Apache-2.0
"""Scheduler that spreads decode requests over pipeline-parallel micro-batches."""

from __future__ import annotations

from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler


class PipelineBalancedScheduler(Scheduler):
    """Keep every pipeline stage busy when few requests decode.

    The engine keeps one batch in flight per pipeline stage, but the base
    scheduler puts every ready decode request into one batch; the next batch
    then finds them all in flight and stays empty, so the stages take turns
    idling. When no decode request is in flight, schedule only a
    ``1 / pipeline_parallel_size`` share of the ready ones and defer the rest
    one step, so they form the next batch.
    """

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        decoding = [
            request
            for request in self.running
            if request.num_computed_tokens >= request.num_prompt_tokens
        ]
        ready = [
            request
            for request in decoding
            if request.num_tokens_with_spec - request.num_computed_tokens == 1
        ]
        if len(ready) > 1 and len(ready) == len(decoding):
            pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
            scheduled_now = cdiv(len(ready), pp_size)
            for request in ready[scheduled_now:]:
                # schedule() advances current_step before it checks decode
                # eligibility, so current_step + 2 skips this call alone.
                request.next_decode_eligible_step = self.current_step + 2
        return super().schedule(throttle_prefills)
