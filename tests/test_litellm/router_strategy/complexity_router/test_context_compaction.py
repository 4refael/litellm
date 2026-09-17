import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from queue import SimpleQueue
from typing import Final, Literal

import pytest
from pydantic import TypeAdapter

from litellm.litellm_core_utils.prompt_templates.compaction import CompactionHistory, split_compaction_history
from litellm.router_strategy.complexity_router.context_compaction import (
    MAX_SUMMARY_CALLS,
    CompactedRequest,
    CompactionFailure,
    CompactionState,
    ModelBudget,
    prepare_compaction,
)
from litellm.types.utils import ModelResponse

_TARGET: Final = ModelBudget("selected", input_limit=1_800, output_limit=80)
_SUMMARY: Final = ModelBudget("summary", input_limit=20_000, output_limit=32)
_MAPPING: Final = TypeAdapter(dict[str, object])


def _serialize(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=dict)


@dataclass(frozen=True)
class _Counter:
    calls: SimpleQueue[tuple[str, Mapping[str, object]]] = field(default_factory=SimpleQueue)

    async def __call__(self, model: str, payload: Mapping[str, object]) -> int:
        await asyncio.sleep(0)
        serialized: Final = _serialize(payload)
        self.calls.put((model, _MAPPING.validate_json(serialized)))
        return len(serialized)


@dataclass(frozen=True)
class _Executor:
    replies: tuple[str | None | Exception, ...] = ("historical fact = 73",)
    finish: str = "stop"
    block: bool = False
    calls: SimpleQueue[tuple[str, Sequence[Mapping[str, str]], int, float]] = field(default_factory=SimpleQueue)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def __call__(
        self, model: str, messages: Sequence[Mapping[str, str]], max_tokens: int, timeout: float
    ) -> ModelResponse:
        self.calls.put((model, messages, max_tokens, timeout))
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        reply: Final = self.replies[min(self.calls.qsize() - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return ModelResponse(
            choices=[{"message": {"role": "assistant", "content": reply}, "finish_reason": self.finish}]
        )


def _payload() -> Mapping[str, object]:
    return {
        "messages": [
            {"role": role, "content": text}
            for role, text in (
                ("user", "historical fact = 73; " * 250), ("assistant", "old answer"), ("user", "current question")
            )
        ],
        "system": "system instructions",
        "instructions": "developer instructions",
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "max_tokens": 32,
        "extra_body": {"provider_feature_enabled": True},
    }


def _state(timeout: float = 10) -> CompactionState:
    state: Final = CompactionState(timeout=timeout)
    state.arm("summary-group")
    return state


async def _run(
    executor: _Executor,
    payload: Mapping[str, object] | None = None,
    target: ModelBudget = _TARGET,
    state: CompactionState | None = None,
    budgets: tuple[ModelBudget, ...] = (_SUMMARY,),
    counter: _Counter | None = None,
) -> CompactedRequest | CompactionFailure | None:
    return await prepare_compaction(
        _payload() if payload is None else payload, target, _state() if state is None else state,
        budgets, executor, _Counter() if counter is None else counter,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("field_name", ("max_tokens", "max_completion_tokens", "max_output_tokens"))
async def test_output_reservation_uses_the_effective_cap(field_name: str) -> None:
    payload: Final = {"messages": [{"role": "user", "content": "old " * 100}, {"role": "user", "content": "new"}]}
    target: Final = ModelBudget("selected", input_limit=len(_serialize(payload)) + 200, output_limit=250)
    low: Final = _Executor()
    high: Final = _Executor()
    assert await _run(low, {**payload, field_name: 10}, target) is None
    assert low.calls.empty()
    result: Final = await _run(high, {**payload, field_name: 200}, target)
    assert isinstance(result, CompactedRequest)
    assert high.calls.qsize() == 1
    assert len(_serialize({**payload, field_name: 200, result.field: result.value})) <= target.input_budget(200)


@pytest.mark.asyncio
@pytest.mark.parametrize("field_name", (None, "max_tokens", "max_completion_tokens", "max_output_tokens"))
async def test_unspecified_output_and_short_responses_need_no_compactable_prefix(field_name: str | None) -> None:
    payload: Final = {"input": "Hello", **({field_name: None} if field_name else {})}
    executor: Final = _Executor()
    assert await _run(executor, payload, ModelBudget("small-input", 500, 10_000), budgets=()) is None
    assert executor.calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "patch",
    (
        *({"max_tokens": value} for value in (0, -1, True, "32", 81)),
        *({"extra_body": value} for value in (False, [], "not-an-object")),
        *({"extra_body": {key: "replacement"}} for key in ("messages", "input", "system", "tools", "max_tokens")),
        {"previous_response_id": "unavailable"},
        {"conversation": "unavailable"},
        {"messages": [{"role": "assistant", "thinking_blocks": [{"signature": "signed"}]}]},
    ),
)
async def test_invalid_requests_fail_before_counting_or_spending(patch: Mapping[str, object]) -> None:
    executor: Final = _Executor()
    counter: Final = _Counter()
    result: Final = await _run(executor, {**_payload(), **patch}, counter=counter)
    assert isinstance(result, CompactionFailure)
    assert executor.calls.empty()
    assert counter.calls.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("recipient", (False, True))
@pytest.mark.parametrize("field_name", ("system", "instructions", "tools"))
async def test_preserved_prompt_fields_must_fit_before_spending(recipient: bool, field_name: str) -> None:
    value: Final = [{"description": "large " * 10_000}] if field_name == "tools" else "large " * 10_000
    payload: Final = _payload() if recipient else {**_payload(), field_name: value}
    budget: Final = replace(_SUMMARY, request_defaults={field_name: value}) if recipient else _SUMMARY
    executor: Final = _Executor()
    result: Final = await _run(executor, payload, budgets=(budget,))
    assert isinstance(result, CompactionFailure)
    assert executor.calls.empty()


@pytest.mark.asyncio
async def test_chunking_covers_history_and_respects_every_recipient_budget() -> None:
    budgets: Final = (_SUMMARY, ModelBudget("smaller-summary", 1_200, 32))
    executor: Final = _Executor(replies=("73",))
    result: Final = await _run(executor, budgets=budgets)
    assert isinstance(result, CompactedRequest)
    history: Final = split_compaction_history(_payload())
    assert isinstance(history, CompactionHistory)
    calls: Final = tuple(executor.calls.get_nowait() for _ in range(executor.calls.qsize()))
    assert 1 < len(calls) <= MAX_SUMMARY_CALLS
    assert "".join(messages[1]["content"] for _, messages, _, _ in calls) == history.history_text
    assert all(
        len(_serialize({"messages": messages})) <= budget.input_budget(ceiling)
        for _, messages, ceiling, _ in calls for budget in budgets
    )


@pytest.mark.asyncio
async def test_call_budget_stops_paid_work() -> None:
    executor: Final = _Executor(replies=("73",))
    state: Final = _state()
    result: Final = await _run(executor, state=state, budgets=(ModelBudget("tiny-summary", 600, 32),))
    assert isinstance(result, CompactionFailure)
    assert "call budget" in result.message
    assert state.calls == executor.calls.qsize() == MAX_SUMMARY_CALLS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply,finish,message",
    ((None, "stop", "empty"), (" ", "stop", "empty"), ("73", "length", "incomplete"),
     ("73", "tool_calls", "incomplete"), (RuntimeError("provider failed"), "stop", "summary call failed")),
)
async def test_summary_failure_is_sticky_but_does_not_block_a_fitting_fallback(
    reply: str | None | Exception, finish: str, message: str
) -> None:
    executor: Final = _Executor(replies=(reply,), finish=finish)
    state: Final = _state()
    first: Final = await _run(executor, state=state)
    assert isinstance(first, CompactionFailure) and message in first.message
    assert await _run(executor, state=state) == first
    assert await _run(executor, target=replace(_TARGET, input_limit=50_000), state=state) is None
    assert state.calls == executor.calls.qsize() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("shrinks", (False, True))
async def test_oversized_summary_is_reduced_and_recounted(shrinks: bool) -> None:
    long_summary: Final = "still too long " * 300
    executor: Final = _Executor(replies=(long_summary, "73" if shrinks else long_summary))
    counter: Final = _Counter()
    result: Final = await _run(executor, counter=counter)
    assert executor.calls.qsize() == 2
    if not shrinks:
        assert isinstance(result, CompactionFailure) and "could not reduce" in result.message
        return
    assert isinstance(result, CompactedRequest)
    counts: Final = tuple(counter.calls.get_nowait() for _ in range(counter.calls.qsize()))
    assert counts[-1][0] == _TARGET.model
    assert len(_serialize(counts[-1][1])) <= _TARGET.input_budget(32)


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("same", "changed", "expired", "expired-failure"))
async def test_memo_reuses_only_matching_history_even_after_deadline(scenario: str) -> None:
    executor: Final = _Executor()
    state: Final = _state()
    first: Final = await _run(executor, state=state)
    assert isinstance(first, CompactedRequest)
    if scenario.startswith("expired"):
        state.limit_timeout(0)
    if scenario == "expired-failure":
        state.failed(CompactionFailure("later failure"))
    payload: Final = (
        {**_payload(), "messages": [{"role": "user", "content": "changed " * 800}, {"role": "user", "content": "new"}]}
        if scenario == "changed" else _payload()
    )
    second: Final = await _run(executor, payload, state=state)
    assert isinstance(second, CompactedRequest)
    assert state.calls == executor.calls.qsize() == (2 if scenario == "changed" else 1)
    if scenario != "changed":
        assert second == first


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("expired", "timeout", "cancel"))
async def test_deadline_and_cancellation_stop_summary_work(mode: Literal["expired", "timeout", "cancel"]) -> None:
    executor: Final = _Executor(block=True)
    state: Final = _state()
    initial: Final = state.deadline
    state.limit_timeout(0 if mode == "expired" else 0.15 if mode == "timeout" else 1)
    limited: Final = state.deadline
    state.limit_timeout(10)
    assert state.deadline == limited < initial
    task: Final = asyncio.create_task(_run(executor, state=state))
    if mode == "cancel":
        await asyncio.wait_for(executor.started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.failure is None
    else:
        result: Final = await asyncio.wait_for(task, 2)
        assert isinstance(result, CompactionFailure) and "time" in result.message
    assert executor.calls.qsize() == (0 if mode == "expired" else 1)
    assert await _run(executor, target=replace(_TARGET, input_limit=50_000), state=state) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("defaults", ({"extra_body": {"tools": []}}, {"max_completion_tokens": 16}))
async def test_conflicting_summary_defaults_never_reach_executor(defaults: Mapping[str, object]) -> None:
    executor: Final = _Executor()
    result: Final = await _run(executor, budgets=(replace(_SUMMARY, request_defaults=defaults),))
    assert isinstance(result, CompactionFailure)
    assert executor.calls.empty()
