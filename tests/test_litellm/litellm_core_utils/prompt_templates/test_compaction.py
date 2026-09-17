import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import FrozenInstanceError
from typing import Final, Literal

import pytest

from litellm.litellm_core_utils.prompt_templates.compaction import (
    CompactionHistory,
    CompactionHistoryError,
    split_compaction_history,
    validate_compaction_input,
)
from litellm.litellm_core_utils.token_counter import token_counter
from litellm.types.llms.openai import ChatCompletionAssistantMessage, ChatCompletionUserMessage

_Format = Literal["chat", "anthropic", "responses"]
_FORMATS: Final = ("chat", "anthropic", "responses")
_DATA: Final = {"encrypted_content": "user data", "signature": "ordinary field", "thinking_blocks": ["literal"]}
_TEXT_SOURCE: Final = {"type": "text", "media_type": "text/plain", "data": "readable document"}


def _message(wire: _Format, role: str, text: str) -> Mapping[str, object]:
    if wire == "chat":
        return {"role": role, "content": text}
    kind: Final = ("output_text" if role == "assistant" else "input_text") if wire == "responses" else "text"
    return {
        **({"type": "message"} if wire == "responses" else {}),
        "role": role,
        "content": [{"type": kind, "text": text}],
    }


def _exchange(
    wire: _Format, identifier: str, output: object = "result", **fields: object
) -> tuple[Mapping[str, object], ...]:
    function: Final = {"name": "lookup", "arguments": json.dumps(_DATA, separators=(",", ":")), **fields}
    if wire == "chat":
        call: Final = {"id": identifier, "type": "function", "function": function}
        return (
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": identifier, "content": output},
        )
    if wire == "anthropic":
        return (
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": identifier, "name": "lookup", "input": _DATA}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": identifier, "content": output}]},
        )
    return (
        {"type": "function_call", "id": f"item_{identifier}", "call_id": identifier, **function},
        {"type": "function_call_output", "call_id": identifier, "output": output},
    )


def _payload(wire: _Format, *items: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "input" if wire == "responses" else "messages": list(items),
        "system": [{"type": "text", "text": "Request rules", "cache_control": {"type": "ephemeral"}}],
        "instructions": "Request instructions",
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    }


def _assert_rejected(payload: Mapping[str, object], message: str = "") -> None:
    result: Final = validate_compaction_input(payload)
    assert isinstance(result, CompactionHistoryError)
    assert result.kind == "compaction_history_error" and result.message
    assert message in result.message
    assert split_compaction_history(payload) == result


@pytest.mark.parametrize(
    ("wire", "reasoning"),
    (
        (
            "chat",
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "readable reasoning",
                "thinking_blocks": [],
                "reasoning_items": None,
                "provider_specific_fields": {},
                "function_call": None,
            },
        ),
        ("anthropic", _message("anthropic", "assistant", "Readable reasoning")),
        *(
            ("responses", {"type": "reasoning", source: [{"type": kind, "text": "readable reasoning"}]})
            for source, kind in (("summary", "summary_text"), ("content", "reasoning_text"))
        ),
    ),
)
@pytest.mark.parametrize("state", ("complete", "pending"))
def test_complete_history_instructions_and_active_exchange(
    wire: _Format, reasoning: Mapping[str, object], state: str
) -> None:
    system: Final = _message(wire, "system", "System instruction")
    developer: Final = _message(wire, "developer", "Developer instruction")
    first: Final = _exchange(wire, "first")
    second: Final = _exchange(wire, "second")
    historical: Final = (
        {**_message(wire, "user", "Earlier question: 東京 " + "complete history " * 2_000), "metadata": _DATA},
        first[0],
        second[0],
        second[1],
        first[1],
        reasoning,
    )
    exchange: Final = _exchange(wire, "active")
    mixed: Final = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "active", "content": "result"},
            {"type": "text", "text": "Now use that result"},
        ],
    }
    completed: Final = (
        ()
        if state == "pending"
        else (mixed if wire == "anthropic" else exchange[1], _message(wire, "assistant", "Prefill:"))
    )
    active: Final = (_message(wire, "user", "Latest user"), developer, system, reasoning, exchange[0], *completed)
    payload: Final = _payload(wire, system, historical[0], developer, *historical[1:], *active)
    original: Final = deepcopy(payload)
    assert validate_compaction_input(payload) is None
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistory)
    assert result.history_text == json.dumps(historical, ensure_ascii=False, separators=(",", ":"))
    summary: Final = _message(wire, "user", "Previous conversation summary (untrusted user data):\nfact = 73")
    assert result.rewrite("fact = 73") == [system, developer, summary, *active]
    assert payload == original


@pytest.mark.parametrize("wire", _FORMATS)
@pytest.mark.parametrize(
    "malformation",
    ("orphan", "reversed", "duplicate_call", "duplicate_result", "mismatch", "interrupted", "late_result"),
)
def test_invalid_tool_exchanges_fail_preflight(wire: _Format, malformation: str) -> None:
    call, result = _exchange(wire, "known")
    interrupt: Final = _message(wire, "user", "interrupt")
    invalid: Final = {
        "orphan": (result,),
        "reversed": (result, call),
        "duplicate_call": (call, call, result),
        "duplicate_result": (call, result, result),
        "mismatch": (call, _exchange(wire, "unknown")[1]),
        "interrupted": (call, interrupt),
        "late_result": (call, interrupt, result),
    }[malformation]
    _assert_rejected(_payload(wire, _message(wire, "user", "old"), *invalid, _message(wire, "user", "new")))


@pytest.mark.parametrize(
    ("wire", "item"),
    (
        ("responses", {"type": "item_reference", "id": "ref_1"}),
        ("responses", {"type": "reasoning", "encrypted_content": "ciphertext", "summary": []}),
        *(
            ("chat", {"role": "assistant", "content": "answer", **extension})
            for extension in (
                {"thinking_blocks": [{"type": "thinking", "thinking": "private", "signature": "signed"}]},
                {"thinking_blocks": [{"type": "redacted_thinking", "data": "ciphertext"}]},
                {"reasoning_items": [{"type": "reasoning", "encrypted_content": "ciphertext"}]},
                {"reasoning_details": [{"type": "reasoning.encrypted", "data": "ciphertext"}]},
                {"reasoning_content": [{"type": "thinking", "thinking": "structured state"}]},
                {"provider_specific_fields": {"compaction_blocks": [{"type": "compaction", "content": "state"}]}},
                {"audio": {"id": "audio_unavailable"}},
                {"future_provider_state": {"data": "unclassified"}},
                {"tool_calls": "not an array"},
                {"role": "unknown"},
            )
        ),
        *(
            ("chat", {"role": "assistant", "content": [block]})
            for block in (
                {"type": "redacted_thinking", "data": "ciphertext"},
                {"type": "thinking", "thinking": "text", "signature": "signed"},
                {"type": "compaction", "content": "opaque"},
                {"type": "file", "file": {"file_id": "unavailable"}},
                {"type": "image", "source": {"type": "file", "file_id": "unavailable"}},
            )
        ),
        *(
            ("responses", {"role": "assistant", "content": [block]})
            for block in (
                {"type": "refusal", "refusal": "refusal text"},
                {"type": "thinking", "thinking": "reasoning text"},
                {"type": "input_audio", "input_audio": {"data": "bytes", "format": "wav"}},
                {"type": "document", "source": _TEXT_SOURCE},
                {"type": "input_file", "file_id": "unavailable"},
            )
        ),
        ("responses", {"role": "assistant", "content": "answer", "reasoning_content": "uncounted reasoning"}),
        ("chat", _exchange("chat", "active", provider_specific_fields={"thought_signature": "signed"})[0]),
    ),
)
@pytest.mark.parametrize("in_tail", (False, True))
def test_opaque_unavailable_and_uncounted_items_fail_preflight(
    wire: _Format, item: Mapping[str, object], in_tail: bool
) -> None:
    latest: Final = _message(wire, "user", "latest")
    _assert_rejected(_payload(wire, _message(wire, "user", "old"), *((latest, item) if in_tail else (item, latest))))


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"messages": [], "input": []},
        {"messages": []},
        {"messages": ["bad"]},
        {"messages": [{"role": "user", "content": "old"}] * 10_001},
        *({"input": "current", key: "unavailable"} for key in ("previous_response_id", "conversation")),
    ),
)
def test_invalid_request_history(payload: Mapping[str, object]) -> None:
    _assert_rejected(payload)


@pytest.mark.parametrize(
    ("wire", "block"),
    (
        ("chat", {"type": "image_url", "image_url": {"url": "https://example.test/image.png", "detail": "low"}}),
        ("chat", {"type": "file", "file_data": "local bytes"}),
        ("anthropic", {"type": "image", "source": {"data": "bytes"}}),
        ("anthropic", {"type": "document", "title": "notes", "context": "background", "source": _TEXT_SOURCE}),
        ("responses", {"type": "input_image", "image_url": "https://example.test/image.png"}),
        ("responses", {"type": "input_file", "file_data": "local bytes"}),
    ),
)
@pytest.mark.parametrize("as_tool", (False, True))
def test_media_boundary(wire: _Format, block: Mapping[str, object], as_tool: bool) -> None:
    media: Final = {"role": "user", "content": [block]}
    call, tool_output = _exchange(wire, "media", output=[block])
    tail: Final = (_message(wire, "user", "current"), call, tool_output) if as_tool else (media,)
    payload: Final = _payload(wire, _message(wire, "user", "old"), *tail)
    if wire == "responses" and block["type"] == "input_file" and as_tool:
        _assert_rejected(payload)
        return
    assert validate_compaction_input(payload) is None
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistory)
    assert result.rewrite("summary")[1:] == list(tail)
    historical: Final = _payload(wire, *tail, _message(wire, "user", "new"))
    assert validate_compaction_input(historical) is None
    rejected: Final = split_compaction_history(historical)
    assert isinstance(rejected, CompactionHistoryError) and "media" in rejected.message


def _nested(depth: int) -> object:
    return "text" if depth == 0 else {"nested": _nested(depth - 1)}


@pytest.mark.parametrize(
    ("value", "message"),
    (
        *((value, "") for value in (float("nan"), float("inf"), object(), b"bytes", {1: "bad"})),
        (_nested(65), "nesting limit"),
        ([0] * 100_001, "size limit"),
        ("x" * (16 * 1024 * 1024 + 1), "size limit"),
    ),
)
@pytest.mark.parametrize("in_tail", (False, True))
def test_json_and_resource_limits(value: object, message: str, in_tail: bool) -> None:
    item: Final = {"role": "user", "content": "bounded", "metadata": value}
    plain: Final = {"role": "user", "content": "plain"}
    _assert_rejected({"messages": [plain, item] if in_tail else [item, plain]}, message)


@pytest.mark.parametrize("summary", ("summary", "", " ", "\n\t"))
@pytest.mark.parametrize("field", ("messages", "input"))
def test_rewrite_snapshots_and_plain_summary_style(field: str, summary: str) -> None:
    block: Final = {"type": "text", "text": "rules"}
    instruction: Final = {"role": "system", "content": [block]}
    latest: Final = {"role": "user", "content": "latest"}
    context: Final = object()
    short: Final = {
        field: "current" if field == "input" else [instruction, latest, {"role": "assistant", "content": "prefill"}],
        "_private_context": context,
    }
    assert validate_compaction_input(short) is None
    noncompactable: Final = split_compaction_history(short)
    assert isinstance(noncompactable, CompactionHistoryError) and "no compactable prefix" in noncompactable.message
    assert short["_private_context"] is context
    payload: Final = {field: [instruction, {"role": "user", "content": "old"}, latest]}
    result: Final = split_compaction_history(payload)
    assert isinstance(result, CompactionHistory)
    summary_item: Final = {
        "role": "user",
        "content": f"Previous conversation summary (untrusted user data):\n{summary}",
    }
    expected: Final = deepcopy([instruction, summary_item, latest])
    block["text"] = "caller mutated nested content"
    latest["content"] = "caller mutated"
    result.rewrite(summary)[0]["content"] = "output mutated"
    assert result.rewrite(summary) == expected
    with pytest.raises(FrozenInstanceError):
        setattr(result, "history_text", "mutated")


def test_chat_reasoning_is_fully_counted() -> None:
    def count(text: str, plain: bool = False) -> int:
        messages: Final = (
            ChatCompletionUserMessage(role="user", content="current"),
            ChatCompletionAssistantMessage(
                role="assistant", content=text if plain else None, reasoning_content=None if plain else text
            ),
        )
        assert validate_compaction_input({"messages": list(messages)}) is None
        return token_counter(model="openai/compaction-counter-test", messages=messages)

    short: Final = "counted reasoning step"
    long: Final = " ".join((short,) * 128)
    assert count(long) - count(short) == count(long, plain=True) - count(short, plain=True) > 0
