from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from plap.plugins.continuity.matching import build_history, overlap_length, restore_reasoning
from plap.responses.contracts import (
    OutputTextContent,
    RequestFunctionCallItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseFunctionCallItem,
    ResponseMessageItem,
    ResponseReasoningItem,
)


def _message(content: str, role: str = "user") -> RequestMessageItem:
    return RequestMessageItem(content=content, role=role, type="message")


def _request_reasoning(reasoning_id: str) -> RequestReasoningItem:
    return RequestReasoningItem(encrypted_content=f"sealed-{reasoning_id}", id=reasoning_id, summary=[], type="reasoning")


def _response_reasoning(reasoning_id: str) -> ResponseReasoningItem:
    return ResponseReasoningItem(
        encrypted_content=f"sealed-{reasoning_id}",
        id=reasoning_id,
        status="completed",
        summary=[],
        type="reasoning",
    )


def _response_message(content: str, message_id: str) -> ResponseMessageItem:
    return ResponseMessageItem(
        content=[OutputTextContent(annotations=[], text=content, type="output_text")],
        id=message_id,
        role="assistant",
        status="completed",
        type="message",
    )


def test_complete_history_restores_reasoning_after_system_prefix() -> None:
    source_items = (
        _message("system", role="system"),
        _message("first"),
        _response_reasoning("rs_first"),
        _response_message("first answer", "msg_generated"),
    )
    incoming = (
        _message("system", role="system"),
        _message("first"),
        _message("first answer", role="assistant"),
        _message("second"),
    )
    source = build_history(source_items)
    current = build_history(incoming)

    assert source is not None
    assert current is not None
    overlap = overlap_length(source.hashes, current.hashes)
    restored = restore_reasoning(incoming, source, overlap)

    assert restored is not None
    assert [item.type for item in restored] == ["message", "message", "reasoning", "message", "message"]
    assert isinstance(restored[2], RequestReasoningItem)
    assert restored[2].id == "rs_first"


def test_rolling_suffix_restores_only_reasoning_after_user_boundary() -> None:
    source_items = (
        _message("system", role="system"),
        _message("first"),
        _request_reasoning("rs_first"),
        _message("first answer", role="assistant"),
        _message("second"),
        _request_reasoning("rs_second_a"),
        _request_reasoning("rs_second_b"),
        _message("second answer", role="assistant"),
    )
    incoming = (
        _message("second"),
        _message("second answer", role="assistant"),
        _message("third"),
    )
    source = build_history(source_items)
    current = build_history(incoming)

    assert source is not None
    assert current is not None
    restored = restore_reasoning(incoming, source, overlap_length(source.hashes, current.hashes))

    assert restored is not None
    assert [item.type for item in restored] == ["message", "reasoning", "reasoning", "message", "message"]
    assert [item.id for item in restored[1:3] if isinstance(item, RequestReasoningItem)] == ["rs_second_a", "rs_second_b"]


def test_rolling_suffix_rejects_assistant_boundary() -> None:
    source_items = (
        _message("first"),
        _request_reasoning("rs_first"),
        _message("first answer", role="assistant"),
        _message("second"),
        _request_reasoning("rs_second"),
    )
    incoming = (
        _message("first answer", role="assistant"),
        _message("second"),
        _message("third"),
    )
    source = build_history(source_items)
    current = build_history(incoming)

    assert source is not None
    assert current is not None
    overlap = overlap_length(source.hashes, current.hashes)

    assert overlap == 2
    assert restore_reasoning(incoming, source, overlap) is None


def test_tool_call_round_trip_ignores_empty_assistant_wrapper_and_output_metadata() -> None:
    source = build_history(
        (
            ResponseFunctionCallItem(
                arguments='{"path":"README.md"}',
                call_id="call_1",
                id="fc_generated",
                name="read_file",
                status="completed",
                type="function_call",
            ),
        )
    )
    incoming = build_history(
        (
            _message("", role="assistant"),
            RequestFunctionCallItem(
                arguments='{ "path": "README.md" }',
                call_id="call_1",
                name="read_file",
                type="function_call",
            ),
        )
    )
    different_call = build_history(
        (
            RequestFunctionCallItem(
                arguments='{"path":"README.md"}',
                call_id="call_2",
                name="read_file",
                type="function_call",
            ),
        )
    )

    assert source is not None
    assert incoming is not None
    assert different_call is not None
    assert source.hashes == incoming.hashes
    assert source.hashes != different_call.hashes


@given(
    stored=st.lists(st.binary(min_size=1, max_size=4), max_size=20),
    incoming=st.lists(st.binary(min_size=1, max_size=4), max_size=20),
)
def test_overlap_matches_reference_implementation(stored: list[bytes], incoming: list[bytes]) -> None:
    expected = 0
    for length in range(1, min(len(stored), len(incoming)) + 1):
        if stored[-length:] == incoming[:length]:
            expected = length

    assert overlap_length(stored, incoming) == expected
