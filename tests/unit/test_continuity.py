from __future__ import annotations

from plap.plugins.continuity import _request_can_record, _request_can_restore
from plap.responses.contracts import (
    RequestCompactionItem,
    RequestItemReference,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
)


def _request(**updates: object) -> ResponseCreateRequest:
    return ResponseCreateRequest(input="current", model="plap/test", store=True).model_copy(update=updates)


def _message() -> RequestMessageItem:
    return RequestMessageItem(content="history", role="user", type="message")


def test_plain_stored_manual_history_can_record_and_restore() -> None:
    items = (_message(),)

    assert _request_can_record(_request())
    assert _request_can_restore(_request(), items)


def test_continuation_and_unstored_requests_are_ineligible() -> None:
    items = (_message(),)

    for request in (
        _request(store=False),
        _request(previous_response_id="resp_previous"),
        _request(conversation="conv_current"),
    ):
        assert not _request_can_record(request)
        assert not _request_can_restore(request, items)


def test_references_and_compaction_skip_preparation_time_restoration() -> None:
    for items in (
        (RequestItemReference(id="msg_stored", type="item_reference"),),
        (RequestCompactionItem(encrypted_content="sealed", type="compaction"),),
    ):
        assert _request_can_record(_request())
        assert not _request_can_restore(_request(), items)


def test_existing_reasoning_can_be_recorded_but_is_not_restored_over() -> None:
    items = (
        _message(),
        RequestReasoningItem(encrypted_content="sealed", id="rs_existing", summary=[], type="reasoning"),
    )

    assert _request_can_record(_request())
    assert not _request_can_restore(_request(), items)
