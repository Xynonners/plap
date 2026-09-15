from __future__ import annotations

import json
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import text

from plap.auth import AuthContext
from plap.keyring import SealingKeyring
from plap.persistence import Database
from plap.plugins.continuity.store import ContinuityStore
from plap.responses.contracts import (
    OutputTextContent,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
    ResponseMessageItem,
    ResponseObject,
    ResponseOutputItem,
    ResponseReasoningItem,
)
from plap.responses.ingest.ingest import ingest_response_request
from plap.responses.ingest.models import ReasoningCheckpoint, ReasoningPayload
from plap.responses.ingest.sealing import seal_reasoning_payload
from plap.responses.store import ResponseStore


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"c" * 32,))


def _message(content: str, *, role: str = "user") -> RequestMessageItem:
    return RequestMessageItem(content=content, role=role, type="message")


async def _persist_response(
    responses: ResponseStore,
    prepared,
    response_id: str,
    output: list[ResponseOutputItem],
) -> None:
    response = ResponseObject(
        created_at=0,
        id=response_id,
        model="plap/test",
        output=[],
        status="in_progress",
    )
    await responses.begin_response(prepared, response)
    for output_index, item in enumerate(output):
        await responses.append_output_item(
            prepared,
            response_id,
            output_index,
            item.model_dump(mode="json", exclude_none=True),
        )
    await responses.finish_response(
        prepared,
        response.model_copy(update={"completed_at": 1, "output": output, "status": "completed"}),
    )


async def test_continuity_records_restores_and_authenticates_sealed_reasoning(
    db_session_maker,
    test_config,
) -> None:
    scope_id = uuid4()
    auth_context = AuthContext(api_key_id=uuid4(), organization_id=None, user_id=scope_id)
    first_user = _message("first question")
    reasoning_payload = ReasoningPayload(
        id="rs_source",
        previous_reasoning_id=None,
        previous_compaction_id=None,
        state=ReasoningCheckpoint(memory={"restored": True}, enabled={"main"}, threads={}),
        main=[],
    )
    sealed = seal_reasoning_payload(reasoning_payload, keyring=_keyring())
    response_reasoning = ResponseReasoningItem(
        encrypted_content=sealed,
        id="rs_source",
        status="completed",
        summary=[],
        type="reasoning",
    )
    response_message = ResponseMessageItem(
        content=[OutputTextContent(annotations=[], text="first answer", type="output_text")],
        id="msg_source",
        role="assistant",
        status="completed",
        type="message",
    )

    async with db_session_maker() as session:
        await session.execute(
            text(
                """
                select responses.create_response_record(
                  :scope_id,
                  'resp_source',
                  null,
                  cast(:input_items as jsonb),
                  cast(:output_items as jsonb),
                  :retention
                )
                """
            ),
            {
                "scope_id": scope_id,
                "input_items": json.dumps(ResponseStore._stored_input_payloads([first_user])),
                "output_items": json.dumps(
                    [
                        response_reasoning.model_dump(mode="json", exclude_none=True),
                        response_message.model_dump(mode="json", exclude_none=True),
                    ]
                ),
                "retention": timedelta(days=2),
            },
        )
        await session.commit()

    database = Database(test_config.database_url)
    responses = ResponseStore(database)
    continuity = ContinuityStore(database, responses)
    try:
        await continuity.record(
            scope_id,
            "resp_source",
            (first_user, response_reasoning, response_message),
        )
        async with database.connection() as connection:
            result = await connection.execute(
                text(
                    """
                    select cardinality(index.visible_item_hashes) as hash_count,
                           index.lookup_hash = index.visible_item_hashes[cardinality(index.visible_item_hashes)]
                             as lookup_matches,
                           index.expires_at = lease.expires_at as expiry_matches
                      from continuity.response_indexes index
                      join responses.response_leases lease
                        on lease.scope_id = index.scope_id
                       and lease.response_id = index.source_response_id
                       and lease.owner_type = 'response'
                       and lease.owner_id = index.source_response_id
                     where index.scope_id = :scope_id
                       and index.source_response_id = 'resp_source'
                    """
                ),
                {"scope_id": scope_id},
            )
            try:
                index_row = result.mappings().one()
            finally:
                result.close()

        incoming = (
            first_user,
            _message("first answer", role="assistant"),
            _message("second question"),
        )
        restored = await continuity.restore(auth_context, incoming)
        ingested = await ingest_response_request(
            ResponseCreateRequest(input=list(restored), model="plap/test"),
            keyring=_keyring(),
            thread_codes={"main": 0},
        )

        deleted = await responses.delete_response(auth_context, "resp_source")
        after_delete = await continuity.restore(auth_context, incoming)
    finally:
        await database.dispose_all()

    async with db_session_maker() as session:
        index_count = (
            await session.execute(
                text(
                    """
                    select count(*)
                      from continuity.response_indexes
                     where scope_id = :scope_id
                       and source_response_id = 'resp_source'
                    """
                ),
                {"scope_id": scope_id},
            )
        ).scalar_one()

    assert index_row["hash_count"] == 2
    assert index_row["lookup_matches"]
    assert index_row["expiry_matches"]
    assert [item.type for item in restored] == ["message", "reasoning", "message", "message"]
    restored_reasoning = restored[1]
    assert isinstance(restored_reasoning, RequestReasoningItem)
    assert restored_reasoning.encrypted_content == sealed
    assert ingested.last_reasoning_id == "rs_source"
    assert ingested.memory == {"restored": True}
    assert deleted
    assert after_delete == incoming
    assert index_count == 0


async def test_continuity_keeps_but_does_not_consume_nonterminal_source(
    db_session_maker,
    test_config,
) -> None:
    scope_id = uuid4()
    first_user = _message("first question")
    response_reasoning = ResponseReasoningItem(
        encrypted_content="not-opened-while-nonterminal",
        id="rs_pending",
        status="in_progress",
        summary=[],
        type="reasoning",
    )
    response_message = ResponseMessageItem(
        content=[OutputTextContent(annotations=[], text="partial answer", type="output_text")],
        id="msg_pending",
        role="assistant",
        status="in_progress",
        type="message",
    )

    async with db_session_maker() as session:
        await session.execute(
            text(
                """
                select responses.create_response_record(
                  :scope_id,
                  'resp_pending',
                  null,
                  cast(:input_items as jsonb),
                  cast(:output_items as jsonb),
                  :retention,
                  'in_progress'
                )
                """
            ),
            {
                "scope_id": scope_id,
                "input_items": json.dumps(ResponseStore._stored_input_payloads([first_user])),
                "output_items": json.dumps(
                    [
                        response_reasoning.model_dump(mode="json", exclude_none=True),
                        response_message.model_dump(mode="json", exclude_none=True),
                    ]
                ),
                "retention": timedelta(days=2),
            },
        )
        await session.commit()

    database = Database(test_config.database_url)
    continuity = ContinuityStore(database, ResponseStore(database))
    incoming = (
        first_user,
        _message("partial answer", role="assistant"),
        _message("second question"),
    )
    try:
        await continuity.record(
            scope_id,
            "resp_pending",
            (first_user, response_reasoning, response_message),
        )
        restored = await continuity.restore(
            AuthContext(api_key_id=uuid4(), organization_id=None, user_id=scope_id),
            incoming,
        )
        async with database.connection() as connection:
            result = await connection.execute(
                text(
                    """
                    select count(*)
                      from continuity.response_indexes
                     where scope_id = :scope_id
                       and source_response_id = 'resp_pending'
                    """
                ),
                {"scope_id": scope_id},
            )
            try:
                index_count = result.scalar_one()
            finally:
                result.close()
    finally:
        await database.dispose_all()

    assert restored == incoming
    assert index_count == 1


async def test_implicit_reasoning_replay_remains_a_continuity_source_after_reasoning_is_stripped(
    db_session_maker,
    test_config,
) -> None:
    scope_id = uuid4()
    auth_context = AuthContext(api_key_id=uuid4(), organization_id=None, user_id=scope_id)
    database = Database(test_config.database_url)
    responses = ResponseStore(database)
    continuity = ContinuityStore(database, responses)
    first_user = _message("first question")
    first_payload = ReasoningPayload(
        id="rs_first",
        previous_reasoning_id=None,
        previous_compaction_id=None,
        state=ReasoningCheckpoint(memory={"first": True}, enabled={"main"}, threads={}),
        main=[],
    )
    first_reasoning = ResponseReasoningItem(
        encrypted_content=seal_reasoning_payload(first_payload, keyring=_keyring()),
        id="rs_first",
        status="completed",
        summary=[],
        type="reasoning",
    )
    first_answer = ResponseMessageItem(
        content=[OutputTextContent(annotations=[], text="first answer", type="output_text")],
        id="msg_first",
        role="assistant",
        status="completed",
        type="message",
    )
    second_payload = ReasoningPayload(
        id="rs_second",
        previous_reasoning_id=None,
        previous_compaction_id=None,
        state=ReasoningCheckpoint(
            memory={"first": True, "second": True},
            enabled={"main"},
            threads={},
        ),
        main=[],
    )
    second_reasoning = ResponseReasoningItem(
        encrypted_content=seal_reasoning_payload(second_payload, keyring=_keyring()),
        id="rs_second",
        status="completed",
        summary=[],
        type="reasoning",
    )
    second_answer = ResponseMessageItem(
        content=[OutputTextContent(annotations=[], text="second answer", type="output_text")],
        id="msg_second",
        role="assistant",
        status="completed",
        type="message",
    )

    try:
        prepared_first = await responses.prepare_request(
            auth_context,
            ResponseCreateRequest(input=[first_user], model="plap/test"),
        )
        await _persist_response(responses, prepared_first, "resp_first", [first_reasoning, first_answer])

        prepared_second = await responses.prepare_request(
            auth_context,
            ResponseCreateRequest(
                input=[
                    first_user,
                    RequestReasoningItem(
                        encrypted_content=None,
                        id="rs_first",
                        status="completed",
                        summary=[],
                        type="reasoning",
                    ),
                    _message("first answer", role="assistant"),
                    _message("second question"),
                ],
                model="plap/test",
            ),
        )
        await _persist_response(responses, prepared_second, "resp_second", [second_reasoning, second_answer])
        second_inputs = await responses.list_input_items(
            auth_context,
            "resp_second",
            after=None,
            limit=None,
            order=None,
        )
        assert isinstance(second_inputs.data[1], ResponseReasoningItem)
        assert second_inputs.data[1].encrypted_content == first_reasoning.encrypted_content
        await continuity.record(
            scope_id,
            "resp_second",
            (
                *[resolved.item for resolved in prepared_second.resolved_current_input_items],
                second_reasoning,
                second_answer,
            ),
        )
        duplicate_lookup = await responses.prepare_request(
            auth_context,
            ResponseCreateRequest(
                input=[RequestReasoningItem(id="rs_first", status="completed", summary=[], type="reasoning")],
                model="plap/test",
            ),
        )
        assert isinstance(duplicate_lookup.resolved_current_input_items[0].item, RequestReasoningItem)
        assert duplicate_lookup.resolved_current_input_items[0].item.encrypted_content == first_reasoning.encrypted_content
        assert await responses.delete_response(auth_context, "resp_first")

        stripped = (
            first_user,
            _message("first answer", role="assistant"),
            _message("second question"),
            _message("second answer", role="assistant"),
            _message("third question"),
        )
        restored = await continuity.restore(auth_context, stripped)
        ingested = await ingest_response_request(
            ResponseCreateRequest(input=list(restored), model="plap/test"),
            keyring=_keyring(),
            thread_codes={"main": 0},
        )

        async with database.connection() as connection:
            result = await connection.execute(
                text(
                    """
                    select first_item.payload_id = second_item.payload_id as reused_payload,
                           (
                             select count(*)
                               from responses.payloads payload
                              where payload.scope_id = :scope_id
                                and payload.item_id = 'rs_first'
                                and payload.item_type = 'reasoning'
                           ) as reasoning_payload_count,
                           (
                             select count(*)
                               from responses.payloads payload
                              where payload.scope_id = :scope_id
                                and payload.item_id = 'rs_first'
                                and payload.item_type = 'item_reference'
                           ) as reference_payload_count
                      from responses.response_output_items first_item
                      join responses.response_input_items second_item
                        on second_item.scope_id = first_item.scope_id
                       and second_item.response_id = 'resp_second'
                       and second_item.input_index = 1
                     where first_item.scope_id = :scope_id
                       and first_item.response_id = 'resp_first'
                       and first_item.output_index = 0
                    """
                ),
                {"scope_id": scope_id},
            )
            try:
                storage_row = result.mappings().one()
            finally:
                result.close()
    finally:
        await database.dispose_all()

    assert storage_row["reused_payload"]
    assert storage_row["reasoning_payload_count"] == 1
    assert storage_row["reference_payload_count"] == 0
    assert [item.type for item in restored] == [
        "message",
        "reasoning",
        "message",
        "message",
        "reasoning",
        "message",
        "message",
    ]
    assert [item.id for item in restored if isinstance(item, RequestReasoningItem)] == ["rs_first", "rs_second"]
    assert ingested.last_reasoning_id == "rs_second"
    assert ingested.memory == {"first": True, "second": True}
