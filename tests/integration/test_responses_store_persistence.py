from __future__ import annotations

import json
from uuid import UUID, uuid4

from sqlalchemy import text

from plap.auth import AuthContext
from plap.persistence import Database
from plap.responses.contracts import RequestItemReference, RequestMessageItem, ResponseCreateRequest
from plap.responses.store import ResponseStore


async def _create_response(session, scope_id: UUID, response_id: str, input_items: list[dict[str, object]]) -> None:
    await session.execute(
        text(
            """
            select responses.create_response_record(
              :scope_id,
              :response_id,
              null,
              cast(:input_items as jsonb),
              '[]'::jsonb,
              null
            )
            """
        ),
        {
            "scope_id": scope_id,
            "response_id": response_id,
            "input_items": json.dumps(input_items),
        },
    )


async def test_anonymous_inputs_share_payload_without_sharing_public_item_ids(db_session_maker, test_config) -> None:
    scope_id = uuid4()
    item = RequestMessageItem(content="hello", role="user", type="message")
    input_items = ResponseStore._stored_input_payloads([item])

    async with db_session_maker() as session:
        await _create_response(session, scope_id, "resp_first", input_items)
        await _create_response(session, scope_id, "resp_second", input_items)
        await session.commit()

        payload_rows = (
            await session.execute(
                text(
                    """
                    select payload_json, refcount
                      from responses.payloads
                     where scope_id = :scope_id
                    """
                ),
                {"scope_id": scope_id},
            )
        ).all()

    assert payload_rows == [(input_items[0], 2)]
    assert "id" not in payload_rows[0].payload_json

    database = Database(test_config.database_url)
    store = ResponseStore(database)
    auth_context = AuthContext(api_key_id=uuid4(), organization_id=None, user_id=scope_id)
    try:
        first_page = await store.list_input_items(auth_context, "resp_first", after=None, limit=None, order=None)
        second_page = await store.list_input_items(auth_context, "resp_second", after=None, limit=None, order=None)
        referenced = await store.prepare_request(
            auth_context,
            ResponseCreateRequest(
                input=[RequestItemReference(id=first_page.data[0].id, type="item_reference")],
                model="plap/test",
                store=True,
            ),
        )
        replayed = await store.prepare_request(
            auth_context,
            ResponseCreateRequest(
                input="next",
                model="plap/test",
                previous_response_id="resp_first",
                store=True,
            ),
        )
    finally:
        await database.dispose_all()

    first_item = first_page.data[0]
    second_item = second_page.data[0]
    assert first_item.id == "in_resp_first_0"
    assert second_item.id == "in_resp_second_0"
    assert first_item.id != second_item.id

    assert isinstance(referenced.execution_request.input, list)
    resolved_reference = referenced.execution_request.input[0]
    assert isinstance(resolved_reference, RequestMessageItem)
    assert resolved_reference.id == first_item.id
    assert resolved_reference.content == "hello"

    assert isinstance(replayed.execution_request.input, list)
    replayed_item = replayed.execution_request.input[0]
    assert isinstance(replayed_item, RequestMessageItem)
    assert replayed_item.id == first_item.id
    assert replayed_item.content == "hello"
