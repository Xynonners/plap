from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import svcs
from svcs.exceptions import ServiceNotFoundError

from plap.auth import AuthContext
from plap.bus import bus
from plap.config import CueBox
from plap.persistence import Database
from plap.plugins.continuity.matching import contains_reasoning, input_is_matchable
from plap.plugins.continuity.store import ContinuityStore
from plap.plugins.easy import bootstrap
from plap.responses.contracts import RequestInputItem, ResponseCreateRequest, ResponseStatus
from plap.responses.state import State
from plap.responses.store import PreparedRequest, ResponseStore
from plap.responses.streaming import StreamCoordinator


@dataclass(frozen=True, slots=True)
class _RequestScope:
    scope_id: UUID


def _request_can_record(request: ResponseCreateRequest) -> bool:
    return request.store is not False and request.previous_response_id is None and request.conversation_id is None


def _request_can_restore(request: ResponseCreateRequest, items: tuple[RequestInputItem, ...]) -> bool:
    return _request_can_record(request) and input_is_matchable(items) and not contains_reasoning(items)


def _continuity_store_factory(svcs_container: svcs.Container) -> ContinuityStore:
    return ContinuityStore(svcs_container.get(Database), svcs_container.get(ResponseStore))


@bootstrap.services
async def services(registry: svcs.Registry, config: CueBox) -> None:
    _ = config
    registry.register_factory(ContinuityStore, _continuity_store_factory)


@bus.listen("response.input")
async def restore(
    items: tuple[RequestInputItem, ...],
    request: ResponseCreateRequest,
    auth_context: AuthContext,
    svcs: svcs.Container,
    *,
    next,
) -> tuple[RequestInputItem, ...]:
    svcs.register_local_value(_RequestScope, _RequestScope(scope_id=auth_context.scope_id))
    restored = items
    if _request_can_restore(request, items):
        restored = await svcs.get(ContinuityStore).restore(auth_context, items)
    return tuple(await next(items=restored))


@bus.listen("response.finish")
async def record(state: State, status: ResponseStatus, *, next) -> None:
    await next(state=state, status=status)
    if not _request_can_record(state.request):
        return
    try:
        request_scope = state.svcs.get(_RequestScope)
    except ServiceNotFoundError:
        return
    prepared = state.svcs.get(PreparedRequest)
    coordinator = state.svcs.get(StreamCoordinator)
    await state.svcs.get(ContinuityStore).record(
        request_scope.scope_id,
        coordinator.response_id,
        (
            *(resolved.item for resolved in prepared.resolved_current_input_items),
            *coordinator.current_response().output,
        ),
    )


__all__ = []
