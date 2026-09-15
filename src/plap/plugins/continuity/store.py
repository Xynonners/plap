from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import cast
from uuid import UUID

import structlog
from sqlalchemy import LargeBinary, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncConnection

from plap.auth import AuthContext
from plap.persistence import Database
from plap.plugins.continuity.matching import (
    HistoryItem,
    PublicHistory,
    build_history,
    has_sealed_reasoning,
    overlap_length,
    restore_reasoning,
)
from plap.responses.contracts import RequestInputItem
from plap.responses.store import ResponseStore

logger = structlog.stdlib.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Candidate:
    source_response_id: str
    hashes: tuple[bytes, ...]
    recorded_at: datetime


class ContinuityStore:
    def __init__(self, database: Database, responses: ResponseStore) -> None:
        self._database = database
        self._responses = responses

    @staticmethod
    async def _cleanup_expired(connection: AsyncConnection) -> None:
        result = await connection.execute(text("delete from continuity.response_indexes where expires_at <= clock_timestamp()"))
        result.close()

    async def _candidate_rows(self, scope_id: UUID, lookup_hashes: tuple[bytes, ...]) -> list[_Candidate]:
        statement = text(
            """
            select source_response_id, visible_item_hashes, recorded_at
              from continuity.response_indexes
             where scope_id = :scope_id
               and expires_at > clock_timestamp()
               and lookup_hash in :lookup_hashes
            """
        ).bindparams(bindparam("lookup_hashes", expanding=True))
        async with self._database.connection() as connection:
            result = await connection.execute(statement, {"scope_id": scope_id, "lookup_hashes": lookup_hashes})
            try:
                rows = result.mappings().all()
            finally:
                result.close()

        return [
            _Candidate(
                source_response_id=str(row["source_response_id"]),
                hashes=tuple(bytes(item_hash) for item_hash in row["visible_item_hashes"]),
                recorded_at=cast(datetime, row["recorded_at"]),
            )
            for row in rows
        ]

    async def _delete(self, scope_id: UUID, source_response_id: str) -> None:
        async with self._database.connection_transaction() as connection:
            result = await connection.execute(
                text(
                    """
                    delete from continuity.response_indexes
                     where scope_id = :scope_id
                       and source_response_id = :source_response_id
                    """
                ),
                {"scope_id": scope_id, "source_response_id": source_response_id},
            )
            result.close()

    async def _ranked_candidates(self, scope_id: UUID, incoming: PublicHistory) -> list[tuple[int, _Candidate]]:
        lookup_hashes = tuple(dict.fromkeys(incoming.hashes))
        ranked = []
        for candidate in await self._candidate_rows(scope_id, lookup_hashes):
            overlap = overlap_length(candidate.hashes, incoming.hashes)
            if overlap:
                ranked.append((overlap, candidate))
        ranked.sort(
            key=lambda ranked_candidate: (
                ranked_candidate[0],
                ranked_candidate[1].recorded_at,
                ranked_candidate[1].source_response_id,
            ),
            reverse=True,
        )
        return ranked

    async def restore(
        self,
        auth_context: AuthContext,
        items: tuple[RequestInputItem, ...],
    ) -> tuple[RequestInputItem, ...]:
        incoming = build_history(items)
        if incoming is None or not incoming.hashes:
            return items

        for overlap, candidate in await self._ranked_candidates(auth_context.scope_id, incoming):
            replay = await self._responses.get_replay(auth_context, candidate.source_response_id)
            if replay is None:
                await self._delete(auth_context.scope_id, candidate.source_response_id)
                logger.info("continuity.source_stale", source_response_id=candidate.source_response_id)
                continue
            if not replay.is_replayable:
                continue
            source_history = build_history(replay.items)
            if source_history is None or source_history.hashes != candidate.hashes:
                await self._delete(auth_context.scope_id, candidate.source_response_id)
                logger.warning("continuity.source_mismatch", source_response_id=candidate.source_response_id)
                continue
            restored = restore_reasoning(items, source_history, overlap)
            if restored is None:
                continue
            logger.info(
                "continuity.restored",
                overlap=overlap,
                source_response_id=candidate.source_response_id,
            )
            return restored
        return items

    async def record(
        self,
        scope_id: UUID,
        source_response_id: str,
        items: tuple[HistoryItem, ...],
    ) -> None:
        history = build_history(items)
        if history is None or not history.hashes or not has_sealed_reasoning(history):
            return
        async with self._database.connection_transaction() as connection:
            await self._cleanup_expired(connection)
            statement = text(
                """
                insert into continuity.response_indexes (
                  scope_id,
                  source_response_id,
                  visible_item_hashes,
                  recorded_at,
                  expires_at
                )
                select
                  :scope_id,
                  :source_response_id,
                  :visible_item_hashes,
                  statement_timestamp(),
                  lease.expires_at
                  from responses.response_leases lease
                 where lease.scope_id = :scope_id
                   and lease.response_id = :source_response_id
                   and lease.owner_type = 'response'
                   and lease.owner_id = :source_response_id
                   and lease.status = 'live'
                   and lease.expires_at > statement_timestamp()
                on conflict (scope_id, source_response_id) do update
                  set visible_item_hashes = excluded.visible_item_hashes,
                      recorded_at = excluded.recorded_at,
                      expires_at = excluded.expires_at
                """
            ).bindparams(bindparam("visible_item_hashes", type_=ARRAY(LargeBinary())))
            result = await connection.execute(
                statement,
                {
                    "scope_id": scope_id,
                    "source_response_id": source_response_id,
                    "visible_item_hashes": list(history.hashes),
                },
            )
            recorded = result.rowcount > 0
            result.close()
        if not recorded:
            return
        logger.debug(
            "continuity.recorded",
            source_response_id=source_response_id,
            visible_item_count=len(history.hashes),
        )


__all__ = ["ContinuityStore"]
