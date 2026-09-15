from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import blake3
import msgspec

from plap.responses.contracts import (
    InputFileContent,
    InputImageContent,
    InputTextContent,
    OutputRefusalContent,
    OutputTextContent,
    RequestCompactionItem,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestInputItem,
    RequestItemReference,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCompactionItem,
    ResponseFunctionCallItem,
    ResponseFunctionCallOutputItem,
    ResponseMessageItem,
    ResponseOutputItem,
    ResponseReasoningItem,
)

type HistoryItem = RequestInputItem | ResponseOutputItem
type FunctionItem = RequestFunctionCallItem | RequestFunctionCallOutputItem | ResponseFunctionCallItem | ResponseFunctionCallOutputItem


@dataclass(frozen=True, slots=True)
class ReasoningPlacement:
    after_visible: int
    item: RequestReasoningItem


@dataclass(frozen=True, slots=True)
class PublicHistory:
    hashes: tuple[bytes, ...]
    reasoning: tuple[ReasoningPlacement, ...]
    user_indices: frozenset[int]


def _without_none(**values: object) -> dict[str, object]:
    return {key: value for key, value in values.items() if value is not None}


def _content_part(
    part: InputTextContent | InputImageContent | InputFileContent | OutputTextContent | OutputRefusalContent,
) -> dict[str, object]:
    if isinstance(part, InputTextContent | OutputTextContent):
        return {"text": part.text, "type": "text"}
    if isinstance(part, OutputRefusalContent):
        return {"refusal": part.refusal, "type": "refusal"}
    if isinstance(part, InputImageContent):
        return _without_none(
            detail=part.detail,
            file_id=part.file_id,
            image_url=part.image_url,
            type="image",
        )
    if isinstance(part, InputFileContent):
        return _without_none(
            detail=part.detail,
            file_data=part.file_data,
            file_id=part.file_id,
            file_url=part.file_url,
            filename=part.filename,
            type="file",
        )
    raise TypeError(f"unsupported continuity content part: {type(part).__name__}")


def _append_content_part(parts: list[dict[str, object]], part: dict[str, object]) -> None:
    if part["type"] == "text" and parts and parts[-1]["type"] == "text":
        parts[-1]["text"] = f"{parts[-1]['text']}{part['text']}"
        return
    parts.append(part)


def _canonical_content(
    content: str | list[InputTextContent | InputImageContent | InputFileContent | OutputTextContent | OutputRefusalContent],
) -> list[dict[str, object]]:
    if isinstance(content, str):
        return [{"text": content, "type": "text"}]
    normalized: list[dict[str, object]] = []
    for part in content:
        _append_content_part(normalized, _content_part(part))
    return normalized


def _canonical_arguments(arguments: str) -> dict[str, object]:
    try:
        decoded = msgspec.json.decode(arguments.encode())
    except msgspec.DecodeError:
        return {"raw": arguments}
    return {"json": decoded}


def _caller_type(item: FunctionItem) -> str | None:
    return None if item.caller is None else item.caller.type


def _canonical_item(item: HistoryItem) -> dict[str, object] | None:
    if isinstance(item, RequestMessageItem | ResponseMessageItem):
        content = _canonical_content(item.content)
        if item.role == "assistant" and content == [{"text": "", "type": "text"}]:
            return None
        return _without_none(
            content=content,
            phase=item.phase,
            role=item.role,
            type="message",
        )
    if isinstance(item, RequestFunctionCallItem | ResponseFunctionCallItem):
        return _without_none(
            arguments=_canonical_arguments(item.arguments),
            call_id=item.call_id,
            caller=_caller_type(item),
            name=item.name,
            namespace=item.namespace,
            type="function_call",
        )
    if isinstance(item, RequestFunctionCallOutputItem | ResponseFunctionCallOutputItem):
        return _without_none(
            call_id=item.call_id,
            caller=_caller_type(item),
            output=_canonical_content(item.output),
            type="function_call_output",
        )
    raise TypeError(f"unsupported continuity history item: {type(item).__name__}")


def _request_reasoning(item: RequestReasoningItem | ResponseReasoningItem) -> RequestReasoningItem:
    return RequestReasoningItem(
        content=None if item.content is None else list(item.content),
        encrypted_content=item.encrypted_content,
        id=item.id,
        status=item.status,
        summary=list(item.summary),
        type="reasoning",
    )


def _item_hash(item: dict[str, object]) -> bytes:
    encoded = msgspec.json.encode(item, order="deterministic")
    return blake3.blake3(encoded).digest()


def build_history(items: Sequence[HistoryItem]) -> PublicHistory | None:
    hashes: list[bytes] = []
    reasoning: list[ReasoningPlacement] = []
    user_indices: set[int] = set()
    for item in items:
        if isinstance(item, RequestItemReference | RequestCompactionItem | ResponseCompactionItem):
            return None
        if isinstance(item, RequestReasoningItem | ResponseReasoningItem):
            reasoning.append(ReasoningPlacement(after_visible=len(hashes), item=_request_reasoning(item)))
            continue
        canonical = _canonical_item(item)
        if canonical is None:
            continue
        if canonical.get("type") == "message" and canonical.get("role") == "user":
            user_indices.add(len(hashes))
        hashes.append(_item_hash(canonical))
    return PublicHistory(
        hashes=tuple(hashes),
        reasoning=tuple(reasoning),
        user_indices=frozenset(user_indices),
    )


def overlap_length(stored: Sequence[bytes], incoming: Sequence[bytes]) -> int:
    if not incoming:
        return 0

    failure = [0] * len(incoming)
    matched = 0
    for index in range(1, len(incoming)):
        while matched and incoming[index] != incoming[matched]:
            matched = failure[matched - 1]
        if incoming[index] == incoming[matched]:
            matched += 1
        failure[index] = matched

    matched = 0
    for item_hash in stored:
        while matched and (matched == len(incoming) or item_hash != incoming[matched]):
            matched = failure[matched - 1]
        if item_hash == incoming[matched]:
            matched += 1
    return matched


def restore_reasoning(
    incoming: tuple[RequestInputItem, ...],
    source: PublicHistory,
    overlap: int,
) -> tuple[RequestInputItem, ...] | None:
    if overlap <= 0 or overlap > len(source.hashes):
        return None
    start = len(source.hashes) - overlap
    if start != 0 and start not in source.user_indices:
        return None

    placements: dict[int, list[RequestReasoningItem]] = {}
    end = len(source.hashes)
    for placement in source.reasoning:
        if placement.item.encrypted_content is None:
            continue
        if placement.after_visible == 0 and start == 0:
            slot = 0
        elif start < placement.after_visible <= end:
            slot = placement.after_visible - start
        else:
            continue
        placements.setdefault(slot, []).append(placement.item)
    if not placements:
        return None

    restored: list[RequestInputItem] = list(placements.get(0, ()))
    matched_visible = 0
    for item in incoming:
        restored.append(item)
        if isinstance(item, RequestReasoningItem):
            continue
        canonical = _canonical_item(item)
        if canonical is None:
            continue
        if matched_visible < overlap:
            matched_visible += 1
            restored.extend(placements.get(matched_visible, ()))
    return tuple(restored)


def input_is_matchable(items: Sequence[RequestInputItem]) -> bool:
    return not any(isinstance(item, RequestItemReference | RequestCompactionItem) for item in items)


def contains_reasoning(items: Sequence[RequestInputItem]) -> bool:
    return any(isinstance(item, RequestReasoningItem) for item in items)


def has_sealed_reasoning(history: PublicHistory) -> bool:
    return any(placement.item.encrypted_content is not None for placement in history.reasoning)


__all__ = [
    "HistoryItem",
    "PublicHistory",
    "build_history",
    "contains_reasoning",
    "has_sealed_reasoning",
    "input_is_matchable",
    "overlap_length",
    "restore_reasoning",
]
