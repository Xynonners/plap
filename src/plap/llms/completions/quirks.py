from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Any

from plap.llms.completions.chat import (
    ChatResponseFormatType,
    ChatToolChoiceFunction,
    ChatToolChoiceMode,
)
from plap.llms.completions.client import Call, Quirk
from plap.llms.completions.errors import ChatCompletionRateLimitError, ChatCompletionUnsupportedRequestError


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
            continue
        merged[key] = value
    return merged


def _first_choice(raw: dict[str, Any]) -> dict[str, Any] | None:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    return choice if isinstance(choice, dict) else None


def _response_message(raw: dict[str, Any]) -> dict[str, Any] | None:
    choice = _first_choice(raw)
    if choice is None:
        return None
    message = choice.get("message")
    return message if isinstance(message, dict) else None


def _response_delta(raw: dict[str, Any]) -> dict[str, Any] | None:
    choice = _first_choice(raw)
    if choice is None:
        return None
    delta = choice.get("delta")
    return delta if isinstance(delta, dict) else None


def _single_terminal_choice(raw: dict[str, Any]) -> dict[str, Any] | None:
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") is None:
        return None
    return choice


def _terminal_identity(raw: dict[str, Any], choice: dict[str, Any]) -> tuple[object, ...] | None:
    response_id = raw.get("id")
    model = raw.get("model")
    created = raw.get("created")
    index = choice.get("index")
    if (
        not isinstance(response_id, str)
        or not response_id
        or not isinstance(model, str)
        or not model
        or created is None
        or not isinstance(index, int)
    ):
        return None
    return (
        response_id,
        model,
        created,
        raw.get("object"),
        raw.get("provider"),
        index,
        choice.get("finish_reason"),
        choice.get("native_finish_reason"),
    )


def _empty_stream_value(value: object) -> bool:
    return value in (None, "", [], {})


def _empty_terminal_delta(choice: dict[str, Any]) -> bool:
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return False
    for name, value in delta.items():
        if name == "role":
            if value not in (None, "assistant"):
                return False
            continue
        if not _empty_stream_value(value):
            return False
    for name, value in choice.items():
        if name in {"delta", "finish_reason", "index", "native_finish_reason"}:
            continue
        if not _empty_stream_value(value):
            return False
    return True


def _nested_mapping_get(value: dict[str, Any] | None, path: tuple[str, ...]) -> Any:
    current: Any = value
    for segment in path:
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def _nested_mapping_pop(target: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current = target
    for segment in path[:-1]:
        child = current.get(segment)
        if not isinstance(child, dict):
            return False, None
        current = child
    leaf = path[-1]
    if leaf not in current:
        return False, None
    return True, current.pop(leaf)


def _nested_mapping_set(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = target
    for segment in path[:-1]:
        child = current.get(segment)
        if not isinstance(child, dict):
            child = {}
            current[segment] = child
        current = child
    current[path[-1]] = value


def _path(value: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not value:
        raise ValueError("path must contain at least one segment")
    return value


def _move_message_field(raw: dict[str, Any], old: str, new: str) -> None:
    message = _response_message(raw)
    if message is None or old not in message:
        return
    if message.get(new) is not None:
        return
    message[new] = message[old]


def _move_delta_field(raw: dict[str, Any], old: str, new: str) -> None:
    delta = _response_delta(raw)
    if delta is None or old not in delta:
        return
    if delta.get(new) is not None:
        return
    delta[new] = delta[old]


def _body_messages(call: Call) -> tuple[dict[str, Any], ...]:
    messages = call.body.get("messages")
    if not isinstance(messages, list):
        return ()
    return tuple(message for message in messages if isinstance(message, dict))


def _message_targets(
    call: Call,
    *,
    role: str | None,
    content_type: str | None,
) -> tuple[dict[str, Any], ...]:
    targets: list[dict[str, Any]] = []
    for message in _body_messages(call):
        if role is not None and message.get("role") != role:
            continue
        if content_type is None:
            targets.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        targets.extend(part for part in content if isinstance(part, dict) and part.get("type") == content_type)
    return tuple(targets)


def _body_tool_functions(call: Call) -> tuple[dict[str, Any], ...]:
    tools = call.body.get("tools")
    if not isinstance(tools, list):
        return ()
    functions: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            functions.append(function)
    return tuple(functions)


class Only(Quirk):
    def __init__(self, *names: str) -> None:
        self._names = frozenset(names)

    def request(self, call: Call) -> None:
        call.body = {key: value for key, value in call.body.items() if key in self._names}


class Move(Quirk):
    def __init__(self, source: str, *target_path: str) -> None:
        if not target_path:
            raise ValueError("Move requires at least one target path segment")
        self._source = source
        self._target_path = tuple(target_path)

    def request(self, call: Call) -> None:
        if self._source not in call.body:
            return
        value = call.body.pop(self._source)
        _nested_mapping_set(call.body, self._target_path, value)


class Drop(Quirk):
    def __init__(self, name: str) -> None:
        self._name = name

    def request(self, call: Call) -> None:
        call.body.pop(self._name, None)


class Set(Quirk):
    def __init__(self, name: str, value: Any) -> None:
        self._name = name
        self._value = value

    def request(self, call: Call) -> None:
        call.body[self._name] = self._value


class SystemRole(Quirk):
    def request(self, call: Call) -> None:
        for message in _body_messages(call):
            if message.get("role") == "developer":
                message["role"] = "system"


class DropMessageField(Quirk):
    def __init__(self, name: str) -> None:
        self._name = name

    def request(self, call: Call) -> None:
        for message in _body_messages(call):
            message.pop(self._name, None)


class MoveMessageField(Quirk):
    def __init__(
        self,
        source: str | tuple[str, ...],
        target: str | tuple[str, ...],
        *,
        role: str | None = None,
        content_type: str | None = None,
    ) -> None:
        self._source = _path(source)
        self._target = _path(target)
        self._role = role
        self._content_type = content_type

    def _move(self, target: dict[str, Any]) -> None:
        found, value = _nested_mapping_pop(target, self._source)
        if not found:
            return
        if _nested_mapping_get(target, self._target) is None:
            _nested_mapping_set(target, self._target, value)

    def request(self, call: Call) -> None:
        for target in _message_targets(call, role=self._role, content_type=self._content_type):
            self._move(target)


class RejectMessageField(Quirk):
    def __init__(
        self,
        path: str | tuple[str, ...],
        *,
        role: str | None = None,
        content_type: str | None = None,
    ) -> None:
        self._path = _path(path)
        self._role = role
        self._content_type = content_type

    def request(self, call: Call) -> None:
        for target in _message_targets(call, role=self._role, content_type=self._content_type):
            if _nested_mapping_get(target, self._path) is None:
                continue
            label = ".".join(self._path)
            raise ChatCompletionUnsupportedRequestError(f"message field {label!r} is not supported for model {call.request.model!r}")


class DropToolFunctionField(Quirk):
    def __init__(self, name: str) -> None:
        self._name = name

    def request(self, call: Call) -> None:
        for function in _body_tool_functions(call):
            function.pop(self._name, None)


class ExtraBody(Quirk):
    def __init__(self, value: dict[str, Any]) -> None:
        self._value = value

    def request(self, call: Call) -> None:
        current = call.body.get("extra_body")
        if isinstance(current, dict):
            call.body["extra_body"] = _merge_dicts(current, self._value)
            return
        call.body["extra_body"] = dict(self._value)


class ExtraBodyIf(Quirk):
    def __init__(self, name: str, values: tuple[Any, ...], extra_body: dict[str, Any], *, negate: bool = False) -> None:
        self._name = name
        self._values = values
        self._extra_body = extra_body
        self._negate = negate

    def request(self, call: Call) -> None:
        matched = call.body.get(self._name) in self._values
        if self._negate:
            matched = not matched
        if not matched:
            return
        current = call.body.get("extra_body")
        if isinstance(current, dict):
            call.body["extra_body"] = _merge_dicts(current, self._extra_body)
            return
        call.body["extra_body"] = dict(self._extra_body)


class DropIf(Quirk):
    def __init__(self, name: str, value: Any) -> None:
        self._name = name
        self._value = value

    def request(self, call: Call) -> None:
        if call.body.get(self._name) == self._value:
            call.body.pop(self._name, None)


class RejectIf(Quirk):
    def __init__(self, name: str, value: Any) -> None:
        self._name = name
        self._value = value

    def request(self, call: Call) -> None:
        if call.body.get(self._name) != self._value:
            return
        raise ChatCompletionUnsupportedRequestError(
            f"field {self._name!r} value {self._value!r} is not supported for model {call.request.model!r}"
        )


class EnsureAssistantReasoningContent(Quirk):
    def request(self, call: Call) -> None:
        for message in _body_messages(call):
            if message.get("role") != "assistant":
                continue
            if message.get("reasoning_content") is None:
                message["reasoning_content"] = ""


class ForceRequiredToolChoice(Quirk):
    def request(self, call: Call) -> None:
        if not isinstance(call.request.tool_choice, ChatToolChoiceFunction):
            return
        if len(call.request.tools) == 1 and call.request.tools[0].function.name == call.request.tool_choice.name:
            call.body["tool_choice"] = "required"
            return
        raise ChatCompletionUnsupportedRequestError(
            "forced function tool_choice objects are unsupported here; use tool_choice='required' or provide exactly one matching tool"
        )


class ForceNamedToolChoice(Quirk):
    def request(self, call: Call) -> None:
        if call.request.tool_choice != ChatToolChoiceMode.REQUIRED:
            return
        if len(call.request.tools) == 1:
            call.body["tool_choice"] = {
                "type": "function",
                "function": {"name": call.request.tools[0].function.name},
            }
            return
        raise ChatCompletionUnsupportedRequestError(
            "tool_choice='required' is unsupported here; provide exactly one tool so it can be named explicitly"
        )


class DeduplicateTerminalUsage(Quirk):
    async def stream(self, call: Call, next_complete, next_stream):
        terminal_identity: tuple[object, ...] | None = None
        terminal_had_usage = False
        normalized_usage_terminal = False

        async for raw in next_stream(None):
            choice = _single_terminal_choice(raw)
            identity = None if choice is None else _terminal_identity(raw, choice)
            if identity is not None and terminal_identity is None:
                terminal_identity = identity
                terminal_had_usage = raw.get("usage") is not None
            elif (
                identity is not None
                and identity == terminal_identity
                and not terminal_had_usage
                and not normalized_usage_terminal
                and isinstance(raw.get("usage"), dict)
                and raw.get("error") is None
                and _empty_terminal_delta(choice)
            ):
                normalized_usage_terminal = True
                yield {**raw, "choices": []}
                continue
            yield raw


class MoveOutput(Quirk):
    def __init__(self, old: str, new: str) -> None:
        self._old = old
        self._new = new

    async def complete(self, call: Call, next_complete) -> dict[str, Any]:
        raw = await next_complete(None)
        _move_message_field(raw, self._old, self._new)
        return raw

    async def stream(self, call: Call, next_complete, next_stream):
        async for raw in next_stream(None):
            _move_delta_field(raw, self._old, self._new)
            yield raw


class PromoteOutput(Quirk):
    def __init__(self, name: str, *path: str) -> None:
        if not path:
            raise ValueError("PromoteOutput requires at least one path segment")
        self._name = name
        self._path = tuple(path)

    async def complete(self, call: Call, next_complete) -> dict[str, Any]:
        raw = await next_complete(None)
        value = _nested_mapping_get(_response_message(raw), self._path)
        if value is not None:
            raw[self._name] = value
        return raw

    async def stream(self, call: Call, next_complete, next_stream):
        async for raw in next_stream(None):
            value = _nested_mapping_get(_response_delta(raw), self._path)
            if value is not None:
                raw[self._name] = value
            yield raw


class RejectResponseFormat(Quirk):
    def __init__(self, *types: str) -> None:
        self._types = frozenset(ChatResponseFormatType(value) for value in types)

    def request(self, call: Call) -> None:
        response_format = call.request.response_format
        if response_format is None:
            return
        if self._types and response_format.type not in self._types:
            return
        if not self._types:
            raise ChatCompletionUnsupportedRequestError(f"response_format is not supported for model {call.request.model!r}")
        raise ChatCompletionUnsupportedRequestError(
            f"response_format is not supported for model {call.request.model!r} when type is {response_format.type.value!r}"
        )


class RateLimit(Quirk):
    def __init__(self, limit: int, window_seconds: float) -> None:
        if limit <= 0:
            raise ValueError("rate limit must be positive")
        if window_seconds <= 0:
            raise ValueError("rate limit window must be positive")
        self._limit = limit
        self._window_seconds = float(window_seconds)
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def __deepcopy__(self, memo: dict[int, Any]) -> RateLimit:
        copied = type(self)(self._limit, self._window_seconds)
        memo[id(self)] = copied
        return copied

    def _admit(self, *, model: str) -> None:
        now = time.monotonic()
        cutoff = now - self._window_seconds
        with self._lock:
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._limit:
                raise ChatCompletionRateLimitError(
                    f"local rate limit exceeded for model {model!r}: {self._limit} requests per {self._window_seconds:g} seconds"
                )
            self._timestamps.append(now)

    async def complete(self, call: Call, next_complete) -> dict[str, Any]:
        self._admit(model=call.request.model)
        return await next_complete(None)

    async def stream(self, call: Call, next_complete, next_stream):
        self._admit(model=call.request.model)
        async for raw in next_stream(None):
            yield raw


__all__ = [
    "DeduplicateTerminalUsage",
    "Drop",
    "DropIf",
    "DropMessageField",
    "DropToolFunctionField",
    "EnsureAssistantReasoningContent",
    "ExtraBody",
    "ExtraBodyIf",
    "ForceNamedToolChoice",
    "ForceRequiredToolChoice",
    "Move",
    "MoveMessageField",
    "MoveOutput",
    "Only",
    "PromoteOutput",
    "RateLimit",
    "RejectMessageField",
    "RejectResponseFormat",
    "Set",
    "SystemRole",
]
