"""LLM providers used by the agent loop.

Providers are called over plain HTTPS (``httpx``) instead of vendor SDKs: the
wire formats are small and stable, the serverless bundle stays light, and the
transport can be replaced in tests.

Conversation history uses the OpenAI chat format internally::

    {"role": "system" | "user", "content": "..."}
    {"role": "assistant", "content": "..." | None, "tool_calls": [...]}
    {"role": "tool", "tool_call_id": "...", "content": "..."}

Keys starting with an underscore are private to a provider adapter and are
never sent to a different one.

Hosted model IDs come and go (Groq shut down ``llama-3.3-70b-versatile``, which
this project used to hardcode, on 2026-08-16). When a provider reports that the
configured model is unavailable, the client asks the provider which models
exist and switches to a known-good one instead of failing - or, worse,
pretending the run succeeded.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import httpx

from ..config import (
    ANTHROPIC_MODEL_PREFERENCES,
    GROQ_MODEL_PREFERENCES,
    OPENAI_MODEL_PREFERENCES,
    Settings,
    env_str,
)

Notice = Callable[[str], None]

_MODEL_GONE_CODES = {"model_not_found", "model_decommissioned", "model_deprecated", "model_not_active"}
_MODEL_GONE_RE = re.compile(
    r"decommission|deprecat|does not exist|do not have access|not found|no longer (?:available|supported)",
    re.IGNORECASE,
)
_NON_CHAT_RE = re.compile(
    r"whisper|tts|orpheus|guard|embed|moderation|image|audio|realtime|transcri|speech|rerank|sora|dall",
    re.IGNORECASE,
)
_RETRY_IN_RE = re.compile(r"try again in\s+(?:(\d+)m)?\s*([0-9.]+)\s*(ms|s)", re.IGNORECASE)
# Groq validates every generated tool call and answers HTTP 400 (code
# ``tool_use_failed``) when the model's reply is not a usable call.
_TOOL_FORMAT_RE = re.compile(
    r"tool call validation failed|failed to call a function|failed to parse tool call|invalid tool call",
    re.IGNORECASE,
)
_REJECTED_TOOL_RE = re.compile(r"attempted to call tool '([^']{1,80})'", re.IGNORECASE)
_REPEATED_PREFIX_RE = re.compile(r"^(?P<prefix>[^:]{4,60}):\s*(?P=prefix):\s*", re.IGNORECASE)

# Sampling temperature for tool calling. Providers default to 1.0, which makes
# malformed tool calls and invented content more likely; Groq's tool-use guide
# recommends starting low and changing it when a generated call is refused.
TOOL_FORMAT_TEMPERATURE_STEP = 0.2
MAX_TEMPERATURE = 1.0


class LLMError(Exception):
    """A provider call failed. ``kind`` tells the agent whether to retry.

    kinds: ``rate_limit``, ``too_large``, ``auth``, ``model_unavailable``,
    ``tool_format``, ``server``, ``network``, ``bad_request``.

    ``tool_format`` means the *model's reply* was unusable (the provider refused
    a tool call that names a tool it was never offered, or whose arguments do
    not fit the schema). Nothing was executed. ``rejected_tool`` is the name the
    model made up, if that was the problem, and ``failed_generation`` is what the
    model wrote, when the provider reports it.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str = "bad_request",
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
        rejected_tool: Optional[str] = None,
        failed_generation: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status_code = status_code
        self.retry_after = retry_after
        self.rejected_tool = rejected_tool
        self.failed_generation = failed_generation


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    error: Optional[str] = None  # set when the model produced unparseable arguments


@dataclass
class LLMResponse:
    content: str
    tool_calls: List[ToolCall]
    finish_reason: str
    model: str
    assistant_message: Dict[str, Any]
    tokens_per_minute_limit: Optional[int] = None
    usage: Dict[str, Any] = field(default_factory=dict)


class ChatClient(Protocol):
    provider: str
    model: str

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        max_output_tokens: int,
        timeout: float,
    ) -> LLMResponse: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _parse_arguments(raw: Any) -> tuple[Dict[str, Any], Optional[str]]:
    if isinstance(raw, dict):
        return raw, None
    if raw is None or raw == "":
        return {}, None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return {}, f"arguments were not valid JSON ({exc})"
    if not isinstance(parsed, dict):
        return {}, "arguments must be a JSON object"
    return parsed, None


def _int_header(headers: httpx.Headers, *names: str) -> Optional[int]:
    for name in names:
        value = headers.get(name)
        if value:
            try:
                return int(float(value))
            except ValueError:
                continue
    return None


def _retry_after(headers: httpx.Headers, message: str) -> Optional[float]:
    value = headers.get("retry-after")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
    match = _RETRY_IN_RE.search(message or "")
    if match:
        minutes = float(match.group(1) or 0)
        amount = float(match.group(2))
        seconds = amount / 1000.0 if match.group(3).lower() == "ms" else amount
        return minutes * 60.0 + seconds
    return None


def _choose_model(available: Sequence[str], preferences: Sequence[str], current: str) -> Optional[str]:
    present = set(available)
    for candidate in preferences:
        if candidate != current and candidate in present:
            return candidate
    return None


class _HttpProvider:
    provider = "llm"
    model_env_var = ""

    def __init__(
        self,
        *,
        base_url: str,
        headers: Dict[str, str],
        transport: Optional[httpx.BaseTransport] = None,
        on_notice: Optional[Notice] = None,
    ) -> None:
        self._http = httpx.Client(base_url=base_url, headers=headers, transport=transport)
        self._notice = on_notice or (lambda _message: None)
        self._fallback_attempted = False

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, payload: Dict[str, Any], timeout: float) -> httpx.Response:
        try:
            return self._http.post(path, json=payload, timeout=max(5.0, timeout))
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"{self.provider} did not answer within {int(timeout)} seconds.", kind="network"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Could not reach {self.provider} ({exc.__class__.__name__}: {exc}).", kind="network"
            ) from exc

    def _list_model_ids(self, path: str) -> List[str]:
        try:
            response = self._http.get(path, timeout=20.0)
        except httpx.HTTPError:
            return []
        if response.status_code != 200:
            return []
        try:
            data = response.json().get("data") or []
        except (ValueError, AttributeError):
            return []
        return [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]

    def _switch_model(self, available: List[str], preferences: Sequence[str], reason: str) -> None:
        """Replace an unavailable model with a preferred one, or explain what to set."""
        current = getattr(self, "model")
        choice = _choose_model(available, preferences, current)
        if not choice:
            candidates = [name for name in available if not _NON_CHAT_RE.search(name)][:8]
            suggestion = (
                f" Set {self.model_env_var} to one of: {', '.join(candidates)}."
                if candidates
                else f" Set {self.model_env_var} to a model your account can use."
            )
            raise LLMError(
                f"Model '{current}' is not available on {self.provider} ({reason}).{suggestion}",
                kind="model_unavailable",
            )
        self._notice(
            f"Model '{current}' is not available on {self.provider} ({reason}); "
            f"using '{choice}' instead. Set {self.model_env_var} to silence this."
        )
        setattr(self, "model", choice)


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat completions (Groq, OpenAI, and compatible gateways)
# --------------------------------------------------------------------------- #
class OpenAICompatibleClient(_HttpProvider):
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        model_env_var: str,
        model_preferences: Sequence[str] = (),
        optional_params: Optional[Dict[str, Any]] = None,
        transport: Optional[httpx.BaseTransport] = None,
        on_notice: Optional[Notice] = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            transport=transport,
            on_notice=on_notice,
        )
        self.provider = provider
        self.model = model
        self.model_env_var = model_env_var
        self._preferences = tuple(model_preferences)
        # Parameters that only some models accept (e.g. reasoning_effort). If the
        # API rejects one by name it is dropped and the request is repeated.
        self._optional = dict(optional_params or {})
        # Replies in a row that the provider refused as invalid tool calls. Each
        # one raises the sampling temperature of the next request a little, so a
        # retry is not a re-run of the request that just failed.
        self._refused_replies = 0

    @staticmethod
    def _wire_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{k: v for k, v in message.items() if not k.startswith("_")} for message in messages]

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        max_output_tokens: int,
        timeout: float,
    ) -> LLMResponse:
        while True:
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": self._wire_messages(messages),
                "max_completion_tokens": int(max_output_tokens),
                **self._optional,
            }
            temperature = self._optional.get("temperature")
            if isinstance(temperature, (int, float)) and self._refused_replies:
                payload["temperature"] = round(
                    min(
                        MAX_TEMPERATURE,
                        float(temperature) + TOOL_FORMAT_TEMPERATURE_STEP * self._refused_replies,
                    ),
                    2,
                )
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            response = self._post("/chat/completions", payload, timeout)
            if response.status_code == 200:
                parsed = self._parse(response)
                self._refused_replies = 0
                return parsed

            error = self._error(response)
            if error.kind == "tool_format":
                self._refused_replies += 1
                raise error
            if response.status_code in (400, 422):
                # An optional parameter this model does not accept: drop it and retry.
                rejected = [name for name in self._optional if name in error.message]
                if rejected:
                    for name in rejected:
                        self._optional.pop(name, None)
                    continue
            if error.kind == "model_unavailable" and not self._fallback_attempted:
                self._fallback_attempted = True
                self._switch_model(self._list_model_ids("/models"), self._preferences, error.message)
                continue
            raise error

    def _error(self, response: httpx.Response) -> LLMError:
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = {}
        detail = body.get("error") if isinstance(body, dict) else None
        failed_generation: Optional[str] = None
        if isinstance(detail, dict):
            message = str(detail.get("message") or "")
            code = str(detail.get("code") or "")
            generated = detail.get("failed_generation")
            if generated:
                failed_generation = generated if isinstance(generated, str) else json.dumps(generated)
        else:
            message = str(detail or (response.text or "")[:300])
            code = ""
        text = f"{self.provider} returned HTTP {status}: {message or 'no details'}"
        lowered = message.lower()
        refused_reply = code == "tool_use_failed" or (
            status in (400, 422)
            and (failed_generation is not None or _TOOL_FORMAT_RE.search(message) is not None)
        )
        # The key works but this model cannot be used with it: retired, never
        # existed, blocked for the project, or its terms were not accepted
        # (codes such as model_not_found, model_decommissioned,
        # model_permission_blocked_project, model_terms_required).
        model_problem = status < 500 and (
            code in _MODEL_GONE_CODES
            or code.startswith("model_")
            or ("model" in lowered and _MODEL_GONE_RE.search(message) is not None)
        )

        if status == 401 or (status == 403 and not model_problem):
            kind = "auth"
            text += f" Check the {self.provider.upper()} API key."
        elif status == 413 or code == "context_length_exceeded" or "request too large" in lowered:
            kind = "too_large"
        elif status == 429:
            kind = "rate_limit"
        elif refused_reply:
            # The request was fine; the model's *reply* was not a usable tool call.
            match = _REJECTED_TOOL_RE.search(message)
            rejected_tool = match.group(1) if match else None
            if rejected_tool:
                reason = f"it tried to call a tool named '{rejected_tool}', which does not exist"
            else:
                reason = _REPEATED_PREFIX_RE.sub(r"\g<prefix>: ", message).strip().rstrip(".") or "no details"
            return LLMError(
                f"{self.provider} refused the model's reply because it was not a valid tool call ({reason}).",
                kind="tool_format",
                status_code=status,
                rejected_tool=rejected_tool,
                failed_generation=failed_generation,
            )
        elif status == 404 or model_problem:
            kind = "model_unavailable"
        elif status >= 500:
            kind = "server"
        else:
            kind = "bad_request"
        return LLMError(
            text, kind=kind, status_code=status, retry_after=_retry_after(response.headers, message)
        )

    def _parse(self, response: httpx.Response) -> LLMResponse:
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.provider} returned an unreadable response.", kind="server") from exc

        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        calls: List[ToolCall] = []
        wire_calls: List[Dict[str, Any]] = []
        for position, raw in enumerate(message.get("tool_calls") or []):
            function = (raw or {}).get("function") or {}
            name = str(function.get("name") or "")
            raw_arguments = function.get("arguments")
            arguments, problem = _parse_arguments(raw_arguments)
            call_id = str(raw.get("id") or f"call_{position}")
            calls.append(ToolCall(id=call_id, name=name, arguments=arguments, error=problem))
            wire_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": raw_arguments
                        if isinstance(raw_arguments, str)
                        else json.dumps(raw_arguments or {}),
                    },
                }
            )

        # Only fields every OpenAI-compatible API accepts back are kept; provider
        # extras such as `reasoning` are dropped from the history.
        assistant: Dict[str, Any] = {"role": "assistant", "content": content}
        if wire_calls:
            assistant["content"] = content or None
            assistant["tool_calls"] = wire_calls
        return LLMResponse(
            content=content,
            tool_calls=calls,
            finish_reason=str(choice.get("finish_reason") or ""),
            model=str(body.get("model") or self.model),
            assistant_message=assistant,
            tokens_per_minute_limit=_int_header(response.headers, "x-ratelimit-limit-tokens"),
            usage=body.get("usage") or {},
        )


# --------------------------------------------------------------------------- #
# Anthropic Messages API
# --------------------------------------------------------------------------- #
class AnthropicClient(_HttpProvider):
    provider = "anthropic"
    model_env_var = "ANTHROPIC_MODEL"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        model_preferences: Sequence[str] = (),
        transport: Optional[httpx.BaseTransport] = None,
        on_notice: Optional[Notice] = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            transport=transport,
            on_notice=on_notice,
        )
        self.model = model
        self._preferences = tuple(model_preferences)

    @staticmethod
    def _convert(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        system_parts: List[str] = []
        converted: List[Dict[str, Any]] = []

        def push(role: str, blocks: List[Dict[str, Any]]) -> None:
            if converted and converted[-1]["role"] == role:
                converted[-1]["content"].extend(blocks)
            else:
                converted.append({"role": role, "content": list(blocks)})

        for message in messages:
            role = message.get("role")
            if role == "system":
                system_parts.append(str(message.get("content") or ""))
            elif role == "user":
                push("user", [{"type": "text", "text": str(message.get("content") or "")}])
            elif role == "tool":
                push(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id"),
                            "content": str(message.get("content") or ""),
                        }
                    ],
                )
            elif role == "assistant":
                # Replay the provider's own blocks when we have them: thinking
                # blocks must be returned unmodified alongside tool_use blocks.
                blocks = message.get("_anthropic_content")
                if not blocks:
                    blocks = []
                    if message.get("content"):
                        blocks.append({"type": "text", "text": str(message["content"])})
                    for call in message.get("tool_calls") or []:
                        arguments, _problem = _parse_arguments(call["function"].get("arguments"))
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": call["id"],
                                "name": call["function"]["name"],
                                "input": arguments,
                            }
                        )
                if blocks:
                    push("assistant", blocks)
        return "\n\n".join(part for part in system_parts if part), converted

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        max_output_tokens: int,
        timeout: float,
    ) -> LLMResponse:
        system, converted = self._convert(messages)
        while True:
            payload: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": int(max_output_tokens),
                "messages": converted,
            }
            if system:
                payload["system"] = system
            if tools:
                payload["tools"] = [
                    {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "input_schema": tool["function"]["parameters"],
                    }
                    for tool in tools
                ]
            response = self._post("/v1/messages", payload, timeout)
            if response.status_code == 200:
                return self._parse(response)
            error = self._error(response)
            if error.kind == "model_unavailable" and not self._fallback_attempted:
                self._fallback_attempted = True
                self._switch_model(
                    self._list_model_ids("/v1/models?limit=100"), self._preferences, error.message
                )
                continue
            raise error

    def _error(self, response: httpx.Response) -> LLMError:
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = {}
        detail = body.get("error") if isinstance(body, dict) else None
        if isinstance(detail, dict):
            message = str(detail.get("message") or "")
            error_type = str(detail.get("type") or "")
        else:
            message = (response.text or "")[:300]
            error_type = ""
        text = f"anthropic returned HTTP {status}: {message or 'no details'}"
        lowered = message.lower()

        if status in (401, 403) or error_type in ("authentication_error", "permission_error"):
            kind = "auth"
            text += " Check ANTHROPIC_API_KEY."
        elif status == 429 or error_type == "rate_limit_error":
            kind = "rate_limit"
        elif status == 413 or error_type == "request_too_large" or "prompt is too long" in lowered:
            kind = "too_large"
        elif status == 404 or (error_type == "not_found_error" and "model" in lowered):
            kind = "model_unavailable"
        elif status >= 500 or error_type in ("overloaded_error", "api_error"):
            kind = "server"
        else:
            kind = "bad_request"
        return LLMError(
            text, kind=kind, status_code=status, retry_after=_retry_after(response.headers, message)
        )

    def _parse(self, response: httpx.Response) -> LLMResponse:
        try:
            body = response.json()
            blocks = body["content"]
            if not isinstance(blocks, list):
                raise TypeError("content is not a list")
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError("anthropic returned an unreadable response.", kind="server") from exc

        texts: List[str] = []
        calls: List[ToolCall] = []
        wire_calls: List[Dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                texts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                arguments = block.get("input") if isinstance(block.get("input"), dict) else {}
                call_id = str(block.get("id") or f"call_{len(calls)}")
                name = str(block.get("name") or "")
                calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
                wire_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                )
        content = "\n".join(text for text in texts if text)
        assistant: Dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
            "_anthropic_content": blocks,
        }
        if wire_calls:
            assistant["tool_calls"] = wire_calls
        stop_reason = str(body.get("stop_reason") or "")
        return LLMResponse(
            content=content,
            tool_calls=calls,
            finish_reason="length" if stop_reason == "max_tokens" else stop_reason,
            model=str(body.get("model") or self.model),
            assistant_message=assistant,
            tokens_per_minute_limit=_int_header(
                response.headers,
                "anthropic-ratelimit-input-tokens-limit",
                "anthropic-ratelimit-tokens-limit",
            ),
            usage=body.get("usage") or {},
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
DEFAULT_GROQ_TEMPERATURE = 0.2


def _temperature_setting(raw: Optional[str], default: Optional[float]) -> Optional[float]:
    """A sampling temperature from the environment: a number between 0 and 2,
    ``default`` to leave it to the provider, or unset for our own default."""
    if raw is None:
        return default
    if raw.strip().lower() == "default":
        return None
    try:
        value = float(raw)
    except ValueError:
        return default
    if value != value:  # NaN
        return default
    return max(0.0, min(2.0, value))


def create_llm_client(
    settings: Settings,
    *,
    transport: Optional[httpx.BaseTransport] = None,
    on_notice: Optional[Notice] = None,
) -> ChatClient:
    provider = settings.llm_provider
    if provider == "groq":
        optional: Dict[str, Any] = {}
        # Reasoning tokens count against Groq's per-minute token limits (8k on
        # the free tier), so the default effort is low. "default" omits it.
        effort = (env_str("GROQ_REASONING_EFFORT", "low") or "low").lower()
        if effort != "default":
            optional["reasoning_effort"] = effort
        # Groq samples at temperature 1.0 unless told otherwise. A low value makes
        # the open models it hosts far less likely to produce a malformed tool
        # call or to make things up. "default" omits the parameter.
        temperature = _temperature_setting(env_str("GROQ_TEMPERATURE"), DEFAULT_GROQ_TEMPERATURE)
        if temperature is not None:
            optional["temperature"] = temperature
        return OpenAICompatibleClient(
            provider="groq",
            base_url=settings.groq_base_url,
            api_key=settings.groq_api_key or "",
            model=settings.groq_model,
            model_env_var="GROQ_MODEL",
            model_preferences=GROQ_MODEL_PREFERENCES,
            optional_params=optional,
            transport=transport,
            on_notice=on_notice,
        )
    if provider == "openai":
        optional = {}
        # OpenAI's current chat-completions models only accept function tools
        # with reasoning disabled. Older models reject the parameter, in which
        # case the client drops it automatically.
        effort = (env_str("OPENAI_REASONING_EFFORT", "none") or "none").lower()
        if effort != "default":
            optional["reasoning_effort"] = effort
        return OpenAICompatibleClient(
            provider="openai",
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key or "",
            model=settings.openai_model,
            model_env_var="OPENAI_MODEL",
            model_preferences=OPENAI_MODEL_PREFERENCES,
            optional_params=optional,
            transport=transport,
            on_notice=on_notice,
        )
    if provider == "anthropic":
        return AnthropicClient(
            base_url=settings.anthropic_base_url,
            api_key=settings.anthropic_api_key or "",
            model=settings.anthropic_model,
            model_preferences=ANTHROPIC_MODEL_PREFERENCES,
            transport=transport,
            on_notice=on_notice,
        )
    raise LLMError(
        "No LLM provider is configured. Set GROQ_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY.",
        kind="auth",
    )
