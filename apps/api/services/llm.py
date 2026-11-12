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

Several providers can be configured at once. ``FailoverClient`` puts them
behind one client: when the provider in use reaches its rate limit, rejects
the key, is down or keeps producing unusable replies, the conversation carries
on with the next one instead of waiting or failing.

Hosted model IDs come and go (Groq shut down ``llama-3.3-70b-versatile``, which
this project used to hardcode, on 2026-08-16). When a provider reports that the
configured model is unavailable, the client asks the provider which models
exist and switches to a known-good one instead of failing - or, worse,
pretending the run succeeded.
"""
from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import httpx

from ..config import (
    ANTHROPIC_MODEL_PREFERENCES,
    GEMINI_MODEL_PREFERENCES,
    GROQ_MODEL_PREFERENCES,
    LLM_PROVIDERS,
    OPENAI_MODEL_PREFERENCES,
    PROVIDER_KEY_VARIABLES,
    PROVIDER_LABELS,
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
_RETRY_IN_RE = re.compile(
    r"(?:try again|retry) in\s+(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([0-9.]+)\s*(ms|s))?", re.IGNORECASE
)
_PER_DAY_RE = re.compile(r"per[- _]?day|\(TPD\)|\(RPD\)|daily", re.IGNORECASE)
# Prefixes some models put in front of a tool's name ("functions/write_file",
# "functions.write_file"): the name of the namespace their training data used.
_CHANNEL_MARK = "<|"
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

    kinds: ``rate_limit``, ``quota``, ``too_large``, ``auth``,
    ``model_unavailable``, ``tool_format``, ``blocked``, ``server``,
    ``network``, ``bad_request``.

    ``tool_format`` means the *model's reply* was unusable (the provider refused
    a tool call that names a tool it was never offered, or whose arguments do
    not fit the schema). Nothing was executed. ``rejected_tool`` is the name the
    model made up, if that was the problem, and ``failed_generation`` is what the
    model wrote, when the provider reports it.

    ``quota`` means the account cannot be used until something changes (no
    credits left); ``daily`` marks a rate limit that does not lift within a
    run. ``provider`` names who failed. When several providers are configured
    and another one has taken over, ``switched_to`` names it: the request can
    be repeated at once.
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
        daily: bool = False,
        provider: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status_code = status_code
        self.retry_after = retry_after
        self.rejected_tool = rejected_tool
        self.failed_generation = failed_generation
        self.daily = daily
        self.provider = provider
        self.switched_to: Optional[str] = None
        # After waiting ``retry_after`` the request goes to this provider.
        self.next_provider: Optional[str] = None
        # Several providers are configured and each of them is waiting for its limit.
        self.all_limited = False


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
    provider: str = ""  # who answered (set by the clients; several can serve one run)
    recovered: bool = False  # rebuilt from a reply the provider had refused


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
    for match in _RETRY_IN_RE.finditer(message or ""):
        hours, minutes, amount, unit = match.groups()
        if not (hours or minutes or amount):
            continue
        seconds = float(amount or 0)
        if (unit or "s").lower() == "ms":
            seconds /= 1000.0
        return float(hours or 0) * 3600.0 + float(minutes or 0) * 60.0 + seconds
    return None


def canonical_tool_name(name: Any, known: Sequence[str]) -> Optional[str]:
    """The offered tool a model meant by ``name``, or None.

    Open models sometimes address a tool the way their training data did:
    ``functions/write_file``, ``functions.write_file``, ``WRITE_FILE`` or with a
    leaked channel marker behind it. Those mean ``write_file``. A name that
    matches no offered tool even then (``commentary``, ``repo_browser.open``)
    stays unknown.
    """
    if isinstance(name, str) and name in known:
        return name
    cleaned = str(name or "").split(_CHANNEL_MARK, 1)[0].strip().strip("\"'`")
    if not cleaned:
        return None
    candidates = [cleaned, cleaned.replace("-", "_")]
    for separator in ("/", ".", ":"):
        if separator in cleaned:
            candidates.append(cleaned.rsplit(separator, 1)[1].strip())
    by_lower = {item.lower(): item for item in known}
    for candidate in candidates:
        if candidate in known:
            return candidate
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _tool_names(tools: Sequence[Dict[str, Any]]) -> List[str]:
    return [str((tool.get("function") or {}).get("name") or "") for tool in tools or []]


_FUNCTION_TAG_RE = re.compile(r"<function=([^>\s]+)>\s*(\{.*?\})\s*(?:</function>|$)", re.DOTALL)


def calls_in_generation(generated: str) -> List[Tuple[str, Any]]:
    """Tool calls found in the raw text a provider refused: ``(name, arguments)``.

    Understands what the hosted open models write: one JSON object with
    ``name`` and ``arguments``, a list of them, or ``<function=name>{...}``.
    Line breaks inside JSON strings - a common reason for "arguments are not
    valid JSON" - are accepted. Returns [] when the text is not a tool call.
    """
    text = (generated or "").strip()
    if not text:
        return []
    found: List[Tuple[str, Any]] = []
    try:
        parsed = json.loads(text, strict=False)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
        parsed = parsed["tool_calls"]
    items = parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, dict) else []
    for item in items:
        if not isinstance(item, dict):
            return []
        function = item.get("function") if isinstance(item.get("function"), dict) else item
        name = function.get("name")
        arguments = function.get("arguments", function.get("parameters"))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments, strict=False)
            except ValueError:
                return []
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return []
        found.append((name, arguments))
    if found:
        return found
    for name, raw in _FUNCTION_TAG_RE.findall(text):
        try:
            arguments = json.loads(raw, strict=False)
        except ValueError:
            return []
        if not isinstance(arguments, dict):
            return []
        found.append((name, arguments))
    return found


def clip_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _reset_delay(headers: httpx.Headers, metadata: Dict[str, Any]) -> Optional[float]:
    """Seconds until a rate limit resets, from ``X-RateLimit-Reset`` (a Unix
    time in seconds or milliseconds), sent as a header or inside the error."""
    candidates: List[Any] = [headers.get("x-ratelimit-reset")]
    inner = metadata.get("headers") if isinstance(metadata, dict) else None
    if isinstance(inner, dict):
        candidates += [value for key, value in inner.items() if str(key).lower() == "x-ratelimit-reset"]
    for value in candidates:
        try:
            moment = float(value)
        except (TypeError, ValueError):
            continue
        if moment > 1e11:  # milliseconds
            moment /= 1000.0
        if moment > 1e9:
            return max(0.0, moment - time.time())
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
        output_tokens_param: str = "max_completion_tokens",
        extra_headers: Optional[Dict[str, str]] = None,
        assumed_tokens_per_minute: Optional[int] = None,
        list_models: bool = True,
        transport: Optional[httpx.BaseTransport] = None,
        on_notice: Optional[Notice] = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **(extra_headers or {}),
            },
            transport=transport,
            on_notice=on_notice,
        )
        self.provider = provider
        self.model = model
        self.model_env_var = model_env_var
        self._preferences = tuple(model_preferences)
        self._output_tokens_param = output_tokens_param
        # For providers that do not report a tokens-per-minute limit: what the
        # agent may assume when it sizes its context.
        self.assumed_tokens_per_minute = assumed_tokens_per_minute
        self._list_models = list_models
        self._recovered_calls = 0
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
                self._output_tokens_param: int(max_output_tokens),
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
                parsed = self._parse(response, tools)
                self._refused_replies = 0
                return parsed

            error = self._error(response)
            if error.kind == "tool_format":
                recovered = self._recover_refused_reply(error, tools)
                if recovered is not None:
                    self._refused_replies = 0
                    return recovered
                self._refused_replies += 1
                raise error
            if response.status_code in (400, 422):
                # An optional parameter this model does not accept: drop it and retry.
                rejected = [name for name in self._optional if name in error.message]
                if rejected:
                    for name in rejected:
                        self._optional.pop(name, None)
                    continue
            if error.kind == "model_unavailable" and not self._fallback_attempted and self._list_models:
                self._fallback_attempted = True
                self._switch_model(self._list_model_ids("/models"), self._preferences, error.message)
                continue
            raise error

    def _recover_refused_reply(self, error: LLMError, tools: List[Dict[str, Any]]) -> Optional[LLMResponse]:
        """Turn a reply the provider refused into the tool calls the model meant.

        Groq checks every generated call and refuses it over things that do not
        change its meaning: ``functions/write_file`` instead of ``write_file``,
        or a line break inside a JSON string. The refused text comes back in
        ``failed_generation``. When it holds complete calls to tools that were
        offered, they are used - the alternative is to ask again and hope, which
        costs a round trip against a tight rate limit each time.
        Refusals about the *content* of a call (missing arguments, a tool that
        does not exist) are not recovered.
        """
        if not error.failed_generation:
            return None
        about_name = error.rejected_tool is not None
        about_json = "json" in error.message.lower() or "parse" in error.message.lower()
        if not (about_name or about_json):
            return None
        known = _tool_names(tools)
        resolved: List[Tuple[str, Dict[str, Any]]] = []
        for name, arguments in calls_in_generation(error.failed_generation):
            canonical = canonical_tool_name(name, known)
            if canonical is None:
                return None
            resolved.append((canonical, arguments))
        if not resolved:
            return None
        calls: List[ToolCall] = []
        wire_calls: List[Dict[str, Any]] = []
        for name, arguments in resolved:
            self._recovered_calls += 1
            call_id = f"call_recovered_{self._recovered_calls}"
            calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
            wire_calls.append(
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            )
        return LLMResponse(
            content="",
            tool_calls=calls,
            finish_reason="tool_calls",
            model=self.model,
            assistant_message={"role": "assistant", "content": None, "tool_calls": wire_calls},
            provider=self.provider,
            recovered=True,
        )

    def _error(self, response: httpx.Response, body: Any = None) -> LLMError:
        status = response.status_code
        if body is None:
            try:
                body = response.json()
            except ValueError:
                body = {}
        detail = body.get("error") if isinstance(body, dict) else None
        failed_generation: Optional[str] = None
        metadata: Dict[str, Any] = {}
        if isinstance(detail, dict):
            message = str(detail.get("message") or "")
            code = str(detail.get("code") or "")
            generated = detail.get("failed_generation")
            if generated:
                failed_generation = generated if isinstance(generated, str) else json.dumps(generated)
            if isinstance(detail.get("metadata"), dict):
                metadata = detail["metadata"]
                # Gateways (OpenRouter) put the upstream provider's own words here.
                raw = metadata.get("raw")
                if isinstance(raw, str) and raw.strip() and raw.strip() not in message:
                    message = f"{message} ({clip_text(raw.strip(), 300)})" if message else raw.strip()
            if status == 200 and code.isdigit():
                status = int(code)  # an error reported inside a 200 response
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

        retry_after = _retry_after(response.headers, message)
        if status == 401 or (status == 403 and not model_problem):
            kind = "auth"
            text += f" Check the {self.provider.upper()} API key."
        elif status == 402:
            # No credits (OpenRouter answers this for free models too when the
            # balance is negative). A short Retry-After means "settling", not "empty".
            kind = "rate_limit" if retry_after is not None and retry_after <= 120 else "quota"
        elif status == 413 or code == "context_length_exceeded" or "request too large" in lowered:
            kind = "too_large"
        elif status == 429:
            kind = "rate_limit"
            if retry_after is None:
                retry_after = _reset_delay(response.headers, metadata)
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
        elif status >= 500 or status == 408:
            kind = "server"
        else:
            kind = "bad_request"
        return LLMError(
            text,
            kind=kind,
            status_code=status,
            retry_after=retry_after,
            daily=kind == "rate_limit" and _PER_DAY_RE.search(message) is not None,
            provider=self.provider,
        )

    def _parse(self, response: httpx.Response, tools: Sequence[Dict[str, Any]] = ()) -> LLMResponse:
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"{self.provider} returned an unreadable response.", kind="server") from exc
        if isinstance(body, dict) and body.get("error") and not body.get("choices"):
            # Gateways answer 200 and report a failure of the model behind them in the body.
            raise self._error(response, body)
        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.provider} returned an unreadable response.", kind="server") from exc
        if isinstance(choice.get("error"), dict) and not (message.get("content") or message.get("tool_calls")):
            raise self._error(response, {"error": choice["error"]})

        known = _tool_names(tools)
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        calls: List[ToolCall] = []
        wire_calls: List[Dict[str, Any]] = []
        for position, raw in enumerate(message.get("tool_calls") or []):
            function = (raw or {}).get("function") or {}
            name = str(function.get("name") or "")
            # "functions/write_file" means write_file; an unknown name stays as it is.
            name = canonical_tool_name(name, known) or name
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
            provider=self.provider,
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
            text,
            kind=kind,
            status_code=status,
            retry_after=_retry_after(response.headers, message),
            provider=self.provider,
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
            provider=self.provider,
        )


# --------------------------------------------------------------------------- #
# Google Gemini (generateContent)
# --------------------------------------------------------------------------- #
# Gemini 3 and later sign the reasoning behind every function call and refuse a
# history in which a call of the current turn has no signature. Calls made by a
# different provider earlier in the same conversation have none; Google
# documents this value for exactly that case ("transferring a trace from a
# different model"): it skips the check for the part that carries it.
GEMINI_FOREIGN_CALL_SIGNATURE = "context_engineering_is_the_way_to_go"
_GEMINI_SCHEMA_KEYS = ("type", "description", "enum", "format", "nullable", "minimum", "maximum", "minItems", "maxItems")
_GEMINI_BLOCKED_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE", "IMAGE_SAFETY"}


class GeminiClient(_HttpProvider):
    """Google's Gemini API, spoken natively.

    The request is the documented ``models/{model}:generateContent`` call with
    the key in ``x-goog-api-key`` - the form Google AI Studio shows next to a
    newly created key, and the one its authorization keys ("AQ....") are
    issued for.
    """

    provider = "gemini"
    model_env_var = "GEMINI_MODEL"
    # Gemini does not report its limits in response headers. Even the free tier
    # allows far more tokens per minute than the agent's context ever holds.
    assumed_tokens_per_minute = 250_000

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        model_preferences: Sequence[str] = (),
        thinking_level: Optional[str] = "low",
        transport: Optional[httpx.BaseTransport] = None,
        on_notice: Optional[Notice] = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            transport=transport,
            on_notice=on_notice,
        )
        self.model = model[len("models/") :] if model.startswith("models/") else model
        self._preferences = tuple(model_preferences)
        # How much the model reasons before it answers. Gemini 3 defaults to
        # "high", which costs seconds per turn that a task with a fixed time
        # budget does not have. None leaves it to the model.
        self._thinking_level = thinking_level
        self._calls = 0

    # ------------------------------------------------------------------ #
    # Request
    # ------------------------------------------------------------------ #
    @classmethod
    def _schema(cls, schema: Any) -> Dict[str, Any]:
        """The subset of JSON Schema that function declarations accept."""
        if not isinstance(schema, dict):
            return {"type": "string"}
        converted: Dict[str, Any] = {key: schema[key] for key in _GEMINI_SCHEMA_KEYS if key in schema}
        kind = converted.get("type")
        if isinstance(kind, list):  # ["string", "null"]
            named = [item for item in kind if item != "null"]
            converted["type"] = named[0] if named else "string"
            if "null" in kind:
                converted["nullable"] = True
        if isinstance(schema.get("properties"), dict) and schema["properties"]:
            converted["properties"] = {
                str(name): cls._schema(value) for name, value in schema["properties"].items()
            }
        if isinstance(schema.get("required"), list) and schema["required"]:
            converted["required"] = [str(name) for name in schema["required"]]
        if "items" in schema:
            converted["items"] = cls._schema(schema["items"])
        return converted

    @classmethod
    def _declarations(cls, tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        declarations: List[Dict[str, Any]] = []
        for tool in tools:
            function = tool.get("function") or {}
            declaration: Dict[str, Any] = {
                "name": function.get("name"),
                "description": function.get("description", ""),
            }
            parameters = cls._schema(function.get("parameters") or {})
            if parameters.get("properties"):  # an object without properties is refused
                declaration["parameters"] = parameters
            declarations.append(declaration)
        return declarations

    @staticmethod
    def _convert(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """The conversation as ``systemInstruction`` text and ``contents``."""
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []
        names: Dict[str, str] = {}  # tool call id -> function name
        echo_id: Dict[str, bool] = {}  # whether the call's id is known to Gemini

        def push(role: str, parts: List[Dict[str, Any]]) -> None:
            if not parts:
                return
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": list(parts)})

        for message in messages:
            role = message.get("role")
            if role == "system":
                text = str(message.get("content") or "")
                if text:
                    system_parts.append(text)
            elif role == "user":
                text = str(message.get("content") or "")
                if text:
                    push("user", [{"text": text}])
            elif role == "assistant":
                calls = message.get("tool_calls") or []
                for call in calls:
                    names[str(call.get("id"))] = str((call.get("function") or {}).get("name") or "")
                native = message.get("_gemini_parts")
                if native:
                    # Gemini's own parts, unchanged: every signature stays in
                    # the part it arrived in.
                    native_ids = {
                        part["functionCall"].get("id")
                        for part in native
                        if isinstance(part, dict) and isinstance(part.get("functionCall"), dict)
                    }
                    for call in calls:
                        echo_id[str(call.get("id"))] = call.get("id") in native_ids
                    push("model", copy.deepcopy(native))
                    continue
                parts: List[Dict[str, Any]] = []
                if message.get("content"):
                    parts.append({"text": str(message["content"])})
                for position, call in enumerate(calls):
                    function = call.get("function") or {}
                    arguments, _problem = _parse_arguments(function.get("arguments"))
                    part: Dict[str, Any] = {
                        "functionCall": {
                            "name": function.get("name"),
                            "args": arguments,
                            "id": str(call.get("id")),
                        }
                    }
                    if position == 0:
                        # Written by another provider (or shortened since): no signature exists.
                        part["thoughtSignature"] = GEMINI_FOREIGN_CALL_SIGNATURE
                    parts.append(part)
                    echo_id[str(call.get("id"))] = True
                push("model", parts)
            elif role == "tool":
                call_id = str(message.get("tool_call_id") or "")
                text = str(message.get("content") or "")
                name = names.get(call_id)
                if not name:
                    # The call it answers is no longer in the history.
                    push("user", [{"text": f"Tool result:\n{text}"}])
                    continue
                answer: Dict[str, Any] = {"name": name, "response": {"result": text}}
                if echo_id.get(call_id):
                    answer["id"] = call_id
                push("user", [{"functionResponse": answer}])
        return "\n\n".join(system_parts), contents

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        max_output_tokens: int,
        timeout: float,
    ) -> LLMResponse:
        system, contents = self._convert(messages)
        while True:
            generation: Dict[str, Any] = {"maxOutputTokens": int(max_output_tokens)}
            if self._thinking_level:
                generation["thinkingConfig"] = {"thinkingLevel": self._thinking_level}
            payload: Dict[str, Any] = {"contents": contents, "generationConfig": generation}
            if system:
                payload["systemInstruction"] = {"parts": [{"text": system}]}
            if tools:
                payload["tools"] = [{"functionDeclarations": self._declarations(tools)}]
            response = self._post(f"/models/{self.model}:generateContent", payload, timeout)
            if response.status_code == 200:
                return self._parse(response, tools)
            error = self._error(response)
            if (
                self._thinking_level
                and response.status_code == 400
                and "thinking" in error.message.lower()
            ):
                # A model that does not take a thinking level: ask without one.
                self._thinking_level = None
                continue
            if error.kind == "model_unavailable" and not self._fallback_attempted:
                self._fallback_attempted = True
                self._switch_model(self._available_models(), self._preferences, error.message)
                continue
            raise error

    def _available_models(self) -> List[str]:
        try:
            response = self._http.get("/models?pageSize=200", timeout=20.0)
        except httpx.HTTPError:
            return []
        if response.status_code != 200:
            return []
        try:
            listed = response.json().get("models") or []
        except (ValueError, AttributeError):
            return []
        names: List[str] = []
        for item in listed:
            if not isinstance(item, dict) or "generateContent" not in (item.get("supportedGenerationMethods") or []):
                continue
            name = str(item.get("name") or "")
            names.append(name[len("models/") :] if name.startswith("models/") else name)
        return [name for name in names if name]

    # ------------------------------------------------------------------ #
    # Response
    # ------------------------------------------------------------------ #
    def _error(self, response: httpx.Response) -> LLMError:
        status = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = {}
        if isinstance(body, list) and body:  # some gateways wrap the error in a list
            body = body[0]
        detail = body.get("error") if isinstance(body, dict) else None
        message, state, reasons, quota_ids = "", "", [], []
        retry_after: Optional[float] = None
        if isinstance(detail, dict):
            message = str(detail.get("message") or "")
            state = str(detail.get("status") or "")
            for item in detail.get("details") or []:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("@type") or "")
                if kind.endswith("ErrorInfo"):
                    reasons.append(str(item.get("reason") or ""))
                elif kind.endswith("RetryInfo"):
                    retry_after = _seconds(item.get("retryDelay"))
                elif kind.endswith("QuotaFailure"):
                    for violation in item.get("violations") or []:
                        if isinstance(violation, dict):
                            quota_ids.append(str(violation.get("quotaId") or violation.get("quotaMetric") or ""))
        else:
            message = (response.text or "")[:300]
        if retry_after is None:
            retry_after = _retry_after(response.headers, message)
        first_line = message.strip().splitlines()[0] if message.strip() else "no details"
        text = f"gemini returned HTTP {status}{f' ({state})' if state else ''}: {clip_text(first_line, 400)}"
        lowered = message.lower()
        bad_key = "API_KEY_INVALID" in reasons or "api key not valid" in lowered or "api key expired" in lowered
        daily = any("perday" in item.lower().replace("_", "") for item in quota_ids) or (
            status == 429 and _PER_DAY_RE.search(message) is not None
        )

        if status in (401, 403) or bad_key:
            kind = "auth"
            text += " Check the Gemini API key (GEMINI_API_KEY) and that its project may use the Gemini API."
        elif status == 402:
            kind = "quota"
        elif status == 429:
            kind = "rate_limit"
        elif status == 404:
            kind = "model_unavailable"
        elif state == "FAILED_PRECONDITION":
            # For example: the free tier is not offered in the caller's country.
            kind = "auth"
        elif status == 400 and ("exceeds the maximum number of tokens" in lowered or "too long" in lowered):
            kind = "too_large"
        elif status >= 500 or status == 408:
            kind = "server"
        else:
            kind = "bad_request"
        return LLMError(
            text, kind=kind, status_code=status, retry_after=retry_after, daily=daily, provider=self.provider
        )

    def _parse(self, response: httpx.Response, tools: Sequence[Dict[str, Any]] = ()) -> LLMResponse:
        try:
            body = response.json()
            if not isinstance(body, dict):
                raise TypeError("not an object")
        except (ValueError, TypeError) as exc:
            raise LLMError("gemini returned an unreadable response.", kind="server", provider=self.provider) from exc

        candidates = body.get("candidates") or []
        candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        finish = str(candidate.get("finishReason") or "")
        parts = (candidate.get("content") or {}).get("parts") or []
        parts = [part for part in parts if isinstance(part, dict)]
        if not candidate:
            reason = str((body.get("promptFeedback") or {}).get("blockReason") or "")
            raise LLMError(
                "gemini returned no answer" + (f" (the request was blocked: {reason})." if reason else "."),
                kind="blocked" if reason else "server",
                provider=self.provider,
            )
        if finish == "MALFORMED_FUNCTION_CALL":
            # The model tried to call a tool and the call could not be read.
            return_text = str(candidate.get("finishMessage") or "")
            raise LLMError(
                "gemini refused the model's reply because it was not a valid tool call (malformed function call).",
                kind="tool_format",
                status_code=200,
                failed_generation=return_text or None,
                provider=self.provider,
            )

        known = _tool_names(tools)
        texts: List[str] = []
        calls: List[ToolCall] = []
        wire_calls: List[Dict[str, Any]] = []
        renamed = False
        for part in parts:
            if isinstance(part.get("functionCall"), dict):
                function = part["functionCall"]
                self._calls += 1
                call_id = str(function.get("id") or f"gemini_call_{self._calls}")
                raw_name = str(function.get("name") or "")
                name = canonical_tool_name(raw_name, known) or raw_name
                renamed = renamed or name != raw_name
                arguments = function.get("args") if isinstance(function.get("args"), dict) else {}
                calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
                wire_calls.append(
                    {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
                )
            elif part.get("thought"):
                continue  # a summary of the model's reasoning, not part of its answer
            elif isinstance(part.get("text"), str):
                texts.append(part["text"])
        content = "".join(texts).strip()
        if not calls and not content and finish in _GEMINI_BLOCKED_REASONS:
            raise LLMError(
                f"gemini stopped without an answer ({finish.lower().replace('_', ' ')}).",
                kind="blocked",
                provider=self.provider,
            )

        assistant: Dict[str, Any] = {"role": "assistant", "content": content or None}
        if wire_calls:
            assistant["tool_calls"] = wire_calls
        if parts and not renamed:
            assistant["_gemini_parts"] = parts
        usage = body.get("usageMetadata") if isinstance(body.get("usageMetadata"), dict) else {}
        return LLMResponse(
            content=content,
            tool_calls=calls,
            finish_reason="length" if finish == "MAX_TOKENS" else "tool_calls" if calls else finish.lower() or "stop",
            model=str(body.get("modelVersion") or self.model),
            assistant_message=assistant,
            usage={
                "prompt_tokens": int(usage.get("promptTokenCount") or 0),
                "completion_tokens": int(usage.get("candidatesTokenCount") or 0)
                + int(usage.get("thoughtsTokenCount") or 0),
                "total_tokens": int(usage.get("totalTokenCount") or 0),
            },
            provider=self.provider,
        )


def _seconds(value: Any) -> Optional[float]:
    """A protobuf duration ("36s", "1.5s") as seconds."""
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    if isinstance(value, str) and value.strip().endswith("s"):
        try:
            return max(0.0, float(value.strip()[:-1]))
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- #
# Several providers behind one client
# --------------------------------------------------------------------------- #
# How long a provider is passed over after a failure that says nothing about
# when it will work again.
COOLDOWN_UNKNOWN_RATE_LIMIT = 30.0
# A router picks a model for every request. When the one it picked is throttled
# upstream, the next request gets another pick: no need to stay away for long.
COOLDOWN_ROUTER_PICK = 3.0
COOLDOWN_TOO_LARGE = 60.0
COOLDOWN_UNREACHABLE = 20.0
COOLDOWN_REFUSED_REPLIES = 30.0
# Replies in a row a provider may refuse as invalid tool calls before the
# conversation moves on to another one.
REFUSED_REPLIES_BEFORE_SWITCH = 2
_OUT_FOR_THE_RUN = ("auth", "quota", "model_unavailable", "bad_request", "blocked")
_SWITCH_KINDS = ("rate_limit", "too_large", "server", "network", "tool_format") + _OUT_FOR_THE_RUN
# What takes a provider out with every one of its models: the key or the account.
_OUT_FOR_THE_PROVIDER = ("auth", "quota")
# How long a model is waited for (see FailoverClient.patience): the longest when
# nobody else could take over, the shortest for one that has not answered yet
# in this run while another is ready - a provider that accepts a request and
# then hangs must not cost a third of the run.
PATIENCE_ALONE = 90.0
PATIENCE_UNPROVEN = 45.0


def is_router(model: str) -> bool:
    """A name that stands for a choice made per request: OpenRouter's own
    routers (``openrouter/free``, ``openrouter/auto``)."""
    return model.startswith("openrouter/")


def _counts_as_free(client: Any) -> bool:
    """OpenRouter's free allowance is one per account, shared by every free model."""
    model = str(getattr(client, "model", "") or "")
    return getattr(client, "provider", "") == "openrouter" and (model == "openrouter/free" or model.endswith(":free"))


@dataclass
class _Slot:
    client: ChatClient
    ready_at: float = 0.0  # on the pool's clock; before that the provider is passed over
    out: Optional[str] = None  # why it is not used again in this run
    refused_in_a_row: int = 0
    requests: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    slowest: float = 0.0  # the longest it took to answer in this run, in seconds
    # The models that answered, when the configured one is a router or an alias
    # that chooses (openrouter/free, gemini-flash-latest): name -> answers.
    served: Dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.client.provider} / {self.client.model}"


class FailoverClient:
    """Chat completion with fallbacks: one conversation, several providers.

    The first provider is used until it fails. What happens then depends on the
    failure: a rate limit takes the provider out until the limit lifts, a
    rejected key or an exhausted quota for the rest of the run, an outage for a
    short while. If another provider is ready it becomes the one in use and the
    error that is raised says so (``switched_to``): the caller repeats the
    request and it goes to the new provider. The pool stays with a provider
    that works; it does not hop back as soon as the first one has recovered.

    When every provider is waiting for its rate limit, the error carries the
    shortest of those waits, and the provider it belongs to is used next.

    A provider can be in the pool more than once, with different models
    (limits are counted per model). Each entry stands on its own, except that
    a rejected key or an empty account takes out all entries of that provider.
    """

    def __init__(self, clients: Sequence[ChatClient], *, clock: Callable[[], float] = time.monotonic) -> None:
        if not clients:
            raise LLMError("No LLM provider is configured.", kind="auth")
        self._slots = [_Slot(client) for client in clients]
        self._active = 0
        self._clock = clock
        self.switches = 0

    # What the rest of the code asks a client for: answered for the provider in use.
    @property
    def provider(self) -> str:
        return self._slots[self._active].client.provider

    @property
    def model(self) -> str:
        return self._slots[self._active].client.model

    @property
    def model_env_var(self) -> str:
        return str(getattr(self._slots[self._active].client, "model_env_var", "") or "")

    @property
    def assumed_tokens_per_minute(self) -> Optional[int]:
        return getattr(self._slots[self._active].client, "assumed_tokens_per_minute", None)

    @property
    def chain(self) -> List[str]:
        """Every provider of the pool as "provider / model", first choice first."""
        return [slot.label for slot in self._slots]

    def usage(self) -> List[Dict[str, Any]]:
        """What each provider did in this run, in the order of the chain."""
        return [
            {
                "provider": slot.client.provider,
                "model": slot.client.model,
                "requests": slot.requests,
                "failures": slot.failures,
                "input_tokens": slot.input_tokens,
                "output_tokens": slot.output_tokens,
                "out": slot.out,
                **({"served": dict(slot.served)} if slot.served else {}),
            }
            for slot in self._slots
        ]

    def patience(self) -> float:
        """Seconds the model in use is worth waiting for with the next request.

        Without streaming, a model that is writing a long reply and one that
        accepted the request and hangs look the same until the answer is
        there. So the wait depends on what there is to go by: a model that has
        answered in this run gets a multiple of the longest it took so far; one
        that has not is given up on sooner - provided somebody else is ready to
        take over, or will be by then. With nobody else, waiting is all there is.
        """
        now = self._clock()
        current = self._slots[self._active]
        others = [slot for slot in self._slots if slot is not current and slot.out is None]
        if not others:
            return PATIENCE_ALONE
        base = PATIENCE_UNPROVEN
        if current.requests:
            base = max(PATIENCE_UNPROVEN, min(PATIENCE_ALONE, 3.0 * current.slowest + 10.0))
        # Giving up on this one before another is free would only add a wait.
        free_in = max(0.0, min(slot.ready_at for slot in others) - now)
        return min(PATIENCE_ALONE, max(base, free_in + 15.0))

    def close(self) -> None:
        for slot in self._slots:
            try:
                slot.client.close()
            except Exception:  # closing must not hide the outcome of a run
                pass

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        max_output_tokens: int,
        timeout: float,
    ) -> LLMResponse:
        slot = self._slots[self._active]
        started = self._clock()
        try:
            response = slot.client.complete(
                messages, tools, max_output_tokens=max_output_tokens, timeout=timeout
            )
        except LLMError as exc:
            exc.provider = exc.provider or slot.client.provider
            slot.failures += 1
            self._after_failure(slot, exc)
            raise
        slot.requests += 1
        slot.refused_in_a_row = 0
        slot.slowest = max(slot.slowest, self._clock() - started)
        usage = response.usage or {}
        slot.input_tokens += _count(usage, "prompt_tokens", "input_tokens")
        slot.output_tokens += _count(usage, "completion_tokens", "output_tokens")
        response.provider = response.provider or slot.client.provider
        if response.model and response.model != slot.client.model:
            slot.served[response.model] = slot.served.get(response.model, 0) + 1
        return response

    # ------------------------------------------------------------------ #
    def _after_failure(self, slot: _Slot, exc: LLMError) -> None:
        now = self._clock()
        kind = exc.kind
        if kind == "rate_limit":
            if exc.daily:
                slot.out = "its daily limit is used up"
                if _counts_as_free(slot.client):
                    # Counted per account, not per model: the other free models are out too.
                    for other in self._slots:
                        if other.out is None and _counts_as_free(other.client):
                            other.out = slot.out
            elif exc.retry_after is None and is_router(slot.client.model) and "upstream" in exc.message.lower():
                # The model the router picked is throttled, not the account.
                slot.ready_at = now + COOLDOWN_ROUTER_PICK
            else:
                slot.ready_at = now + max(1.0, exc.retry_after or COOLDOWN_UNKNOWN_RATE_LIMIT)
        elif kind in _OUT_FOR_THE_RUN:
            slot.out = {
                "auth": "its API key was rejected",
                "quota": "its account has no credit left",
                "model_unavailable": "its model is not available",
            }.get(kind, "it refused the request")
            if kind in _OUT_FOR_THE_PROVIDER:
                # The key or the account: no other model of this provider will work either.
                for other in self._slots:
                    if other.out is None and other.client.provider == slot.client.provider:
                        other.out = slot.out
        elif kind == "too_large":
            slot.ready_at = now + COOLDOWN_TOO_LARGE
        elif kind in ("server", "network"):
            slot.ready_at = now + COOLDOWN_UNREACHABLE
        elif kind == "tool_format":
            slot.refused_in_a_row += 1
            if slot.refused_in_a_row < REFUSED_REPLIES_BEFORE_SWITCH:
                return  # one unusable reply: the same model gets to try again
            slot.ready_at = now + COOLDOWN_REFUSED_REPLIES
        if kind not in _SWITCH_KINDS:
            return

        current = self._slots.index(slot)
        usable = [index for index, other in enumerate(self._slots) if other.out is None]
        ready = [index for index in usable if index != current and self._slots[index].ready_at <= now]
        if ready:
            self._active = ready[0]
            self.switches += 1
            slot.refused_in_a_row = 0
            exc.switched_to = self._slots[ready[0]].label
            return
        # Nobody is ready. A provider that is merely unreachable or confused is
        # retried by the caller. One that is rate limited or out for the run is
        # not: the caller is told how long the shortest wait is, and the
        # provider that is free first answers the repeated request.
        if kind == "rate_limit":
            waiting = usable
        elif kind in _OUT_FOR_THE_RUN:
            waiting = [index for index in usable if index != current]
        else:
            return
        if not waiting:
            return
        exc.all_limited = len(self._slots) > 1
        soonest = min(waiting, key=lambda index: self._slots[index].ready_at)
        if soonest != current:
            self._active = soonest
            self.switches += 1
            exc.next_provider = self._slots[soonest].label
            if kind != "rate_limit":
                # The provider that failed is out; another one is only rate
                # limited. That is a wait, not the end of the run.
                exc.message = (
                    f"{exc.message} {self._slots[soonest].client.provider} is used next; "
                    "it is rate limited for a moment."
                )
                exc.kind = "rate_limit"
            exc.daily = False
        exc.retry_after = max(0.0, self._slots[soonest].ready_at - now)


def _count(usage: Dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


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


def _thinking_level(raw: Optional[str]) -> Optional[str]:
    """GEMINI_THINKING_LEVEL: minimal, low (our default), medium, high, or
    "default" to leave it to the model."""
    value = (raw or "low").strip().lower()
    if value in ("default", "auto", "none", ""):
        return None
    return value if value in ("minimal", "low", "medium", "high") else "low"


def build_provider_client(
    settings: Settings,
    provider: str,
    *,
    model: Optional[str] = None,
    transport: Optional[httpx.BaseTransport] = None,
    on_notice: Optional[Notice] = None,
) -> ChatClient:
    """The client for one provider, configured from ``settings``, for
    ``model`` or - without one - the first model configured for the provider."""
    model = model or settings.model_for(provider) or ""
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
            model=model,
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
            model=model,
            model_env_var="OPENAI_MODEL",
            model_preferences=OPENAI_MODEL_PREFERENCES,
            optional_params=optional,
            transport=transport,
            on_notice=on_notice,
        )
    if provider == "openrouter":
        optional = {}
        # The models behind OpenRouter differ in what they accept; anything one
        # of them rejects by name is dropped and the request repeated.
        effort = (env_str("OPENROUTER_REASONING_EFFORT", "low") or "low").lower()
        if effort != "default":
            optional["reasoning_effort"] = effort
        temperature = _temperature_setting(env_str("OPENROUTER_TEMPERATURE"), DEFAULT_GROQ_TEMPERATURE)
        if temperature is not None:
            optional["temperature"] = temperature
        return OpenAICompatibleClient(
            provider="openrouter",
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key or "",
            model=model,
            model_env_var="OPENROUTER_MODEL",
            optional_params=optional,
            output_tokens_param="max_tokens",
            # Shown on OpenRouter's side as the application the requests come from.
            extra_headers={"X-Title": "Jules Clone"},
            # No tokens-per-minute limit is reported; free models are limited by
            # requests, and their context windows are large.
            assumed_tokens_per_minute=200_000,
            # Its catalogue has hundreds of models: there is no sensible automatic pick.
            list_models=False,
            transport=transport,
            on_notice=on_notice,
        )
    if provider == "gemini":
        return GeminiClient(
            base_url=settings.gemini_base_url,
            api_key=settings.gemini_api_key or "",
            model=model,
            model_preferences=GEMINI_MODEL_PREFERENCES,
            thinking_level=_thinking_level(env_str("GEMINI_THINKING_LEVEL")),
            transport=transport,
            on_notice=on_notice,
        )
    if provider == "anthropic":
        return AnthropicClient(
            base_url=settings.anthropic_base_url,
            api_key=settings.anthropic_api_key or "",
            model=model,
            model_preferences=ANTHROPIC_MODEL_PREFERENCES,
            transport=transport,
            on_notice=on_notice,
        )
    raise LLMError(f"Unknown LLM provider '{provider}'.", kind="auth")


def create_llm_client(
    settings: Settings,
    *,
    transport: Optional[httpx.BaseTransport] = None,
    on_notice: Optional[Notice] = None,
) -> ChatClient:
    """The client a task talks to: the first model of the first provider, with
    the other configured providers and models behind it as fallbacks (in the
    order of ``Settings.llm_slots``)."""
    slots = settings.llm_slots
    if not slots:
        raise LLMError(
            "No LLM provider is configured. Set "
            + ", ".join(PROVIDER_KEY_VARIABLES[name] for name in LLM_PROVIDERS[:-1])
            + f" or {PROVIDER_KEY_VARIABLES[LLM_PROVIDERS[-1]]}.",
            kind="auth",
        )
    clients = [
        build_provider_client(settings, name, model=model, transport=transport, on_notice=on_notice)
        for name, model in slots
    ]
    return clients[0] if len(clients) == 1 else FailoverClient(clients)


# --------------------------------------------------------------------------- #
# Trying a key out
# --------------------------------------------------------------------------- #
_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "confirm",
        "description": "Confirm that the connection works.",
        "parameters": {
            "type": "object",
            "properties": {"ready": {"type": "boolean", "description": "Always true."}},
            "required": ["ready"],
        },
    },
}
_PROBE_MESSAGES = [
    {"role": "system", "content": "You are a connection test. Reply only by calling the provided tool."},
    {"role": "user", "content": "Call the confirm tool with ready set to true."},
]


def probe_provider(
    settings: Settings,
    provider: str,
    *,
    transport: Optional[httpx.BaseTransport] = None,
    timeout: float = 25.0,
) -> Dict[str, Any]:
    """Send one small request with a tool to ``provider`` and say what came back.

    This is the same kind of request the agent makes, so a key that passes can
    run tasks: the key is accepted, the model exists and it calls tools.
    When the provider is given several models, each one is tried: the result
    describes the first, and ``models`` lists them all.
    Never raises; a failure is described in the result.
    """
    label = PROVIDER_LABELS.get(provider, provider)
    result: Dict[str, Any] = {
        "provider": provider,
        "model": settings.model_for(provider),
        "ok": False,
        "tool_call": False,
        "latency_ms": None,
        "kind": None,
        "message": "",
    }
    if provider not in LLM_PROVIDERS:
        result.update(kind="unknown_provider", message=f"'{provider}' is not a provider this server knows.")
        return result
    if not settings.api_key_for(provider):
        result.update(kind="no_key", message=f"No API key for {label} was sent or is set on the server.")
        return result
    tried: List[Dict[str, Any]] = []
    for model in settings.models_for(provider):
        if tried and tried[0]["kind"] in _OUT_FOR_THE_PROVIDER:
            break  # the key or the account: the other models would say the same
        tried.append(_probe_model(settings, provider, model, label, transport, timeout))
    result.update(tried[0])
    if len(settings.models_for(provider)) > 1:
        result["models"] = tried
    return result


def _probe_model(
    settings: Settings,
    provider: str,
    model: str,
    label: str,
    transport: Optional[httpx.BaseTransport],
    timeout: float,
) -> Dict[str, Any]:
    """One test request to one model of a provider."""
    outcome: Dict[str, Any] = {"model": model, "ok": False, "tool_call": False, "latency_ms": None, "kind": None, "message": ""}
    notices: List[str] = []
    client = build_provider_client(settings, provider, model=model, transport=transport, on_notice=notices.append)
    started = time.monotonic()
    try:
        response = client.complete(_PROBE_MESSAGES, [_PROBE_TOOL], max_output_tokens=1024, timeout=timeout)
    except LLMError as exc:
        outcome.update(kind=exc.kind, message=exc.message, model=client.model)
        return outcome
    finally:
        client.close()
    called = any(call.name == "confirm" for call in response.tool_calls)
    # A router or an alias answers with the model it chose.
    chosen = response.model if response.model and response.model != client.model else ""
    who = f"{client.model} (answered by {chosen})" if chosen else client.model
    outcome.update(
        ok=True,
        tool_call=called,
        latency_ms=int((time.monotonic() - started) * 1000),
        model=client.model,
        message=(
            f"{label} accepted the key and {who} called the test tool."
            if called
            else f"{label} accepted the key, but {who} answered in text instead of calling the "
            "test tool. The agent needs a model that calls tools reliably."
        )
        + (" " + " ".join(notices) if notices else ""),
    )
    if chosen:
        outcome["answered_by"] = chosen
    return outcome
