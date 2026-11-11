"""LLM provider clients, exercised against mocked HTTP transports."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

import httpx
import pytest

from apps.api.config import get_settings
from apps.api.services.llm import (
    AnthropicClient,
    LLMError,
    OpenAICompatibleClient,
    create_llm_client,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    }
]
MESSAGES = [{"role": "system", "content": "be useful"}, {"role": "user", "content": "hello"}]


def chat_completion(model: str, *, content: Any = None, tool_calls: Any = None, **extra: Any) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": content, **extra}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class Recorder:
    """A mock transport that records requests and answers from a handler."""

    def __init__(self, handler: Callable[[httpx.Request, Dict[str, Any]], httpx.Response]) -> None:
        self.requests: List[httpx.Request] = []
        self.bodies: List[Dict[str, Any]] = []
        self._handler = handler
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append(request)
        self.bodies.append(body)
        return self._handler(request, body)


def groq_client(recorder: Recorder, model: str = "openai/gpt-oss-120b", **kwargs: Any) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key="gsk_test",
        model=model,
        model_env_var="GROQ_MODEL",
        model_preferences=("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"),
        transport=recorder.transport,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# OpenAI-compatible (Groq / OpenAI)
# --------------------------------------------------------------------------- #
def test_request_shape_and_tool_call_parsing():
    recorder = Recorder(
        lambda request, body: httpx.Response(
            200,
            headers={"x-ratelimit-limit-tokens": "8000"},
            json=chat_completion(
                body["model"],
                reasoning="thinking out loud",  # provider-specific extra field
                tool_calls=[
                    {"id": "call_abc", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
                ],
            ),
        )
    )
    client = groq_client(recorder, optional_params={"reasoning_effort": "low"})
    response = client.complete(MESSAGES, TOOLS, max_output_tokens=1234, timeout=30)

    request = recorder.requests[0]
    assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer gsk_test"
    body = recorder.bodies[0]
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["max_completion_tokens"] == 1234
    assert body["tool_choice"] == "auto" and body["tools"] == TOOLS
    assert body["reasoning_effort"] == "low"
    assert body["messages"] == MESSAGES

    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [
        ("call_abc", "read_file", {"path": "a.py"})
    ]
    assert response.tokens_per_minute_limit == 8000
    # The history entry contains only fields every compatible API accepts back.
    assert response.assistant_message == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_abc", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
        ],
    }


def test_private_history_keys_are_not_sent():
    recorder = Recorder(lambda request, body: httpx.Response(200, json=chat_completion(body["model"], content="ok")))
    history = MESSAGES + [{"role": "assistant", "content": "x", "_anthropic_content": [{"type": "text", "text": "x"}]}]
    groq_client(recorder).complete(history, [], max_output_tokens=100, timeout=30)
    assert all("_anthropic_content" not in message for message in recorder.bodies[0]["messages"])
    assert "tools" not in recorder.bodies[0]


@pytest.mark.parametrize(
    "status, error",
    [
        # What Groq answers for a model that was shut down or moved to another tier.
        (404, {"message": "The model `llama-3.3-70b-versatile` does not exist or you do not have access to it.", "type": "invalid_request_error", "code": "model_not_found"}),
        (400, {"message": "The model `llama-3.3-70b-versatile` has been decommissioned and is no longer supported.", "type": "invalid_request_error", "code": "model_decommissioned"}),
        # The key is fine but this model is not enabled for the account or project.
        (403, {"message": "The model `llama-3.3-70b-versatile` is blocked at the project level.", "type": "permissions_error", "code": "model_permission_blocked_project"}),
        (403, {"message": "The model `llama-3.3-70b-versatile` requires terms acceptance.", "type": "invalid_request_error", "code": "model_terms_required"}),
    ],
)
def test_retired_model_falls_back_to_an_available_one(status, error):
    """Regression: the project hardcoded llama-3.3-70b-versatile, which Groq shut
    down on 2026-08-16. Every run then errored and was reported as a success."""

    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={"data": [{"id": "whisper-large-v3"}, {"id": "openai/gpt-oss-20b"}, {"id": "openai/gpt-oss-120b"}]},
            )
        if body["model"] == "llama-3.3-70b-versatile":
            return httpx.Response(status, json={"error": error})
        return httpx.Response(200, json=chat_completion(body["model"], content="hello"))

    recorder = Recorder(handler)
    notices: List[str] = []
    client = groq_client(recorder, model="llama-3.3-70b-versatile", on_notice=notices.append)
    response = client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)

    assert response.content == "hello"
    assert client.model == "openai/gpt-oss-120b"  # first preference that the account can use
    assert len(notices) == 1 and "llama-3.3-70b-versatile" in notices[0] and "GROQ_MODEL" in notices[0]
    assert [body.get("model") for body in recorder.bodies if body] == [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
    ]


def test_retired_model_without_known_replacement_says_what_to_set():
    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "whisper-large-v3"}, {"id": "some/new-chat-model"}]})
        return httpx.Response(404, json={"error": {"message": "The model `x` does not exist", "code": "model_not_found"}})

    client = groq_client(Recorder(handler), model="x")
    with pytest.raises(LLMError) as caught:
        client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert caught.value.kind == "model_unavailable"
    assert "GROQ_MODEL" in caught.value.message and "some/new-chat-model" in caught.value.message
    assert "whisper" not in caught.value.message  # not a chat model


def test_unsupported_optional_parameter_is_dropped_and_retried():
    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        if "reasoning_effort" in body:
            return httpx.Response(
                400, json={"error": {"message": "`reasoning_effort` is not supported with this model", "type": "invalid_request_error"}}
            )
        return httpx.Response(200, json=chat_completion(body["model"], content="fine"))

    recorder = Recorder(handler)
    client = groq_client(recorder, optional_params={"reasoning_effort": "low"})
    assert client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content == "fine"
    assert len(recorder.bodies) == 2 and "reasoning_effort" not in recorder.bodies[1]
    client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert "reasoning_effort" not in recorder.bodies[2]  # remembered for later calls


@pytest.mark.parametrize(
    "status, headers, error, kind, retry_after",
    [
        (429, {"retry-after": "7"}, {"message": "Rate limit reached", "code": "rate_limit_exceeded"}, "rate_limit", 7.0),
        (429, {}, {"message": "Rate limit reached. Please try again in 2m3.5s."}, "rate_limit", 123.5),
        (429, {}, {"message": "Please try again in 450ms."}, "rate_limit", 0.45),
        (413, {}, {"message": "Request too large for model on tokens per minute (TPM): Limit 8000, Requested 9920"}, "too_large", None),
        (401, {}, {"message": "Invalid API Key"}, "auth", None),
        (403, {}, {"message": "Organization restricted", "code": "organization_restricted"}, "auth", None),
        (503, {}, {"message": "The model is overloaded", "code": "model_overloaded"}, "server", None),
        (400, {}, {"message": "Failed to call a function.", "code": "tool_use_failed"}, "tool_format", None),
        (503, {}, {"message": "Service unavailable"}, "server", None),
        (400, {}, {"message": "messages.2.content must be a string"}, "bad_request", None),
    ],
)
def test_error_classification(status, headers, error, kind, retry_after):
    recorder = Recorder(lambda request, body: httpx.Response(status, headers=headers, json={"error": error}))
    with pytest.raises(LLMError) as caught:
        groq_client(recorder).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert caught.value.kind == kind
    assert caught.value.status_code == status
    assert caught.value.retry_after == retry_after
    assert error["message"] in caught.value.message


def test_malformed_tool_arguments_are_reported_not_raised():
    recorder = Recorder(
        lambda request, body: httpx.Response(
            200,
            json=chat_completion(
                body["model"],
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{not json"}}],
            ),
        )
    )
    response = groq_client(recorder).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert response.tool_calls[0].arguments == {}
    assert "not valid JSON" in (response.tool_calls[0].error or "")


def test_network_failure_is_an_llm_error():
    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(LLMError) as caught:
        groq_client(Recorder(handler)).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert caught.value.kind == "network"


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
def test_anthropic_request_conversion_and_response_parsing():
    blocks = [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": "Reading the file."},
        {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}},
    ]
    recorder = Recorder(
        lambda request, body: httpx.Response(
            200,
            headers={"anthropic-ratelimit-input-tokens-limit": "30000"},
            json={"model": body["model"], "content": blocks, "stop_reason": "tool_use", "usage": {}},
        )
    )
    client = AnthropicClient(
        base_url="https://api.anthropic.com", api_key="sk-ant", model="claude-sonnet-5-5", transport=recorder.transport
    )
    history = MESSAGES + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "toolu_0", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b.py"}'}}],
        },
        {"role": "tool", "tool_call_id": "toolu_0", "content": "file b"},
        {"role": "user", "content": "continue"},
    ]
    response = client.complete(history, TOOLS, max_output_tokens=2000, timeout=30)

    request, body = recorder.requests[0], recorder.bodies[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "sk-ant" and request.headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "be useful" and body["max_tokens"] == 2000
    assert body["tools"] == [
        {"name": "read_file", "description": "Read a file", "input_schema": TOOLS[0]["function"]["parameters"]}
    ]
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_0", "name": "read_file", "input": {"path": "b.py"}}]},
        {
            "role": "user",  # the tool result and the following user text share one turn
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_0", "content": "file b"},
                {"type": "text", "text": "continue"},
            ],
        },
    ]

    assert response.content == "Reading the file."
    assert [(c.id, c.name, c.arguments) for c in response.tool_calls] == [("toolu_1", "read_file", {"path": "a.py"})]
    assert response.tokens_per_minute_limit == 30000
    # Provider blocks (including thinking) are kept so they can be replayed verbatim.
    assert response.assistant_message["_anthropic_content"] == blocks

    client.complete(history + [response.assistant_message], TOOLS, max_output_tokens=2000, timeout=30)
    assert recorder.bodies[1]["messages"][-1] == {"role": "assistant", "content": blocks}


def test_anthropic_retired_model_falls_back():
    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "claude-opus-5-5"}, {"id": "claude-sonnet-5-5"}]})
        if body["model"] == "claude-3-5-sonnet-20241022":
            return httpx.Response(404, json={"type": "error", "error": {"type": "not_found_error", "message": "model: claude-3-5-sonnet-20241022"}})
        return httpx.Response(200, json={"model": body["model"], "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"})

    notices: List[str] = []
    client = AnthropicClient(
        base_url="https://api.anthropic.com",
        api_key="sk-ant",
        model="claude-3-5-sonnet-20241022",
        model_preferences=("claude-sonnet-5-5", "claude-haiku-5-5"),
        transport=Recorder(handler).transport,
        on_notice=notices.append,
    )
    assert client.complete(MESSAGES, [], max_output_tokens=100, timeout=30).content == "hi"
    assert client.model == "claude-sonnet-5-5" and len(notices) == 1


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #
def test_provider_selection_and_defaults(monkeypatch):
    with pytest.raises(LLMError):
        create_llm_client(get_settings())

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    assert create_llm_client(get_settings()).provider == "anthropic"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert create_llm_client(get_settings()).provider == "openai"
    monkeypatch.setenv("GROQ_API_KEY", "gsk")
    groq = create_llm_client(get_settings())
    assert (groq.provider, groq.model) == ("groq", "openai/gpt-oss-120b")
    assert groq.model != "llama-3.3-70b-versatile"

    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    assert create_llm_client(get_settings()).model == "qwen/qwen3.8-27b"
    monkeypatch.setenv("JULES_LLM_PROVIDER", "anthropic")
    assert create_llm_client(get_settings()).provider == "anthropic"


def test_placeholder_values_from_env_example_count_as_unset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "your_groq_api_key_here")
    monkeypatch.setenv("E2B_API_KEY", "   ")
    settings = get_settings()
    assert settings.groq_api_key is None and settings.e2b_api_key is None
    assert settings.llm_provider is None
