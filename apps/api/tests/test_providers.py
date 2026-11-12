"""Gemini and OpenRouter clients, and what is done about replies a provider refuses.

Everything runs against mocked HTTP transports that answer the way the
providers document (and, for Groq, the way production answered).
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List

import httpx
import pytest

from apps.api.config import DEFAULT_GEMINI_MODEL, DEFAULT_OPENROUTER_MODEL, get_settings, split_models
from apps.api.services.llm import (
    GEMINI_FOREIGN_CALL_SIGNATURE,
    FailoverClient,
    GeminiClient,
    LLMError,
    OpenAICompatibleClient,
    build_provider_client,
    calls_in_generation,
    canonical_tool_name,
    create_llm_client,
    probe_provider,
)

from .test_llm import MESSAGES, TOOLS, Recorder, chat_completion, groq_client

WRITE_TOOLS = TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {"type": "function", "function": {"name": "finish", "description": "Done", "parameters": {"type": "object", "properties": {}}}},
]


def gemini_client(recorder: Recorder, **kwargs: Any) -> GeminiClient:
    kwargs.setdefault("model", "gemini-flash-latest")
    return GeminiClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="AQ.test-key",
        transport=recorder.transport,
        **kwargs,
    )


def gemini_answer(parts: List[Dict[str, Any]], finish: str = "STOP", **extra: Any) -> Dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": finish, "index": 0}],
        "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 30, "thoughtsTokenCount": 12, "totalTokenCount": 162},
        "modelVersion": "gemini-3.8-flash",
        **extra,
    }


def gemini_error(status: int, state: str, message: str, details: Any = None) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "message": message, "status": state, "details": details or []}})


# --------------------------------------------------------------------------- #
# Gemini: request and response
# --------------------------------------------------------------------------- #
def test_gemini_request_is_the_documented_generate_content_call():
    recorder = Recorder(
        lambda request, body: httpx.Response(
            200,
            json=gemini_answer(
                [
                    {"text": "Reading it."},
                    {"functionCall": {"name": "read_file", "args": {"path": "calc.py"}, "id": "fc-1"}, "thoughtSignature": "SIG-A"},
                ]
            ),
        )
    )
    response = gemini_client(recorder).complete(MESSAGES, WRITE_TOOLS, max_output_tokens=2048, timeout=30)
    request, body = recorder.requests[0], recorder.bodies[0]
    # The call Google AI Studio shows next to a new key: this URL, the key in this header.
    assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"
    assert request.headers["x-goog-api-key"] == "AQ.test-key" and "authorization" not in request.headers
    assert body["systemInstruction"] == {"parts": [{"text": "be useful"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert body["generationConfig"] == {"maxOutputTokens": 2048, "thinkingConfig": {"thinkingLevel": "low"}}
    # Sampling parameters are ignored by current models and announced to become errors.
    assert "temperature" not in json.dumps(body)
    declarations = body["tools"][0]["functionDeclarations"]
    assert [item["name"] for item in declarations] == ["read_file", "write_file", "finish"]
    assert declarations[1]["parameters"] == {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }  # only what a function declaration accepts: no additionalProperties
    assert "parameters" not in declarations[2]  # an object without properties is refused

    assert response.provider == "gemini" and response.model == "gemini-3.8-flash"
    assert response.content == "Reading it." and response.finish_reason == "tool_calls"
    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [("fc-1", "read_file", {"path": "calc.py"})]
    assert response.usage == {"prompt_tokens": 120, "completion_tokens": 42, "total_tokens": 162}
    assert response.assistant_message["tool_calls"][0] == {
        "id": "fc-1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "calc.py"}'},
    }


def test_gemini_gets_its_own_parts_back_with_every_signature_in_place():
    """Gemini 3 and later refuse a history in which a function call of the
    current turn comes back without the signature it was sent with."""
    first = gemini_answer(
        [
            {"functionCall": {"name": "read_file", "args": {"path": "a.py"}, "id": "fc-1"}, "thoughtSignature": "SIG-A"},
            {"functionCall": {"name": "read_file", "args": {"path": "b.py"}, "id": "fc-2"}},
        ]
    )
    answers = [first, gemini_answer([{"text": "Both read.", "thoughtSignature": "SIG-B"}])]
    recorder = Recorder(lambda request, body: httpx.Response(200, json=answers[len(recorder.bodies) - 1]))
    client = gemini_client(recorder)
    reply = client.complete(MESSAGES, TOOLS, max_output_tokens=500, timeout=30)
    history = MESSAGES + [
        reply.assistant_message,
        {"role": "tool", "tool_call_id": "fc-2", "content": "B"},  # answered in a different order
        {"role": "tool", "tool_call_id": "fc-1", "content": "A"},
        {"role": "user", "content": "Go on."},
    ]
    client.complete(history, TOOLS, max_output_tokens=500, timeout=30)
    contents = recorder.bodies[1]["contents"]
    assert [item["role"] for item in contents] == ["user", "model", "user"]
    assert contents[1]["parts"] == first["candidates"][0]["content"]["parts"]  # unchanged, signature included
    # Results are grouped behind the calls (never interleaved) and carry the id of their call.
    assert contents[2]["parts"] == [
        {"functionResponse": {"name": "read_file", "response": {"result": "B"}, "id": "fc-2"}},
        {"functionResponse": {"name": "read_file", "response": {"result": "A"}, "id": "fc-1"}},
        {"text": "Go on."},
    ]


def test_calls_made_by_another_provider_are_marked_for_gemini():
    """After a switch from Groq the history holds calls Gemini never signed.
    Google documents a signature value for exactly this; without it the first
    request after the switch is refused."""
    history = MESSAGES + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                {"id": "call_b", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b.py"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "A"},
        {"role": "tool", "tool_call_id": "call_b", "content": "B"},
        {"role": "assistant", "content": "Now writing.", "tool_calls": [
            {"id": "call_c", "type": "function", "function": {"name": "write_file", "arguments": "not json"}},
        ]},
        {"role": "tool", "tool_call_id": "call_c", "content": "Error: unreadable"},
        {"role": "tool", "tool_call_id": "call_gone", "content": "orphan"},
    ]
    recorder = Recorder(lambda request, body: httpx.Response(200, json=gemini_answer([{"text": "ok"}])))
    gemini_client(recorder).complete(history, WRITE_TOOLS, max_output_tokens=500, timeout=30)
    contents = recorder.bodies[0]["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [
            {"functionCall": {"name": "read_file", "args": {"path": "a.py"}, "id": "call_a"}, "thoughtSignature": GEMINI_FOREIGN_CALL_SIGNATURE},
            {"functionCall": {"name": "read_file", "args": {"path": "b.py"}, "id": "call_b"}},
        ],
    }
    assert GEMINI_FOREIGN_CALL_SIGNATURE == "context_engineering_is_the_way_to_go"
    assert contents[2]["parts"] == [
        {"functionResponse": {"name": "read_file", "response": {"result": "A"}, "id": "call_a"}},
        {"functionResponse": {"name": "read_file", "response": {"result": "B"}, "id": "call_b"}},
    ]
    assert contents[3]["parts"] == [
        {"text": "Now writing."},
        {"functionCall": {"name": "write_file", "args": {}, "id": "call_c"}, "thoughtSignature": GEMINI_FOREIGN_CALL_SIGNATURE},
    ]
    # A result whose call is no longer in the history is passed on as text.
    assert contents[4]["parts"][-1] == {"text": "Tool result:\norphan"}


def test_gemini_calls_without_an_id_are_answered_without_one():
    recorder = Recorder(
        lambda request, body: httpx.Response(
            200, json=gemini_answer([{"functionCall": {"name": "read_file", "args": {"path": "a.py"}}, "thoughtSignature": "S"}])
        )
    )
    client = gemini_client(recorder)
    reply = client.complete(MESSAGES, TOOLS, max_output_tokens=500, timeout=30)
    call_id = reply.tool_calls[0].id
    assert call_id == "gemini_call_1"
    client.complete(MESSAGES + [reply.assistant_message, {"role": "tool", "tool_call_id": call_id, "content": "A"}], TOOLS, max_output_tokens=500, timeout=30)
    assert recorder.bodies[1]["contents"][-1]["parts"] == [{"functionResponse": {"name": "read_file", "response": {"result": "A"}}}]


def test_gemini_thought_summaries_are_not_part_of_the_answer_and_cut_off_replies_are_reported():
    recorder = Recorder(
        lambda request, body: httpx.Response(
            200, json=gemini_answer([{"text": "thinking about it", "thought": True}, {"text": "Half an ans"}], finish="MAX_TOKENS")
        )
    )
    response = gemini_client(recorder).complete(MESSAGES, TOOLS, max_output_tokens=50, timeout=30)
    assert response.content == "Half an ans" and response.finish_reason == "length" and response.tool_calls == []


# --------------------------------------------------------------------------- #
# Gemini: failures
# --------------------------------------------------------------------------- #
QUOTA_PER_MINUTE = [
    {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests", "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}],
    },
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "36s"},
]
QUOTA_PER_DAY = [
    {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
    },
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"},
]


@pytest.mark.parametrize(
    "response, kind, retry_after, daily",
    [
        # What the API answers for a wrong key: 400, not 401.
        (
            gemini_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]),
            "auth", None, False,
        ),
        (gemini_error(403, "PERMISSION_DENIED", "Your API key was reported as leaked."), "auth", None, False),
        (gemini_error(400, "FAILED_PRECONDITION", "Gemini API free tier is not available in your country."), "auth", None, False),
        (gemini_error(402, "RESOURCE_EXHAUSTED", "Your Prepay credit balance is depleted."), "quota", None, False),
        (gemini_error(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota.", QUOTA_PER_MINUTE), "rate_limit", 36.0, False),
        (gemini_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded. Please retry in 7.5s."), "rate_limit", 7.5, False),
        (gemini_error(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota.", QUOTA_PER_DAY), "rate_limit", 12.0, True),
        (gemini_error(503, "UNAVAILABLE", "The model is overloaded. Please try again later."), "server", None, False),
        (gemini_error(500, "INTERNAL", "An internal error has occurred."), "server", None, False),
        (gemini_error(400, "INVALID_ARGUMENT", "Function call `read_file` in the 2. content block is missing a `thought_signature`."), "bad_request", None, False),
        (gemini_error(400, "INVALID_ARGUMENT", "The input token count (1200000) exceeds the maximum number of tokens allowed (1048576)."), "too_large", None, False),
    ],
)
def test_gemini_error_classification(response, kind, retry_after, daily):
    recorder = Recorder(lambda request, body: response)
    with pytest.raises(LLMError) as caught:
        gemini_client(recorder).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    error = caught.value
    assert (error.kind, error.retry_after, error.daily, error.provider) == (kind, retry_after, daily, "gemini")
    assert error.message.startswith(f"gemini returned HTTP {response.status_code} (")
    assert "AQ.test-key" not in error.message


def test_gemini_model_that_takes_no_thinking_level_is_asked_without_one():
    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        if "thinkingConfig" in body["generationConfig"]:
            return gemini_error(400, "INVALID_ARGUMENT", "Thinking level is not supported for this model.")
        return httpx.Response(200, json=gemini_answer([{"text": "hi"}]))

    recorder = Recorder(handler)
    client = gemini_client(recorder)
    assert client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content == "hi"
    client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert ["thinkingConfig" in body["generationConfig"] for body in recorder.bodies] == [True, False, False]

    # Left to the model when configured so.
    recorder = Recorder(lambda request, body: httpx.Response(200, json=gemini_answer([{"text": "hi"}])))
    gemini_client(recorder, thinking_level=None).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert recorder.bodies[0]["generationConfig"] == {"maxOutputTokens": 100}


def test_gemini_model_that_is_gone_is_replaced_by_one_that_exists():
    notices: List[str] = []

    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        if request.method == "GET":
            assert request.url.path == "/v1beta/models"
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "models/gemini-embedding-2", "supportedGenerationMethods": ["embedContent"]},
                        {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent", "countTokens"]},
                    ]
                },
            )
        if "gemini-9-retired" in request.url.path:
            return gemini_error(404, "NOT_FOUND", "models/gemini-9-retired is not found for API version v1beta.")
        return httpx.Response(200, json=gemini_answer([{"text": "hi"}]))

    recorder = Recorder(handler)
    client = gemini_client(
        recorder, model="models/gemini-9-retired", model_preferences=("gemini-flash-latest", "gemini-3.8-flash"), on_notice=notices.append
    )
    assert client.model == "gemini-9-retired"
    assert client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content == "hi"
    assert client.model == "gemini-3.8-flash" and "using 'gemini-3.8-flash' instead" in notices[0]


def test_gemini_unusable_answers_are_errors_not_empty_replies():
    recorder = Recorder(lambda request, body: httpx.Response(200, json=gemini_answer([], finish="MALFORMED_FUNCTION_CALL", )))
    with pytest.raises(LLMError) as caught:
        gemini_client(recorder).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert caught.value.kind == "tool_format"  # the agent asks again, as with any refused reply

    recorder = Recorder(lambda request, body: httpx.Response(200, json={"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}))
    with pytest.raises(LLMError) as caught:
        gemini_client(recorder).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert caught.value.kind == "blocked" and "PROHIBITED_CONTENT" in caught.value.message

    recorder = Recorder(lambda request, body: httpx.Response(200, json=gemini_answer([], finish="SAFETY")))
    with pytest.raises(LLMError) as caught:
        gemini_client(recorder).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert caught.value.kind == "blocked"


# --------------------------------------------------------------------------- #
# OpenRouter
# --------------------------------------------------------------------------- #
def openrouter_client(recorder: Recorder, monkeypatch) -> OpenAICompatibleClient:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    client = build_provider_client(get_settings(), "openrouter", transport=recorder.transport)
    assert isinstance(client, OpenAICompatibleClient)
    return client


def test_openrouter_request(monkeypatch):
    recorder = Recorder(
        lambda request, body: httpx.Response(
            200,
            json=chat_completion(
                "baidu/ernie-5.5-flash-preview:free",
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}],
            ),
        )
    )
    client = openrouter_client(recorder, monkeypatch)
    response = client.complete(MESSAGES, TOOLS, max_output_tokens=900, timeout=30)
    request, body = recorder.requests[0], recorder.bodies[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-or-test" and request.headers["x-title"] == "Jules Clone"
    # Asked of OpenRouter's free-models router; the answer names the model it picked.
    assert body["model"] == DEFAULT_OPENROUTER_MODEL == "openrouter/free"
    assert response.model == "baidu/ernie-5.5-flash-preview:free"
    assert body["max_tokens"] == 900 and "max_completion_tokens" not in body
    assert body["tool_choice"] == "auto" and body["temperature"] == 0.2 and body["reasoning_effort"] == "low"
    assert response.provider == "openrouter" and response.tool_calls[0].name == "read_file"
    assert response.tokens_per_minute_limit is None and client.assumed_tokens_per_minute == 200_000


def test_openrouter_failure_reported_inside_a_200_answer(monkeypatch):
    """OpenRouter answers 200 once the model behind it has started and puts a
    failure of that model into the body."""
    recorder = Recorder(
        lambda request, body: httpx.Response(
            200, json={"error": {"code": 502, "message": "Provider returned error", "metadata": {"raw": "upstream connect error", "provider_name": "Baidu"}}}
        )
    )
    with pytest.raises(LLMError) as caught:
        openrouter_client(recorder, monkeypatch).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    error = caught.value
    assert error.kind == "server" and error.status_code == 502
    assert error.message == "openrouter returned HTTP 502: Provider returned error (upstream connect error)"


@pytest.mark.parametrize(
    "status, headers, error, kind, daily",
    [
        (401, {}, {"code": 401, "message": "No auth credentials found"}, "auth", False),
        (402, {}, {"code": 402, "message": "Insufficient credits. Add more using https://openrouter.ai/credits"}, "quota", False),
        # Credits are settling: OpenRouter says when to come back.
        (402, {"Retry-After": "3"}, {"code": 402, "message": "This request exceeds your in-flight budget", "metadata": {"limit_source": "openrouter_in_flight_budget"}}, "rate_limit", False),
        (429, {"Retry-After": "20"}, {"code": 429, "message": "Rate limit exceeded: free-models-per-min."}, "rate_limit", False),
        (429, {}, {"code": 429, "message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day"}, "rate_limit", True),
        (404, {}, {"code": 404, "message": "No endpoints found that support tool use."}, "model_unavailable", False),
        (408, {}, {"code": 408, "message": "Your request timed out"}, "server", False),
        (503, {}, {"code": 503, "message": "No available model provider that meets your routing requirements"}, "server", False),
    ],
)
def test_openrouter_error_classification(monkeypatch, status, headers, error, kind, daily):
    recorder = Recorder(lambda request, body: httpx.Response(status, headers=headers, json={"error": error}))
    with pytest.raises(LLMError) as caught:
        openrouter_client(recorder, monkeypatch).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert (caught.value.kind, caught.value.daily, caught.value.provider) == (kind, daily, "openrouter")
    # Its catalogue is not searched for a replacement: one request, one answer.
    assert len(recorder.bodies) == 1


def test_rate_limit_reset_time_is_turned_into_a_wait(monkeypatch):
    reset = int((time.time() + 42) * 1000)
    recorder = Recorder(
        lambda request, body: httpx.Response(
            429, json={"error": {"code": 429, "message": "Rate limit exceeded", "metadata": {"headers": {"X-RateLimit-Reset": str(reset)}}}}
        )
    )
    with pytest.raises(LLMError) as caught:
        openrouter_client(recorder, monkeypatch).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert 38 < caught.value.retry_after <= 42


@pytest.mark.parametrize(
    "message, seconds",
    [
        ("Please try again in 31.7175s.", 31.7175),
        ("Please try again in 7m12s.", 432.0),
        ("Please try again in 450ms.", 0.45),
        ("Please try again in 2m.", 120.0),
        ("Please try again in 1h2m3s.", 3723.0),
        ("Please retry in 36.44s.", 36.44),
        ("No hint here.", None),
    ],
)
def test_wait_times_in_error_messages(message, seconds):
    recorder = Recorder(lambda request, body: httpx.Response(429, json={"error": {"message": message}}))
    with pytest.raises(LLMError) as caught:
        groq_client(recorder).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert caught.value.retry_after == pytest.approx(seconds) if seconds is not None else caught.value.retry_after is None


# --------------------------------------------------------------------------- #
# Tool names and refused replies
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, expected",
    [
        ("write_file", "write_file"),
        ("functions/write_file", "write_file"),  # what openai/gpt-oss-120b sent in production
        ("functions.write_file", "write_file"),
        ("functions:write_file", "write_file"),
        ("default_api.write_file", "write_file"),
        ("Write_File", "write_file"),
        ("write-file", "write_file"),
        ("write_file<|channel|>commentary", "write_file"),
        (" 'write_file' ", "write_file"),
        ("commentary", None),
        ("repo_browser.open_file", None),
        ("functions/", None),
        ("", None),
        (None, None),
    ],
)
def test_canonical_tool_name(name, expected):
    assert canonical_tool_name(name, ["read_file", "write_file", "finish"]) == expected


def test_calls_in_generation():
    assert calls_in_generation('{"name": "functions/write_file", "arguments": {"path": "a", "content": "x"}}') == [
        ("functions/write_file", {"path": "a", "content": "x"})
    ]
    # A raw line break inside a JSON string: the usual cause of "not valid JSON".
    assert calls_in_generation('{"name": "write_file", "arguments": {"path": "a", "content": "line 1\nline 2"}}') == [
        ("write_file", {"path": "a", "content": "line 1\nline 2"})
    ]
    assert calls_in_generation('{"name": "write_file", "arguments": "{\\"path\\": \\"a\\"}"}') == [("write_file", {"path": "a"})]
    assert calls_in_generation('[{"name": "a", "arguments": {}}, {"function": {"name": "b", "parameters": {"x": 1}}}]') == [("a", {}), ("b", {"x": 1})]
    assert calls_in_generation('<function=write_file>{"path": "a"}</function>') == [("write_file", {"path": "a"})]
    for not_a_call in ("", "I will now write the file.", '{"reason": "Tool call arguments are not valid JSON"}', '{"name": "write_file", "arguments": {"path": "a", "content": "cut o', "[1, 2]"):
        assert calls_in_generation(not_a_call) == []


def refusal(message: str, generation: Any) -> httpx.Response:
    return httpx.Response(400, json={"error": {"message": message, "code": "tool_use_failed", "failed_generation": generation}})


def test_reply_refused_over_a_prefixed_tool_name_is_used_as_meant():
    """Production, 2026-11-11: three replies in a row were refused because the
    model wrote 'functions/write_file'. Each cost a round trip against a limit
    of 8,000 tokens per minute. The refused text holds the complete call."""
    recorder = Recorder(
        lambda request, body: refusal(
            "Tool call validation failed: tool call validation failed: attempted to call tool 'functions/write_file' which was not in request.tools",
            '{"name": "functions/write_file", "arguments": {"path": "package.json", "content": "{\\n  \\"name\\": \\"omni-pulse\\"\\n}\\n"}}',
        )
    )
    client = groq_client(recorder)
    response = client.complete(MESSAGES, WRITE_TOOLS, max_output_tokens=100, timeout=30)
    assert response.recovered is True and response.provider == "groq" and len(recorder.bodies) == 1
    call = response.tool_calls[0]
    assert (call.name, call.arguments) == ("write_file", {"path": "package.json", "content": '{\n  "name": "omni-pulse"\n}\n'})
    assert call.id == "call_recovered_1"
    assert response.assistant_message == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_recovered_1", "type": "function", "function": {"name": "write_file", "arguments": json.dumps(call.arguments)}}],
    }
    # Ids stay unique over the conversation.
    assert client.complete(MESSAGES, WRITE_TOOLS, max_output_tokens=100, timeout=30).tool_calls[0].id == "call_recovered_2"


def test_reply_refused_over_a_line_break_in_json_is_used_as_meant():
    recorder = Recorder(
        lambda request, body: refusal(
            "Failed to parse tool call arguments as JSON",
            '{"name": "write_file", "arguments": {"path": "src/App.tsx", "content": "export default function App() {\n  return null;\n}\n"}}',
        )
    )
    response = groq_client(recorder).complete(MESSAGES, WRITE_TOOLS, max_output_tokens=100, timeout=30)
    assert response.recovered and response.tool_calls[0].arguments["content"].count("\n") == 3


@pytest.mark.parametrize(
    "message, generation",
    [
        # A tool that does not exist stays a refusal: there is nothing it could mean.
        ("attempted to call tool 'commentary' which was not in request.tools", '{"name": "commentary", "arguments": {"text": "hi"}}'),
        # So does a call the provider refused for what is in it.
        ("parameters for tool write_file did not match schema: errors: [missing properties: 'content']", '{"name": "write_file", "arguments": {"path": "a"}}'),
        # And text that was cut off.
        ("Failed to parse tool call arguments as JSON", '{"name": "write_file", "arguments": {"path": "a", "content": "cut o'),
        ("attempted to call tool 'functions/write_file' which was not in request.tools", '[{"name": "functions/write_file", "arguments": {"path": "a", "content": ""}}, {"name": "commentary", "arguments": {}}]'),
    ],
)
def test_refusals_that_are_about_the_content_are_not_recovered(message, generation):
    recorder = Recorder(lambda request, body: refusal(message, generation))
    with pytest.raises(LLMError) as caught:
        groq_client(recorder).complete(MESSAGES, WRITE_TOOLS, max_output_tokens=100, timeout=30)
    assert caught.value.kind == "tool_format" and caught.value.failed_generation == generation


def test_prefixed_tool_names_in_an_accepted_reply_are_understood():
    """Providers that do not check tool names (OpenRouter, OpenAI) pass them through."""
    recorder = Recorder(
        lambda request, body: httpx.Response(
            200,
            json=chat_completion(
                "m",
                tool_calls=[
                    {"id": "c1", "type": "function", "function": {"name": "functions.read_file", "arguments": '{"path": "a"}'}},
                    {"id": "c2", "type": "function", "function": {"name": "commentary", "arguments": "{}"}},
                ],
            ),
        )
    )
    response = groq_client(recorder).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert [call.name for call in response.tool_calls] == ["read_file", "commentary"]
    assert response.assistant_message["tool_calls"][0]["function"]["name"] == "read_file"


# --------------------------------------------------------------------------- #
# Choosing providers, and trying a key out
# --------------------------------------------------------------------------- #
def test_every_configured_provider_joins_the_chain(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk")
    # One provider is a chain already: its model, and the one that stands in for it.
    alone = create_llm_client(get_settings())
    assert isinstance(alone, FailoverClient)
    assert alone.chain == ["groq / openai/gpt-oss-120b", "groq / openai/gpt-oss-20b"]
    # A model that is named is the only one used.
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    single = create_llm_client(get_settings())
    assert isinstance(single, OpenAICompatibleClient) and (single.provider, single.model) == ("groq", "openai/gpt-oss-120b")
    monkeypatch.delenv("GROQ_MODEL")

    monkeypatch.setenv("GEMINI_API_KEY", "AQ.key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    pool = create_llm_client(get_settings())
    assert isinstance(pool, FailoverClient)
    # Every provider's first model comes before anybody's stand-in.
    assert pool.chain == [
        "groq / openai/gpt-oss-120b",
        f"gemini / {DEFAULT_GEMINI_MODEL}",
        "openrouter / openrouter/free",
        "groq / openai/gpt-oss-20b",
        "gemini / gemini-3.5-flash-lite",
    ]
    assert (pool.provider, pool.model, pool.model_env_var) == ("groq", "openai/gpt-oss-120b", "GROQ_MODEL")

    monkeypatch.setenv("JULES_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-4-argon")
    assert create_llm_client(get_settings()).chain == [
        "gemini / gemini-4-argon",
        "groq / openai/gpt-oss-120b",
        "openrouter / openrouter/free",
        "groq / openai/gpt-oss-20b",
    ]
    monkeypatch.setenv("JULES_LLM_FALLBACK", "false")
    assert isinstance(create_llm_client(get_settings()), GeminiClient)
    monkeypatch.delenv("JULES_LLM_PROVIDER")
    monkeypatch.delenv("JULES_LLM_FALLBACK")
    monkeypatch.setenv("JULES_LLM_ORDER", "openrouter,gemini")
    assert [label.split(" / ")[0] for label in create_llm_client(get_settings()).chain] == ["openrouter", "gemini", "groq", "groq"]
    # Several models for OpenRouter as well: each is an entry of its own.
    monkeypatch.setenv("OPENROUTER_MODEL", "baidu/ernie-5.5-flash-preview:free, openrouter/free")
    assert create_llm_client(get_settings()).chain == [
        "openrouter / baidu/ernie-5.5-flash-preview:free",
        "gemini / gemini-4-argon",
        "groq / openai/gpt-oss-120b",
        "openrouter / openrouter/free",
        "groq / openai/gpt-oss-20b",
    ]
    monkeypatch.delenv("OPENROUTER_MODEL")
    # The order covers every provider, with or without a key: what the dashboard is told.
    assert get_settings().llm_preference == ["openrouter", "gemini", "groq", "openai", "anthropic"]


def test_google_api_key_is_accepted_as_the_gemini_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AQ.other-name")
    settings = get_settings()
    assert settings.gemini_api_key == "AQ.other-name" and settings.llm_chain == ["gemini"]
    assert "AQ.other-name" in settings.secret_values()


def test_keys_of_one_request_are_used_for_that_request_only(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-server")
    base = get_settings()
    mine = base.with_overrides(
        api_keys={"gemini": " AQ.browser ", "groq": "", "nonsense": "x"},
        models={"gemini": "gemini-4-argon", "openrouter": "not a model name"},
        primary="gemini",
    )
    assert mine.llm_chain == ["gemini", "groq"] and mine.gemini_api_key == "AQ.browser"
    # A model named with the request is the one used; a name that cannot be one changes nothing.
    assert mine.models_for("gemini") == ["gemini-4-argon"]
    assert mine.models_for("openrouter") == ["openrouter/free"]
    assert mine.groq_api_key == "gsk-server"  # an empty value does not erase the server's key
    assert base.llm_chain == ["groq"] and base.gemini_api_key is None  # the server's own settings are untouched
    # Preferring a provider that has no key changes nothing: a preference is not a requirement.
    assert base.with_overrides(primary="gemini").llm_chain == ["groq"]
    assert base.with_overrides(primary="auto", fallback=False).llm_chain == ["groq"]
    assert base.with_overrides() is base


def test_a_request_that_brings_a_key_works_whatever_the_server_prefers(monkeypatch):
    # The server names a first choice it has no key for: on its own it cannot run a task ...
    monkeypatch.setenv("JULES_LLM_PROVIDER", "gemini")
    server = get_settings()
    assert server.llm_chain == [] and server.needs()["llm"] == (
        "an API key for the selected LLM provider (JULES_LLM_PROVIDER=gemini)"
    )
    # ... but a request with a key for another provider is served by that one,
    assert server.with_overrides(api_keys={"openrouter": "sk-or-browser"}).llm_chain == ["openrouter"]
    # and one that brings the key the server wanted gets the server's choice.
    both = server.with_overrides(api_keys={"openrouter": "sk-or-browser", "gemini": "AQ.browser"})
    assert both.llm_chain == ["gemini", "openrouter"] and both.missing_requirements() == server.missing_requirements()[1:]
    # A request that changes something else does not make a key appear,
    assert server.with_overrides(fallback=False).llm_chain == []
    # and it does not talk the server out of its choice either: only a key does that.
    monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "g" * 40)
    with_groq = get_settings()
    assert with_groq.llm_chain == [] and with_groq.with_overrides(fallback=False).llm_chain == []
    assert with_groq.with_overrides(models={"groq": "some-model"}).llm_chain == []
    assert with_groq.with_overrides(api_keys={"openrouter": "sk-or-browser"}).llm_chain == ["groq", "openrouter"]


def probe_transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_probe_reports_a_working_key(monkeypatch):
    seen: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=gemini_answer([{"functionCall": {"name": "confirm", "args": {"ready": True}}, "thoughtSignature": "S"}]))

    settings = get_settings().with_overrides(api_keys={"gemini": "AQ.browser"})
    result = probe_provider(settings, "gemini", transport=probe_transport(handler))
    assert result["ok"] is True and result["tool_call"] is True and result["kind"] is None
    assert result["provider"] == "gemini" and result["model"] == "gemini-flash-latest" and result["latency_ms"] >= 0
    # An alias answers with the model that is behind it at the moment.
    assert result["message"] == (
        "Google Gemini accepted the key and gemini-flash-latest (answered by gemini-3.8-flash) called the test tool."
    )
    assert result["answered_by"] == "gemini-3.8-flash"
    # The same kind of request the agent makes: with a tool to call.
    assert seen[0]["tools"][0]["functionDeclarations"][0]["name"] == "confirm"
    # Every model the provider is given was tried: a task may end up on any of them.
    assert [item["model"] for item in result["models"]] == ["gemini-flash-latest", "gemini-3.5-flash-lite"]
    assert all(item["ok"] and item["tool_call"] for item in result["models"]) and len(seen) == 2


def test_probe_tries_every_model_and_says_which_one_fails(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "gone:free":
            return httpx.Response(404, json={"error": {"code": 404, "message": "No endpoints found for gone:free."}})
        answer = chat_completion("upstage/solar-pro-3:free", tool_calls=[{"id": "c", "type": "function", "function": {"name": "confirm", "arguments": "{}"}}])
        return httpx.Response(200, json=answer)

    settings = get_settings().with_overrides(api_keys={"openrouter": "sk-or-browser"}, models={"openrouter": "gone:free, openrouter/free"})
    result = probe_provider(settings, "openrouter", transport=probe_transport(handler))
    # The result is about the first model; the list says what each one did.
    assert result["ok"] is False and result["kind"] == "model_unavailable" and result["model"] == "gone:free"
    first, second = result["models"]
    assert (first["model"], first["ok"], first["kind"]) == ("gone:free", False, "model_unavailable")
    assert (second["model"], second["ok"], second["tool_call"]) == ("openrouter/free", True, True)
    # The router answers with a model of its choice.
    assert second["answered_by"] == "upstage/solar-pro-3:free"
    assert second["message"] == "OpenRouter accepted the key and openrouter/free (answered by upstage/solar-pro-3:free) called the test tool."


def test_probe_stops_at_a_rejected_key(monkeypatch):
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return gemini_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")

    settings = get_settings().with_overrides(api_keys={"gemini": "AQ.wrong"})
    result = probe_provider(settings, "gemini", transport=probe_transport(handler))
    # The other model would say the same about the key: it is not asked.
    assert result["kind"] == "auth" and len(result["models"]) == 1 and len(calls) == 1


def test_probe_describes_what_is_wrong(monkeypatch):
    settings = get_settings().with_overrides(api_keys={"gemini": "AQ.wrong", "openrouter": "sk-or"})
    bad_key = probe_provider(
        settings,
        "gemini",
        transport=probe_transport(lambda request: gemini_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")),
    )
    assert bad_key["ok"] is False and bad_key["kind"] == "auth" and "API key not valid" in bad_key["message"]
    assert "AQ.wrong" not in json.dumps(bad_key)

    text_only = probe_provider(settings, "openrouter", transport=probe_transport(lambda request: httpx.Response(200, json=chat_completion("m", content="Ready!"))))
    assert text_only["ok"] is True and text_only["tool_call"] is False
    assert "answered in text instead of calling the test tool" in text_only["message"]

    assert probe_provider(settings, "groq")["kind"] == "no_key"
    assert probe_provider(settings, "skynet")["kind"] == "unknown_provider"


# --------------------------------------------------------------------------- #
# Several models per provider
# --------------------------------------------------------------------------- #
def test_a_model_setting_can_name_several_models():
    assert split_models("openai/gpt-oss-120b") == ["openai/gpt-oss-120b"]
    assert split_models(" a:free , openrouter/free;b\nc ") == ["a:free", "openrouter/free", "b", "c"]
    # Not names, repeats and more than a provider may have are dropped.
    assert split_models("a, not a name, a, <script>, b") == ["a", "b"]
    assert split_models("a,b,c,d,e,f") == ["a", "b", "c", "d"]
    assert split_models(["a", 3, None, "b"]) == ["a", "b"] and split_models(None) == [] and split_models("") == []


def test_defaults_give_every_provider_a_model_that_stands_in(monkeypatch):
    settings = get_settings()
    assert settings.models_for("groq") == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    # Google's alias for its current Flash model, then the lighter one.
    assert settings.models_for("gemini") == ["gemini-flash-latest", "gemini-3.5-flash-lite"]
    # OpenRouter's router over whatever free model is available: a pool by itself.
    assert settings.models_for("openrouter") == ["openrouter/free"]
    assert settings.model_for("openrouter") == DEFAULT_OPENROUTER_MODEL and settings.model_for("nobody") is None
    # Nothing that is used unasked is a version Google has announced the end of.
    assert not any("2.5" in model or "2.0" in model for model in settings.models_for("gemini"))
    # What is configured replaces the list - one model or several.
    monkeypatch.setenv("OPENROUTER_MODEL", "baidu/ernie-5.5-flash-preview:free,openrouter/free")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash, gemini-3.6-flash ,gemini-3.5-flash-lite")
    settings = get_settings()
    assert settings.models_for("openrouter") == ["baidu/ernie-5.5-flash-preview:free", "openrouter/free"]
    assert settings.models_for("gemini") == ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
    # A setting nothing can be made of falls back to the defaults.
    monkeypatch.setenv("GEMINI_MODEL", "not a model")
    assert get_settings().models_for("gemini") == ["gemini-flash-latest", "gemini-3.5-flash-lite"]


def test_the_order_in_which_models_are_tried(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setenv("GROQ_API_KEY", "gsk")
    monkeypatch.setenv("GEMINI_API_KEY", "AQ.key")
    monkeypatch.setenv("JULES_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    settings = get_settings()
    # The providers of the run of 2026-11-12, in its order: the router instead of
    # one free model, and what stands in behind Groq's and Gemini's models.
    assert settings.llm_chain == ["openrouter", "groq", "gemini"]
    assert settings.llm_slots == [
        ("openrouter", "openrouter/free"),
        ("groq", "openai/gpt-oss-120b"),
        ("gemini", "gemini-flash-latest"),
        ("gemini", "gemini-3.5-flash-lite"),
    ]
    # Without fallback there is one provider and one model.
    assert settings.with_overrides(fallback=False).llm_slots == [("openrouter", "openrouter/free")]
    # The browser's own list for a provider replaces the server's.
    mine = settings.with_overrides(
        models={"openrouter": "baidu/ernie-5.5-flash-preview:free, openrouter/free", "gemini": "gemini-3.8-flash,gemini-3.6-flash"}
    )
    assert mine.llm_slots == [
        ("openrouter", "baidu/ernie-5.5-flash-preview:free"),
        ("groq", "openai/gpt-oss-120b"),
        ("gemini", "gemini-3.8-flash"),
        ("openrouter", "openrouter/free"),
        ("gemini", "gemini-3.6-flash"),
    ]
    assert get_settings().with_overrides(api_keys={}).llm_slots == settings.llm_slots
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert get_settings().llm_slots == []


def test_each_model_of_a_provider_gets_its_own_client(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setenv("OPENROUTER_MODEL", "baidu/ernie-5.5-flash-preview:free, openrouter/free")
    models: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        models.append(body["model"])
        if body["model"] != "openrouter/free":
            return httpx.Response(429, json={"error": {"code": 429, "message": "Provider returned error", "metadata": {"raw": "baidu/ernie-5.5-flash-preview:free is temporarily rate-limited upstream."}}})
        return httpx.Response(200, json=chat_completion("google/gemma-4-31b-it:free", content="hello"))

    pool = create_llm_client(get_settings(), transport=httpx.MockTransport(handler))
    assert pool.chain == ["openrouter / baidu/ernie-5.5-flash-preview:free", "openrouter / openrouter/free"]
    with pytest.raises(LLMError) as caught:
        pool.complete([{"role": "user", "content": "hi"}], [], max_output_tokens=64, timeout=5)
    # The model is limited upstream; OpenRouter's free router is asked instead, at once.
    assert caught.value.kind == "rate_limit" and caught.value.switched_to == "openrouter / openrouter/free"
    answer = pool.complete([{"role": "user", "content": "hi"}], [], max_output_tokens=64, timeout=5)
    assert answer.content == "hello" and answer.model == "google/gemma-4-31b-it:free"
    assert models == ["baidu/ernie-5.5-flash-preview:free", "openrouter/free"]
    assert pool.usage()[1]["served"] == {"google/gemma-4-31b-it:free": 1}


def test_openrouter_alone_is_its_router(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    client = create_llm_client(get_settings())
    # One entry: the router is a pool of models by itself.
    assert isinstance(client, OpenAICompatibleClient) and (client.provider, client.model) == ("openrouter", "openrouter/free")
