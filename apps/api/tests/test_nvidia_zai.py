"""NVIDIA NIM and Z.AI (GLM): two more providers behind the same chain.

Both speak OpenAI's chat-completions format; what differs is what they accept,
how they report failures and how they count their limits. Everything runs
against mocked HTTP transports that answer the way the providers document.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.api.config import DEFAULT_MODELS, LLM_PROVIDERS, NVIDIA_MODEL_PREFERENCES, get_settings
from apps.api.services.llm import (
    NVIDIA_RATE_LIMIT_COOLDOWN,
    FailoverClient,
    LLMError,
    OpenAICompatibleClient,
    build_provider_client,
    create_llm_client,
    probe_provider,
)

from .conftest import CHECK_ADD
from .support import FakeGitHub
from .test_api import TASK, messages, run_stream
from .test_fallback import Clock, failure, provider, rate_limit
from .test_llm import MESSAGES, TOOLS, Recorder, chat_completion

READ_CALL = [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]
NOT_FOR_ACCOUNT = {
    "status": 404,
    "title": "Not Found",
    "detail": "Function '0f1e2d3c-aaaa-4bbb-8ccc-0123456789ab': Not found for account 'acct-123'",
}
TOO_MANY = {"status": 429, "title": "Too Many Requests"}


def nvidia_client(recorder: Recorder, monkeypatch, **env: str) -> OpenAICompatibleClient:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    client = build_provider_client(get_settings(), "nvidia", transport=recorder.transport)
    assert isinstance(client, OpenAICompatibleClient)
    return client


def zai_client(recorder: Recorder, monkeypatch, model: Optional[str] = None, **env: str) -> OpenAICompatibleClient:
    monkeypatch.setenv("ZAI_API_KEY", "zai-test")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    client = build_provider_client(get_settings(), "zai", model=model, transport=recorder.transport)
    assert isinstance(client, OpenAICompatibleClient)
    return client


def zai_error(status: int, code: str, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": message}})


# --------------------------------------------------------------------------- #
# NVIDIA NIM
# --------------------------------------------------------------------------- #
def test_nvidia_request(monkeypatch):
    recorder = Recorder(lambda request, body: httpx.Response(200, json=chat_completion(body["model"], tool_calls=READ_CALL)))
    client = nvidia_client(recorder, monkeypatch)
    response = client.complete(MESSAGES, TOOLS, max_output_tokens=900, timeout=30)
    request, body = recorder.requests[0], recorder.bodies[0]
    assert str(request.url) == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer nvapi-test"
    assert body["model"] == "deepseek-ai/deepseek-v4.1-flash" == DEFAULT_MODELS["nvidia"][0]
    assert body["max_tokens"] == 900 and "max_completion_tokens" not in body
    assert body["tools"] == TOOLS and body["tool_choice"] == "auto" and body["temperature"] == 0.2
    # Its models differ in what they take: nothing about reasoning is sent unless it is set.
    assert "reasoning_effort" not in body and "thinking" not in body
    assert (response.provider, response.model) == ("nvidia", "deepseek-ai/deepseek-v4.1-flash")
    assert response.tool_calls[0].name == "read_file" and response.tool_calls[0].arguments == {"path": "a.py"}
    # No tokens-per-minute limit is reported: the context is sized for a large window.
    assert response.tokens_per_minute_limit is None and client.assumed_tokens_per_minute == 200_000
    assert client.model_env_var == "NVIDIA_MODEL" and client.limits_per_account is True


def test_nvidia_settings(monkeypatch):
    recorder = Recorder(lambda request, body: httpx.Response(200, json=chat_completion(body["model"], content="ok")))
    client = nvidia_client(
        recorder,
        monkeypatch,
        NVIDIA_MODEL="moonshotai/kimi-k2.6",
        NVIDIA_REASONING_EFFORT="low",
        NVIDIA_TEMPERATURE="default",
        NVIDIA_BASE_URL="https://nim.example.test/v1/",
    )
    client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    body = recorder.bodies[0]
    assert str(recorder.requests[0].url) == "https://nim.example.test/v1/chat/completions"
    assert body["model"] == "moonshotai/kimi-k2.6" and body["reasoning_effort"] == "low" and "temperature" not in body


@pytest.mark.parametrize(
    "setting, sent",
    [
        # What the page of one of its models documents, for every request.
        ('{"chat_template_kwargs": {"enable_thinking": false}, "top_p": 0.9}', {"chat_template_kwargs": {"enable_thinking": False}, "top_p": 0.9}),
        # A setting does not get to replace what a request is made of.
        ('{"stream": true, "model": "other", "messages": [], "seed": 7}', {"seed": 7}),
        ("enable_thinking=false", {}),
        ('["not", "an", "object"]', {}),
    ],
)
def test_more_fields_for_nvidia_requests(monkeypatch, setting, sent):
    recorder = Recorder(lambda request, body: httpx.Response(200, json=chat_completion(body["model"], content="ok")))
    nvidia_client(recorder, monkeypatch, NVIDIA_EXTRA_BODY=setting).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    body = recorder.bodies[0]
    usual = {"model", "messages", "max_tokens", "temperature", "tools", "tool_choice"}
    assert {name: value for name, value in body.items() if name not in usual} == sent
    assert body["model"] == "deepseek-ai/deepseek-v4.1-flash" and body["messages"] and "stream" not in body


def test_a_parameter_a_model_does_not_take_is_left_out(monkeypatch):
    """Models behind one endpoint differ; the one that rejects a parameter by name is asked without it."""

    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        if "reasoning_effort" in body:
            return httpx.Response(
                422,
                json={"detail": [{"type": "extra_forbidden", "loc": ["body", "reasoning_effort"], "msg": "Extra inputs are not permitted: reasoning_effort"}]},
            )
        return httpx.Response(200, json=chat_completion(body["model"], content="ok"))

    recorder = Recorder(handler)
    client = nvidia_client(recorder, monkeypatch, NVIDIA_REASONING_EFFORT="low")
    assert client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content == "ok"
    assert ["reasoning_effort" in body for body in recorder.bodies] == [True, False]
    client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert len(recorder.bodies) == 3  # it is not sent again


@pytest.mark.parametrize(
    "status, body, kind, message",
    [
        (
            401,
            {"status": 401, "title": "Unauthorized", "detail": "Authentication failed"},
            "auth",
            "nvidia returned HTTP 401: Unauthorized: Authentication failed Check the NVIDIA API key.",
        ),
        (403, {"status": 403, "title": "Forbidden", "detail": "Authorization failed"}, "auth", None),
        (
            402,
            {"status": 402, "title": "Payment Required", "detail": "Account 'acct-123': Cloud credits expired - Please contact NVIDIA representatives"},
            "quota",
            None,
        ),
        (429, TOO_MANY, "rate_limit", "nvidia returned HTTP 429: Too Many Requests"),
        (500, {"status": 500, "title": "Internal Server Error"}, "server", "nvidia returned HTTP 500: Internal Server Error"),
        (504, "upstream request timeout", "server", "nvidia returned HTTP 504: upstream request timeout"),
        # A model server behind the gateway, in its own words.
        (
            400,
            {"object": "error", "message": "tools is not supported by this model", "type": "BadRequestError", "param": None, "code": 400},
            "bad_request",
            "nvidia returned HTTP 400: tools is not supported by this model",
        ),
        (
            422,
            {"detail": [{"type": "missing", "loc": ["body", "messages"], "msg": "Field required"}]},
            "bad_request",
            "nvidia returned HTTP 422: Field required",
        ),
    ],
)
def test_nvidia_error_classification(monkeypatch, status, body, kind, message):
    answer = httpx.Response(status, text=body) if isinstance(body, str) else httpx.Response(status, json=body)
    recorder = Recorder(lambda request, _body: answer)
    with pytest.raises(LLMError) as caught:
        nvidia_client(recorder, monkeypatch).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    error = caught.value
    assert (error.kind, error.daily, error.provider, error.status_code) == (kind, False, "nvidia", status)
    if message:
        assert error.message == message
    assert len(recorder.bodies) == 1


def test_nvidia_model_that_was_withdrawn_is_replaced_by_one_it_still_lists(monkeypatch):
    """NVIDIA adds and withdraws models every few weeks. A default that is gone
    must not need a new release: another model known to call tools takes over."""
    notices: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.url.path == "/v1/models"
            listed = ["meta/llama2-70b", "nvidia/nemotron-3-super-120b-a12b", "moonshotai/kimi-k2.6", "nvidia/nemotron-3-ultra-550b-a55b"]
            return httpx.Response(200, json={"object": "list", "data": [{"id": name, "object": "model"} for name in listed]})
        body = json.loads(request.content)
        if body["model"] == "deepseek-ai/deepseek-v4.1-flash":
            return httpx.Response(404, json=NOT_FOR_ACCOUNT)
        return httpx.Response(200, json=chat_completion(body["model"], tool_calls=READ_CALL))

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    client = build_provider_client(get_settings(), "nvidia", transport=httpx.MockTransport(handler), on_notice=notices.append)
    response = client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    # Not Nemotron 3 Super: that is the chain's own second entry for NVIDIA. The next one known to it.
    assert response.model == client.model == "nvidia/nemotron-3-ultra-550b-a55b"
    assert len(notices) == 1 and notices[0].startswith("Model 'deepseek-ai/deepseek-v4.1-flash' is not available on nvidia")
    assert notices[0].endswith("using 'nvidia/nemotron-3-ultra-550b-a55b' instead. Set NVIDIA_MODEL to silence this.")
    # The second entry, for its part, would not pick the first one's model.
    second = build_provider_client(get_settings(), "nvidia", model="nvidia/nemotron-3-super-120b-a12b")
    assert "deepseek-ai/deepseek-v4.1-flash" not in second._preferences and second._preferences[0] == "nvidia/nemotron-3-super-120b-a12b"
    assert set(NVIDIA_MODEL_PREFERENCES) >= set(DEFAULT_MODELS["nvidia"])


def test_trying_a_key_out_says_when_a_model_was_replaced(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": name} for name in NVIDIA_MODEL_PREFERENCES]})
        body = json.loads(request.content)
        if body["model"] == "nvidia/nemotron-3-super-120b-a12b":
            return httpx.Response(404, json=NOT_FOR_ACCOUNT)
        call = [{"id": "c", "type": "function", "function": {"name": "confirm", "arguments": '{"ready": true}'}}]
        return httpx.Response(200, json=chat_completion(body["model"], tool_calls=call))

    settings = get_settings().with_overrides(api_keys={"nvidia": "nvapi-browser"})
    result = probe_provider(settings, "nvidia", transport=httpx.MockTransport(handler))
    first, second = result["models"]
    assert (first["model"], first["ok"], first["tool_call"]) == ("deepseek-ai/deepseek-v4.1-flash", True, True) and "requested" not in first
    # The second one is gone: what answered instead, and what had been asked for.
    assert (second["model"], second["requested"]) == ("nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b")
    assert second["ok"] is True and second["tool_call"] is True
    assert "Model 'nvidia/nemotron-3-super-120b-a12b' is not available on nvidia" in second["message"]
    assert "nvapi-browser" not in json.dumps(result)


def test_nvidia_account_without_access_to_the_hosted_endpoints_is_explained(monkeypatch):
    """Such an account can list the models and call none: every model answers 404."""
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": name} for name in NVIDIA_MODEL_PREFERENCES]})
        return httpx.Response(404, json=NOT_FOR_ACCOUNT)

    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    client = build_provider_client(get_settings(), "nvidia", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError) as caught:
        client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    error = caught.value
    assert error.kind == "model_unavailable" and error.status_code == 404
    assert error.message == (
        "nvidia returned HTTP 404: Not Found: Function '0f1e2d3c-aaaa-4bbb-8ccc-0123456789ab': Not found for account "
        "'acct-123' NVIDIA answers this for a model it has withdrawn - and for every model when the account has not "
        "been given access to its hosted endpoints."
    )
    # One replacement is tried, not the whole catalogue.
    assert calls == ["POST /v1/chat/completions", "GET /v1/models", "POST /v1/chat/completions"]


def test_a_rate_limit_at_nvidia_sets_all_of_its_models_aside():
    """NVIDIA counts requests per account, and asking again while limited
    prolongs the limit: its second model is no way round it."""
    clock = Clock()
    first = provider("nvidia", ["one", rate_limit("nvidia returned HTTP 429: Too Many Requests", None), "again"], model="deepseek")
    second = provider("nvidia", ["never asked while the account is limited"], model="nemotron")
    for client in (first, second):
        client.limits_per_account, client.rate_limit_cooldown = True, NVIDIA_RATE_LIMIT_COOLDOWN
    zai = provider("zai", ["two", rate_limit("zai returned HTTP 429: Rate limit reached for requests", None)], model="glm")
    pool = FailoverClient([first, zai, second], clock=clock)
    assert pool.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "one"
    error = failure(pool)
    assert error.switched_to == "zai / glm" and pool.provider == "zai"
    assert pool.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "two"
    # Z.AI is limited in turn. Nobody is ready: the shortest wait is NVIDIA's, a
    # minute from its refusal - for either of its models, not just the one that was asked.
    clock.sleep(20.0)
    waiting = failure(pool)
    assert waiting.switched_to is None and waiting.all_limited is True
    assert waiting.retry_after == pytest.approx(30.0) and waiting.next_provider is None  # Z.AI's own 30 s
    clock.sleep(30.0)
    assert second.requests == [] and len(first.requests) == 2


def test_nvidia_alone_waits_out_its_limit_and_then_asks_its_other_model():
    """With nobody else to turn to, the minute is waited. Should the limit be
    about that one model after all, the other one finds out."""
    clock = Clock()
    limited = rate_limit("nvidia returned HTTP 429: Too Many Requests", None)
    first = provider("nvidia", [limited, "first again"], model="deepseek")
    second = provider("nvidia", [limited, "never reached"], model="nemotron")
    for client in (first, second):
        client.limits_per_account, client.rate_limit_cooldown = True, NVIDIA_RATE_LIMIT_COOLDOWN
    pool = FailoverClient([first, second], clock=clock)
    error = failure(pool)
    assert error.switched_to is None and error.retry_after == pytest.approx(NVIDIA_RATE_LIMIT_COOLDOWN)
    assert error.next_provider == "nvidia / nemotron" and second.requests == []
    clock.sleep(NVIDIA_RATE_LIMIT_COOLDOWN)
    # The other model is limited as well: the account it is, then. Back to the first after another minute.
    again = failure(pool)
    assert again.retry_after == pytest.approx(NVIDIA_RATE_LIMIT_COOLDOWN) and again.next_provider == "nvidia / deepseek"
    clock.sleep(NVIDIA_RATE_LIMIT_COOLDOWN)
    assert pool.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "first again"


def test_other_providers_limits_stay_per_model():
    clock = Clock()
    first = provider("groq", [rate_limit("slow down", 20.0)], model="big")
    second = provider("groq", ["its own allowance"], model="small")
    pool = FailoverClient([first, second], clock=clock)
    assert failure(pool).switched_to == "groq / small"
    assert pool.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "its own allowance"


# --------------------------------------------------------------------------- #
# Z.AI (GLM)
# --------------------------------------------------------------------------- #
def test_zai_request(monkeypatch):
    # Z.AI documents the arguments of a call as an object, not as the JSON text OpenAI sends.
    call = [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a.py"}}}]
    recorder = Recorder(
        lambda request, body: httpx.Response(
            200, json=chat_completion(body["model"], tool_calls=call, reasoning_content="thinking, not part of the answer")
        )
    )
    client = zai_client(recorder, monkeypatch)
    response = client.complete(MESSAGES, TOOLS, max_output_tokens=900, timeout=30)
    request, body = recorder.requests[0], recorder.bodies[0]
    assert str(request.url) == "https://api.z.ai/api/paas/v4/chat/completions"
    assert request.headers["authorization"] == "Bearer zai-test" and request.headers["accept-language"] == "en-US,en"
    assert body["model"] == "glm-4.7-flash" == DEFAULT_MODELS["zai"][0]
    assert body["max_tokens"] == 900 and "max_completion_tokens" not in body
    assert body["tool_choice"] == "auto" and body["temperature"] == 0.2
    # Thinking before every reply is more than a run of a few minutes can afford.
    assert body["thinking"] == {"type": "disabled"} and "reasoning_effort" not in body
    assert response.provider == "zai" and response.tool_calls[0].arguments == {"path": "a.py"}
    # What goes back into the conversation is in the common format, without the thinking.
    kept = response.assistant_message
    assert kept["tool_calls"][0]["function"]["arguments"] == '{"path": "a.py"}' and "reasoning_content" not in kept
    assert client.assumed_tokens_per_minute == 200_000 and client.limits_per_account is False


@pytest.mark.parametrize(
    "model, setting, expected",
    [
        ("glm-4.7-flash", None, {"thinking": {"type": "disabled"}}),
        ("glm-4.5-flash", "none", {"thinking": {"type": "disabled"}}),
        ("glm-5.2", "off", {"thinking": {"type": "disabled"}}),
        # GLM-5.3 and later cannot switch thinking off: the lowest effort, then.
        ("glm-5.3", None, {"reasoning_effort": "low"}),
        ("glm-5.3-flash", "none", {"reasoning_effort": "low"}),
        ("glm-6", None, {"reasoning_effort": "low"}),
        ("glm-4.7-flash", "high", {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}),
        ("glm-5.2", "medium", {"thinking": {"type": "enabled"}, "reasoning_effort": "medium"}),
        # ... which knows three levels.
        ("glm-5.3", "medium", {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}),
        ("glm-5.3", "MAX", {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}),
        ("glm-4.7-flash", "a lot", {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}),
        ("glm-5.3", "default", {}),
        ("glm-4.7-flash", "default", {}),
    ],
)
def test_how_much_a_glm_model_thinks(monkeypatch, model, setting, expected):
    recorder = Recorder(lambda request, body: httpx.Response(200, json=chat_completion(body["model"], content="ok")))
    env = {"ZAI_REASONING_EFFORT": setting} if setting else {}
    zai_client(recorder, monkeypatch, model=model, **env).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    body = recorder.bodies[0]
    assert {name: body[name] for name in ("thinking", "reasoning_effort") if name in body} == expected


def test_zai_settings(monkeypatch):
    recorder = Recorder(lambda request, body: httpx.Response(200, json=chat_completion(body["model"], content="ok")))
    client = zai_client(
        recorder,
        monkeypatch,
        ZAI_MODEL="glm-5.3, glm-4.7-flash",
        ZAI_TEMPERATURE="1.6",
        ZAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4",
    )
    client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert str(recorder.requests[0].url) == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    # Z.AI takes temperatures from 0 to 1.
    assert recorder.bodies[0]["model"] == "glm-5.3" and recorder.bodies[0]["temperature"] == 1.0
    monkeypatch.setenv("ZAI_TEMPERATURE", "default")
    other = zai_client(recorder, monkeypatch)
    other.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert "temperature" not in recorder.bodies[-1]


def test_a_parameter_zai_rejects_is_left_out(monkeypatch):
    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        if "thinking" in body:
            return zai_error(400, "1214", "Parameter `thinking` is invalid. Please check the documentation.")
        return httpx.Response(200, json=chat_completion(body["model"], content="ok"))

    recorder = Recorder(handler)
    assert zai_client(recorder, monkeypatch).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content == "ok"
    assert ["thinking" in body for body in recorder.bodies] == [True, False]


@pytest.mark.parametrize(
    "status, code, text, kind, daily, model_only",
    [
        (401, "1000", "Authentication Failed", "auth", False, False),
        (401, "1001", "Authentication parameter not received in Header, unable to authenticate", "auth", False, False),
        (401, "1003", "Authentication Token expired, please regenerate/obtain", "auth", False, False),
        # Arrives as a 429, and is no rate limit: a paid model on an account without a balance.
        (429, "1113", "Insufficient balance or no resource package. Please recharge.", "quota", False, True),
        (400, "1211", "Unknown Model, please check the model code.", "model_unavailable", False, False),
        (400, "1212", "Current model does not support `tools` call method", "model_unavailable", False, False),
        (403, "1220", "You do not have permission to access glm-5.3", "model_unavailable", False, False),
        (429, "1311", "Your current subscription plan does not yet include access to glm-5.3", "model_unavailable", False, False),
        (400, "1261", "Prompt too long", "too_large", False, False),
        (400, "1301", "System detected potentially unsafe or sensitive content in input or generation.", "blocked", False, False),
        (429, "1302", "Rate limit reached for requests", "rate_limit", False, False),
        (429, "1305", "The service may be temporarily overloaded, please try again later", "server", False, False),
        # Allowances that are used up until a time hours or days away: nothing to wait for within a run.
        (429, "1308", "Usage limit reached for 5 hour. Your limit will reset at 2026-11-13 02:00:00", "rate_limit", True, False),
        (429, "1310", "Weekly/Monthly Limit Exhausted. Your limit will reset at 2026-11-16 00:00:00", "rate_limit", True, False),
        (429, "1316", "Usage limit reached for the past 5 hours. Insufficient balance for extra usage. Resets at 2026-11-13 02:00:00.", "rate_limit", True, False),
        (429, "1321", "Usage limit reached for the past 7 days. Extra usage is not available due to monthly spend limit.", "rate_limit", True, False),
        # The subscription itself: no model of this key will answer.
        (429, "1309", "Your GLM Coding Plan package has expired and is temporarily unavailable.", "quota", False, False),
        (429, "1313", "Your account's current usage pattern does not comply with the Fair Usage Policy", "quota", False, False),
        (429, "1315", "This API Key is limited to enterprise coding package scenarios.", "quota", False, False),
        (400, "1210", "Invalid API parameter, please check the documentation.", "bad_request", False, False),
        (500, "1234", "Network error, error id: 77, please try again later", "server", False, False),
    ],
)
def test_zai_error_classification(monkeypatch, status, code, text, kind, daily, model_only):
    recorder = Recorder(lambda request, body: zai_error(status, code, text))
    with pytest.raises(LLMError) as caught:
        zai_client(recorder, monkeypatch).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    error = caught.value
    assert (error.kind, error.daily, error.model_only, error.provider) == (kind, daily, model_only, "zai")
    assert error.status_code == status and text in error.message
    # Its catalogue is not searched for a replacement: one request, one answer.
    assert len(recorder.bodies) == 1


def test_zai_errors_are_understood_in_either_of_their_documented_forms(monkeypatch):
    """Z.AI's error page nests code and message under "error"; its API schema has them at the top."""
    flat = Recorder(lambda request, body: httpx.Response(429, json={"code": 1113, "message": "Insufficient balance or no resource package. Please recharge."}))
    with pytest.raises(LLMError) as caught:
        zai_client(flat, monkeypatch).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert (caught.value.kind, caught.value.model_only) == ("quota", True)
    assert caught.value.message.startswith("zai returned HTTP 429: Insufficient balance or no resource package.")
    numeric = Recorder(lambda request, body: httpx.Response(400, json={"error": {"code": 1211, "message": "Unknown Model, please check the model code."}}))
    with pytest.raises(LLMError) as caught:
        zai_client(numeric, monkeypatch).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert caught.value.kind == "model_unavailable"


def test_zai_says_what_having_no_balance_means(monkeypatch):
    recorder = Recorder(lambda request, body: zai_error(429, "1113", "Insufficient balance or no resource package. Please recharge."))
    with pytest.raises(LLMError) as caught:
        zai_client(recorder, monkeypatch, model="glm-5.3").complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert caught.value.message == (
        "zai returned HTTP 429: Insufficient balance or no resource package. Please recharge. That model is paid for "
        "from the account's balance; Z.AI's free Flash models do not need one."
    )


@pytest.mark.parametrize(
    "finish_reason, kind",
    [("sensitive", "blocked"), ("network_error", "server"), ("model_context_window_exceeded", "too_large")],
)
def test_a_reply_zai_stopped_is_a_failure_not_an_empty_answer(monkeypatch, finish_reason, kind):
    def answer(content: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={"model": "glm-4.7-flash", "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}]},
        )

    recorder = Recorder(lambda request, body: answer(""))
    with pytest.raises(LLMError) as caught:
        zai_client(recorder, monkeypatch).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert caught.value.kind == kind and f"finish_reason {finish_reason}" in caught.value.message
    # What did arrive is passed on as it is.
    partial = Recorder(lambda request, body: answer("So far, so"))
    response = zai_client(partial, monkeypatch).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert response.content == "So far, so" and response.finish_reason == finish_reason


def test_a_paid_glm_model_without_a_balance_leaves_the_free_one_in_use(monkeypatch):
    """Z.AI's larger models are paid for from the account's balance, its Flash
    models are free: an empty account is no reason to give up on the provider."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "glm-5.3":
            return zai_error(429, "1113", "Insufficient balance or no resource package. Please recharge.")
        return httpx.Response(200, json=chat_completion(body["model"], content="free of charge"))

    monkeypatch.setenv("ZAI_API_KEY", "zai-test")
    monkeypatch.setenv("ZAI_MODEL", "glm-5.3, glm-4.7-flash")
    pool = create_llm_client(get_settings(), transport=httpx.MockTransport(handler))
    assert isinstance(pool, FailoverClient) and pool.chain == ["zai / glm-5.3", "zai / glm-4.7-flash"]
    error = failure(pool)
    assert (error.kind, error.model_only, error.switched_to) == ("quota", True, "zai / glm-4.7-flash")
    assert pool.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "free of charge"
    assert [(item["model"], item["out"]) for item in pool.usage()] == [
        ("glm-5.3", "its account has no credit for this model"),
        ("glm-4.7-flash", None),
    ]


def test_a_subscription_that_has_expired_takes_out_every_glm_model():
    expired = LLMError("zai returned HTTP 429: Your GLM Coding Plan package has expired", kind="quota", status_code=429)
    first, second = provider("zai", [expired], model="glm-5.3"), provider("zai", ["not asked"], model="glm-4.7-flash")
    groq = provider("groq", ["carries on"])
    pool = FailoverClient([first, second, groq], clock=Clock())
    assert failure(pool).switched_to == "groq / groq-model"
    assert [item["out"] for item in pool.usage()] == ["its account has no credit left", "its account has no credit left", None]
    assert second.requests == []


def test_trying_a_key_out_goes_on_to_the_free_model(monkeypatch):
    asked: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(body["model"])
        if body["model"] == "glm-5.3":
            return zai_error(429, "1113", "Insufficient balance or no resource package. Please recharge.")
        call = [{"id": "c", "type": "function", "function": {"name": "confirm", "arguments": {"ready": True}}}]
        return httpx.Response(200, json=chat_completion(body["model"], tool_calls=call))

    settings = get_settings().with_overrides(api_keys={"zai": "zai-browser"}, models={"zai": "glm-5.3, glm-4.7-flash"})
    result = probe_provider(settings, "zai", transport=httpx.MockTransport(handler))
    assert asked == ["glm-5.3", "glm-4.7-flash"]
    first, second = result["models"]
    assert (first["model"], first["ok"], first["kind"]) == ("glm-5.3", False, "quota")
    assert (second["model"], second["ok"], second["tool_call"]) == ("glm-4.7-flash", True, True)
    assert second["message"] == "Z.AI (GLM) accepted the key and glm-4.7-flash called the test tool."
    # How the answer was arrived at is not part of it.
    assert "model_only" not in result and all("model_only" not in item for item in result["models"])
    # A key that is rejected still ends the test at once.
    rejected = probe_provider(settings, "zai", transport=httpx.MockTransport(lambda request: zai_error(401, "1000", "Authentication Failed")))
    assert rejected["kind"] == "auth" and len(rejected["models"]) == 1


# --------------------------------------------------------------------------- #
# Their place among the others
# --------------------------------------------------------------------------- #
def test_where_the_two_stand_in_the_chain(monkeypatch):
    assert LLM_PROVIDERS == ("groq", "gemini", "nvidia", "zai", "openrouter", "openai", "anthropic")
    for name, key in (("GROQ", "gsk"), ("GEMINI", "AQ.key"), ("OPENROUTER", "sk-or"), ("NVIDIA", "nvapi-key"), ("ZAI", "zai-key")):
        monkeypatch.setenv(f"{name}_API_KEY", key)
    # Every provider's first model, then the ones that stand in for them. The two
    # that are asked for a model of our choosing come before OpenRouter's router.
    assert create_llm_client(get_settings()).chain == [
        "groq / openai/gpt-oss-120b",
        "gemini / gemini-flash-latest",
        "nvidia / deepseek-ai/deepseek-v4.1-flash",
        "zai / glm-4.7-flash",
        "openrouter / openrouter/free",
        "groq / openai/gpt-oss-20b",
        "gemini / gemini-3.5-flash-lite",
        "nvidia / nvidia/nemotron-3-super-120b-a12b",
        "zai / glm-4.5-flash",
    ]
    monkeypatch.setenv("JULES_LLM_ORDER", "zai, nvidia")
    assert get_settings().llm_chain == ["zai", "nvidia", "groq", "gemini", "openrouter"]
    monkeypatch.setenv("JULES_LLM_PROVIDER", "nvidia")
    assert get_settings().llm_chain[:2] == ["nvidia", "zai"]


def test_keys_under_their_other_names(monkeypatch, client: TestClient):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-other-name")
    monkeypatch.setenv("GLM_API_KEY", "glm-other-name")
    settings = get_settings()
    assert (settings.nvidia_api_key, settings.zai_api_key) == ("nvapi-other-name", "glm-other-name")
    assert settings.llm_chain == ["nvidia", "zai"] and settings.needs()["llm"] is None
    assert {"nvapi-other-name", "glm-other-name"} <= set(settings.secret_values())
    health = client.get("/api/v1/health/env").json()
    assert health["variables"]["NVIDIA_API_KEY"] is True and health["variables"]["ZAI_API_KEY"] is True
    assert [item["id"] for item in health["providers"] if item["configured"]] == ["nvidia", "zai"]
    assert "nvapi-other-name" not in json.dumps(health) and "glm-other-name" not in json.dumps(health)


def test_without_any_key_every_way_in_is_named(client: TestClient):
    needed = get_settings().needs()["llm"]
    assert needed == (
        "an LLM API key (GROQ_API_KEY, GEMINI_API_KEY, NVIDIA_API_KEY, ZAI_API_KEY, OPENROUTER_API_KEY, "
        "OPENAI_API_KEY or ANTHROPIC_API_KEY)"
    )
    with pytest.raises(LLMError) as caught:
        create_llm_client(get_settings())
    assert "NVIDIA_API_KEY, ZAI_API_KEY" in caught.value.message


# --------------------------------------------------------------------------- #
# A task
# --------------------------------------------------------------------------- #
def provider_calls(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[Callable[[str], None]], Any]:
    """The real provider clients of the environment, talking to ``handler``."""
    return lambda notice: create_llm_client(get_settings(), transport=httpx.MockTransport(handler), on_notice=notice)


def test_a_task_carries_on_with_nvidia_and_zai_when_the_others_have_used_up_their_day(
    client: TestClient, app_env, install_deps, github: FakeGitHub, monkeypatch
):
    """Groq's daily allowance is gone. NVIDIA takes over, runs into its limit
    two steps later, and Z.AI finishes the task: one conversation, three providers."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-" + "n" * 40)
    monkeypatch.setenv("ZAI_API_KEY", "z" * 32 + ".zaisecret")
    steps = [
        ("read_file", {"path": "calc.py"}),
        ("replace_in_file", {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}),
        ("run_tests", {"command": CHECK_ADD}),
        ("finish", {"summary": "add() subtracted instead of adding; fixed the operator.", "pr_title": "Fix add()"}),
    ]
    asked: List[str] = []
    done = {"steps": 0, "nvidia": 0}

    def next_step(model: str, as_object: bool = False) -> httpx.Response:
        name, arguments = steps[done["steps"]]
        done["steps"] += 1
        call = {"id": f"call_{done['steps']}", "type": "function", "function": {"name": name, "arguments": arguments if as_object else json.dumps(arguments)}}
        return httpx.Response(200, json=chat_completion(model, tool_calls=[call]))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(f"{request.url.host} {body['model']}")
        if request.url.host == "api.groq.com":
            return httpx.Response(
                429,
                json={"error": {"code": "rate_limit_exceeded", "message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, Used 199800, Requested 4788. Please try again in 14m24s."}},
            )
        if request.url.host == "integrate.api.nvidia.com":
            done["nvidia"] += 1
            return next_step(body["model"]) if done["nvidia"] <= 2 else httpx.Response(429, json=TOO_MANY)
        assert request.url.host == "api.z.ai" and request.headers["authorization"].endswith(".zaisecret")
        return next_step(body["model"], as_object=True)

    install_deps(provider_calls(handler))
    _events, task = run_stream(client, TASK)
    assert task["status"] == "completed" and task["outcome"] == "pull_request" and not task.get("unfinished")
    assert task["tests_status"] == "passed" and [change["path"] for change in task["changed_files"]] == ["calc.py"]
    assert task["llm_chain"] == [
        "groq / openai/gpt-oss-120b",
        "nvidia / deepseek-ai/deepseek-v4.1-flash",
        "zai / glm-4.7-flash",
        "groq / openai/gpt-oss-20b",
        "nvidia / nvidia/nemotron-3-super-120b-a12b",
        "zai / glm-4.5-flash",
    ]
    # NVIDIA's second model is not asked after the account was limited, nor is anybody's stand-in needed.
    assert asked == [
        "api.groq.com openai/gpt-oss-120b",
        "integrate.api.nvidia.com deepseek-ai/deepseek-v4.1-flash",
        "integrate.api.nvidia.com deepseek-ai/deepseek-v4.1-flash",
        "integrate.api.nvidia.com deepseek-ai/deepseek-v4.1-flash",
        "api.z.ai glm-4.7-flash",
        "api.z.ai glm-4.7-flash",
    ]
    usage = {(item["provider"], item["model"]): item for item in task["llm_usage"]}
    assert usage[("groq", "openai/gpt-oss-120b")]["out"] == "its daily limit is used up"
    assert (usage[("nvidia", "deepseek-ai/deepseek-v4.1-flash")]["requests"], usage[("nvidia", "deepseek-ai/deepseek-v4.1-flash")]["failures"]) == (2, 1)
    assert usage[("zai", "glm-4.7-flash")]["requests"] == 2 and usage[("zai", "glm-4.7-flash")]["out"] is None
    assert (task["llm_provider"], task["llm_model"]) == ("zai", "glm-4.7-flash")
    log = messages(task)
    assert any("switching to nvidia / deepseek-ai/deepseek-v4.1-flash" in line for line in log)
    assert any("switching to zai / glm-4.7-flash" in line for line in log)
    # Neither key appears anywhere in what the task leaves behind.
    record = json.dumps(task) + json.dumps(github.pulls["acme/demo"][0])
    assert "nvapi-" not in record and ".zaisecret" not in record


def test_keys_for_the_two_can_come_with_the_request(client: TestClient, app_env, install_deps, monkeypatch):
    from .conftest import fix_add_script

    deps = install_deps(fix_add_script())
    run_stream(client, TASK, headers={"X-Nvidia-Key": "nvapi-browser", "X-Zai-Key": "zai-browser"})
    seen = deps.settings_seen[0]
    assert (seen.nvidia_api_key, seen.zai_api_key) == ("nvapi-browser", "zai-browser")
    assert seen.llm_chain == ["groq", "nvidia", "zai"]
    # They were for that task: the server has none of its own.
    assert get_settings().llm_chain == ["groq"]
