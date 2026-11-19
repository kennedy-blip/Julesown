"""Cloudflare Workers AI, SambaNova, Pollinations and DeepSeek behind the chain.

All four speak OpenAI's chat-completions format. What sets them apart is where
they are called (Cloudflare's address has the account in it), what they accept,
how they count what a key may use, and how they say that it is used up -
Pollinations says it in a reply that looks like an answer. Everything runs
against mocked HTTP transports that answer the way the providers document.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.api.config import (
    DEFAULT_MODELS,
    LLM_PROVIDERS,
    SAMBANOVA_MODEL_PREFERENCES,
    clean_account_id,
    get_settings,
    split_account_key,
    split_models,
)
from apps.api.services.llm import (
    FailoverClient,
    LLMError,
    OpenAICompatibleClient,
    build_provider_client,
    create_llm_client,
    probe_provider,
)

from .conftest import CHECK_ADD, fix_add_script
from .support import FakeGitHub
from .test_api import TASK, messages, run_stream
from .test_fallback import Clock, failure, provider, rate_limit
from .test_llm import MESSAGES, TOOLS, Recorder, chat_completion, groq_client

ACCOUNT = "0123456789abcdef0123456789abcdef"
OTHER_ACCOUNT = "fedcba9876543210fedcba9876543210"
READ_CALL = [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]
CONFIRM_CALL = [{"id": "c", "type": "function", "function": {"name": "confirm", "arguments": '{"ready": true}'}}]
KEYS = {
    "cloudflare": {"CLOUDFLARE_API_TOKEN": "cf-test-token", "CLOUDFLARE_ACCOUNT_ID": ACCOUNT},
    "sambanova": {"SAMBANOVA_API_KEY": "samba-test"},
    "pollinations": {"POLLINATIONS_API_KEY": "sk_test"},
    "deepseek": {"DEEPSEEK_API_KEY": "sk-test"},
}
# What Pollinations writes instead of failing a request (its typographic apostrophes included).
NOT_YOUR_ACCOUNT = "If this isn\u2019t your Pollinations account, contact whoever runs the app or service you\u2019re using."
OUT_OF_CREDIT = (
    "The account behind this API key doesn't have enough credits. Please "
    "[top up](<https://enter.pollinations.ai/top-up?ref=agent_low_balance_topup>) or "
    "[complete a quest](<https://enter.pollinations.ai/quests?ref=agent_low_balance_quests>), then try again."
    f"\n\n{NOT_YOUR_ACCOUNT}"
)
PAID_MODEL = (
    "The account behind this API key doesn't have enough credits. This model needs the paid balance. Please "
    f"[top up](<https://enter.pollinations.ai/top-up?ref=agent_low_balance_topup>), then try again.\n\n{NOT_YOUR_ACCOUNT}"
)
KEY_BUDGET = (
    "The API key used for this request has reached its budget. Please "
    "[raise the key budget](<https://enter.pollinations.ai/edit-key?id=k1&ref=agent_key_budget>), then try again."
    f"\n\nTopping up the wallet does not raise this limit. {NOT_YOUR_ACCOUNT}"
)
QUEST_LOCKED = (
    "We couldn\u2019t verify the account behind this API key. Its Quest balance is locked. Please "
    "[make your first purchase](<https://enter.pollinations.ai/top-up?ref=quest_pollen_restricted>) to unlock it, or "
    f"[contact us](<mailto:hello@pollinations.ai>) if you think this is a mistake.\n\n{NOT_YOUR_ACCOUNT}"
)
NOT_ALLOWED = (
    "The API key used for this request isn't allowed to use openai. Please "
    f"[allow it for this key](<https://enter.pollinations.ai/edit-key?id=k1>), then try again.\n\n{NOT_YOUR_ACCOUNT}"
)


def client_for(
    name: str, recorder: Recorder, monkeypatch, model: Optional[str] = None, **env: str
) -> OpenAICompatibleClient:
    for variable, value in {**KEYS[name], **env}.items():
        monkeypatch.setenv(variable, value)
    client = build_provider_client(get_settings(), name, model=model, transport=recorder.transport)
    assert isinstance(client, OpenAICompatibleClient)
    return client


def answering(**reply: Any) -> Recorder:
    """A provider that answers every request the same way."""
    return Recorder(lambda request, body: httpx.Response(200, json=chat_completion(body["model"], **reply)))


def cloudflare_error(status: int, code: int, message: str, **more: Any) -> httpx.Response:
    return httpx.Response(
        status, json={"result": None, "success": False, "errors": [{"code": code, "message": message, **more}], "messages": []}
    )


def notice(text: str, model: str = "openai") -> httpx.Response:
    """Pollinations' way of refusing: HTTP 200, a chat reply, no tokens used."""
    body = chat_completion(model, content=text)
    body["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return httpx.Response(200, json=body, headers={"cache-control": "private, no-store"})


def error_of(client: OpenAICompatibleClient) -> LLMError:
    with pytest.raises(LLMError) as caught:
        client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    return caught.value


# --------------------------------------------------------------------------- #
# Cloudflare Workers AI
# --------------------------------------------------------------------------- #
def test_cloudflare_request(monkeypatch):
    recorder = answering(tool_calls=READ_CALL)
    client = client_for("cloudflare", recorder, monkeypatch)
    response = client.complete(MESSAGES, TOOLS, max_output_tokens=900, timeout=30)
    request, body = recorder.requests[0], recorder.bodies[0]
    # The account is part of the address; the token is the bearer.
    assert str(request.url) == f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer cf-test-token"
    assert body["model"] == "@cf/zai-org/glm-4.7-flash" == DEFAULT_MODELS["cloudflare"][0]
    assert body["max_tokens"] == 900 and "max_completion_tokens" not in body
    assert body["tools"] == TOOLS and body["tool_choice"] == "auto" and body["temperature"] == 0.2
    # No thinking before every step, and no waiting in a queue while others could answer.
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["options"] == {"rejectIfBusy": True}
    assert (response.provider, response.model) == ("cloudflare", "@cf/zai-org/glm-4.7-flash")
    assert response.tool_calls[0].name == "read_file" and response.tool_calls[0].arguments == {"path": "a.py"}
    assert client.model_env_var == "CLOUDFLARE_MODEL" and client.assumed_tokens_per_minute == 200_000
    # Neurons are counted per account and day; requests per minute are no concern.
    assert client.daily_limit_per_account is True and client.limits_per_account is False


def test_cloudflare_is_sent_text_where_openai_writes_null(monkeypatch):
    """A turn that only calls tools has no text: OpenAI writes ``null`` for
    it, Cloudflare refuses a request that does."""
    history = MESSAGES + [
        {"role": "assistant", "content": None, "tool_calls": READ_CALL, "_private": "never sent"},
        {"role": "tool", "tool_call_id": "c1", "content": "print('hi')"},
        {"role": "assistant", "content": "Read it."},
    ]
    recorder = answering(content="ok")
    client_for("cloudflare", recorder, monkeypatch).complete(history, TOOLS, max_output_tokens=100, timeout=30)
    sent = recorder.bodies[0]["messages"]
    assert sent[2] == {"role": "assistant", "content": "", "tool_calls": READ_CALL}
    assert sent[3]["content"] == "print('hi')" and sent[4]["content"] == "Read it."
    # The conversation itself is left as it was: the next provider gets it unchanged.
    assert history[2]["content"] is None
    elsewhere = answering(content="ok")
    groq_client(elsewhere).complete(history, TOOLS, max_output_tokens=100, timeout=30)
    assert elsewhere.bodies[0]["messages"][2]["content"] is None


@pytest.mark.parametrize(
    "model, setting, expected",
    [
        # Our default: thinking off, for the models that have a switch for it.
        ("@cf/zai-org/glm-4.7-flash", None, {"chat_template_kwargs": {"enable_thinking": False}}),
        ("@cf/google/gemma-4-26b-a4b-it", "none", {"chat_template_kwargs": {"enable_thinking": False}}),
        ("@cf/qwen/qwen3.8-27b", "off", {"chat_template_kwargs": {"enable_thinking": False}}),
        # The others are not sent a field they do not know.
        ("@cf/openai/gpt-oss-20b", None, {}),
        ("@cf/meta/llama-4-scout-17b-16e-instruct", None, {}),
        ("@cf/ibm-granite/granite-4.0-h-micro", "none", {}),
        # An effort that is asked for is passed on; "default" leaves it all to the model.
        ("@cf/openai/gpt-oss-120b", "high", {"reasoning_effort": "high"}),
        ("@cf/zai-org/glm-4.7-flash", "low", {"reasoning_effort": "low"}),
        ("@cf/zai-org/glm-4.7-flash", "default", {}),
        ("@cf/zai-org/glm-4.7-flash", "a lot", {"chat_template_kwargs": {"enable_thinking": False}}),
    ],
)
def test_how_much_a_model_at_cloudflare_thinks(monkeypatch, model, setting, expected):
    recorder = answering(content="ok")
    env = {"CLOUDFLARE_REASONING_EFFORT": setting} if setting else {}
    client_for("cloudflare", recorder, monkeypatch, model=model, **env).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    body = recorder.bodies[0]
    usual = {"model", "messages", "max_tokens", "temperature", "tools", "tool_choice", "options"}
    assert {name: value for name, value in body.items() if name not in usual} == expected


def test_cloudflare_settings(monkeypatch):
    recorder = answering(content="ok")
    client = client_for(
        "cloudflare",
        recorder,
        monkeypatch,
        CLOUDFLARE_MODEL="@cf/openai/gpt-oss-120b, @cf/meta/llama-4-scout-17b-16e-instruct",
        CLOUDFLARE_TEMPERATURE="default",
        # An AI Gateway of the account: the address is given whole, so nothing is filled in.
        CLOUDFLARE_BASE_URL="https://gateway.ai.cloudflare.com/v1/acct/gateway/workers-ai/v1/",
    )
    client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert str(recorder.requests[0].url) == "https://gateway.ai.cloudflare.com/v1/acct/gateway/workers-ai/v1/chat/completions"
    assert recorder.bodies[0]["model"] == "@cf/openai/gpt-oss-120b" and "temperature" not in recorder.bodies[0]
    assert get_settings().models_for("cloudflare") == ["@cf/openai/gpt-oss-120b", "@cf/meta/llama-4-scout-17b-16e-instruct"]
    # With "{account_id}" in it, the account ID is put where it stands.
    monkeypatch.setenv("CLOUDFLARE_BASE_URL", "https://proxy.example.test/{account_id}/v1")
    assert get_settings().base_url_for("cloudflare") == f"https://proxy.example.test/{ACCOUNT}/v1"


def test_a_field_cloudflare_refuses_by_name_is_left_out(monkeypatch):
    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        if "options" in body:
            return cloudflare_error(400, 5006, "Error: unknown property 'options' at '/'")
        return httpx.Response(200, json=chat_completion(body["model"], content="ok"))

    recorder = Recorder(handler)
    client = client_for("cloudflare", recorder, monkeypatch)
    assert client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content == "ok"
    assert "options" in recorder.bodies[0] and "options" not in recorder.bodies[1]
    # It is not sent again with the requests that follow.
    client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert len(recorder.bodies) == 3 and "options" not in recorder.bodies[2]


def test_a_model_server_that_does_not_know_tool_choice_gets_the_tools_alone(monkeypatch):
    """Some of the models Cloudflare hosts take tools but no ``tool_choice``.
    "auto" is what tools mean without it: the field is left out, the tools stay."""

    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        if "tool_choice" in body:
            return cloudflare_error(400, 5006, "Error: unknown property 'tool_choice' at '/'")
        return httpx.Response(200, json=chat_completion(body["model"], tool_calls=READ_CALL))

    recorder = Recorder(handler)
    client = client_for("cloudflare", recorder, monkeypatch, model="@cf/meta/llama-4-scout-17b-16e-instruct")
    assert client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).tool_calls[0].name == "read_file"
    assert [("tool_choice" in body, body["tools"] == TOOLS) for body in recorder.bodies] == [(True, True), (False, True)]
    client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert len(recorder.bodies) == 3 and "tool_choice" not in recorder.bodies[2]
    # A refusal that is about something else is not answered by asking again without it.
    stubborn = Recorder(lambda request, body: cloudflare_error(400, 5006, "Error: oneOf at '/' not met"))
    assert error_of(client_for("cloudflare", stubborn, monkeypatch)).kind == "bad_request" and len(stubborn.bodies) == 1


DAY_USED_UP = (
    "AiError: AiError: you have used up your daily free allocation of 10,000 neurons, please upgrade to "
    "Cloudflare's Workers Paid plan if you would like to continue usage."
)


@pytest.mark.parametrize(
    "response, kind, daily, model_only, said",
    [
        # The code Cloudflare sends for the day's allowance, and the one it documents.
        (cloudflare_error(429, 4006, DAY_USED_UP), "rate_limit", True, False, "starts again at 00:00 UTC"),
        (cloudflare_error(429, 3036, "Account limited"), "rate_limit", True, False, "Account limited"),
        (cloudflare_error(429, 3040, "Capacity temporarily exceeded, please try again."), "rate_limit", False, False, "Capacity temporarily exceeded"),
        # A model is the problem, not the key: the status alone would not say so.
        (cloudflare_error(400, 5007, "No such model @cf/nobody/nothing or task"), "model_unavailable", False, False, "No such model"),
        (cloudflare_error(404, 3042, "The model name is invalid"), "model_unavailable", False, False, "model name is invalid"),
        (cloudflare_error(410, 5028, "@cf/qwen/qwq-32b was deprecated on 2026-11-20."), "model_unavailable", False, False, "was deprecated"),
        (cloudflare_error(403, 5016, "User has not agreed to Llama3.2 model terms"), "model_unavailable", False, False, "model terms"),
        (
            cloudflare_error(403, 5035, "This model requires a Workers Paid plan. See pricing for details."),
            "quota",
            False,
            True,
            "on its paid plan only",
        ),
        # The token, or the account it is used with.
        (cloudflare_error(401, 10000, "Authentication error"), "auth", False, False, "Workers AI permission"),
        (cloudflare_error(401, 9109, "Invalid access token"), "auth", False, False, "Check the CLOUDFLARE API key."),
        (
            cloudflare_error(404, 7003, "Could not route to /client/v4/accounts/x/ai/v1/chat/completions, perhaps your object identifier is invalid?"),
            "auth",
            False,
            False,
            "check CLOUDFLARE_ACCOUNT_ID",
        ),
        (
            cloudflare_error(400, 6003, "Invalid request headers", error_chain=[{"code": 6111, "message": "Invalid format for Authorization header"}]),
            "auth",
            False,
            False,
            "Invalid request headers (Invalid format for Authorization header)",
        ),
        # What is wrong with a request stays a request that is wrong.
        (cloudflare_error(400, 5006, "Error: oneOf at '/' not met"), "bad_request", False, False, "oneOf at '/' not met"),
        (cloudflare_error(408, 3007, "Request timeout"), "server", False, False, "Request timeout"),
    ],
)
def test_cloudflare_error_classification(monkeypatch, response, kind, daily, model_only, said):
    error = error_of(client_for("cloudflare", Recorder(lambda request, body: response), monkeypatch))
    assert (error.kind, error.daily, error.model_only) == (kind, daily, model_only), error.message
    assert error.message.startswith(f"cloudflare returned HTTP {response.status_code}: ") and said in error.message
    assert error.provider == "cloudflare"


def test_tool_calls_in_the_form_of_cloudflares_own_models(monkeypatch):
    """Name and arguments on the call itself, the arguments as an object."""
    calls = [{"name": "read_file", "arguments": {"path": "a.py"}}, {"id": "x2", "name": "read_file", "arguments": '{"path": "b.py"}'}]
    response = client_for("cloudflare", answering(tool_calls=calls), monkeypatch).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert [(call.id, call.name, call.arguments) for call in response.tool_calls] == [
        ("call_0", "read_file", {"path": "a.py"}),
        ("x2", "read_file", {"path": "b.py"}),
    ]
    # What goes back into the conversation is in the form every provider reads.
    assert response.assistant_message["tool_calls"][0] == {
        "id": "call_0",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
    }


def test_the_days_neurons_are_one_allowance_for_all_of_cloudflares_models(monkeypatch):
    asked: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(f"{request.url.host} {body['model']}")
        if request.url.host == "api.cloudflare.com":
            return cloudflare_error(429, 4006, DAY_USED_UP)
        return httpx.Response(200, json=chat_completion(body["model"], content="carries on"))

    for variable, value in {**KEYS["cloudflare"], **KEYS["sambanova"]}.items():
        monkeypatch.setenv(variable, value)
    pool = create_llm_client(get_settings(), transport=httpx.MockTransport(handler))
    assert isinstance(pool, FailoverClient)
    assert pool.chain == [
        "cloudflare / @cf/zai-org/glm-4.7-flash",
        "sambanova / DeepSeek-V3.1",
        "cloudflare / @cf/google/gemma-4-26b-a4b-it",
        "sambanova / gpt-oss-120b",
    ]
    error = failure(pool)
    assert (error.kind, error.daily, error.switched_to) == ("rate_limit", True, "sambanova / DeepSeek-V3.1")
    assert pool.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "carries on"
    # Its other model would be refused for the same reason: it is not asked.
    assert [item["out"] for item in pool.usage()] == ["its daily limit is used up", None, "its daily limit is used up", None]
    assert asked == ["api.cloudflare.com @cf/zai-org/glm-4.7-flash", "api.sambanova.ai DeepSeek-V3.1"]


def test_a_day_that_is_the_accounts_ends_the_run_when_nobody_else_is_there():
    """Cloudflare alone: its second model is no way round the account's
    allowance, and there is nothing to wait for before midnight."""
    used_up = rate_limit("cloudflare returned HTTP 429: the daily free allocation is used up", None, daily=True)
    first, second = provider("cloudflare", [used_up], model="glm"), provider("cloudflare", ["never asked"], model="gemma")
    for scripted in (first, second):
        scripted.daily_limit_per_account = True
    pool = FailoverClient([first, second], clock=Clock())
    error = failure(pool)
    assert (error.daily, error.switched_to, error.next_provider, error.retry_after) == (True, None, None, None)
    assert second.requests == [] and [item["out"] for item in pool.usage()] == ["its daily limit is used up"] * 2
    # Where a day is counted for each model, the provider's next model takes over.
    per_model = FailoverClient(
        [provider("sambanova", [rate_limit("day used up", None, daily=True)], model="a"), provider("sambanova", ["its own day"], model="b")],
        clock=Clock(),
    )
    assert failure(per_model).switched_to == "sambanova / b"
    assert per_model.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "its own day"


def test_a_model_of_cloudflares_paid_plan_leaves_the_free_one_in_use(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "@cf/moonshotai/kimi-k2.6":
            return cloudflare_error(403, 5035, "This model requires a Workers Paid plan. See pricing for details.")
        return httpx.Response(200, json=chat_completion(body["model"], content="within the allowance"))

    for variable, value in KEYS["cloudflare"].items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setenv("CLOUDFLARE_MODEL", "@cf/moonshotai/kimi-k2.6, @cf/zai-org/glm-4.7-flash")
    pool = create_llm_client(get_settings(), transport=httpx.MockTransport(handler))
    error = failure(pool)
    assert (error.kind, error.model_only, error.switched_to) == ("quota", True, "cloudflare / @cf/zai-org/glm-4.7-flash")
    assert pool.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "within the allowance"


def test_a_cloudflare_token_without_its_account(monkeypatch, client: TestClient):
    """The token alone is half of what Cloudflare needs. It is said - not
    tried, and not passed over in silence."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token")
    settings = get_settings()
    assert settings.stored_key_for("cloudflare") == "cf-test-token" and settings.api_key_for("cloudflare") is None
    assert settings.missing_for("cloudflare") == "CLOUDFLARE_ACCOUNT_ID" and settings.base_url_for("cloudflare") is None
    assert settings.llm_chain == [] and settings.configured_llm_providers == []
    assert settings.needs()["llm"].endswith(
        "; CLOUDFLARE_API_TOKEN is set, but Cloudflare Workers AI needs CLOUDFLARE_ACCOUNT_ID as well"
    )
    # A token that cannot be used yet is still a secret.
    assert "cf-test-token" in settings.secret_values()
    tried = probe_provider(settings, "cloudflare", transport=httpx.MockTransport(lambda request: pytest.fail("nothing is sent")))
    assert tried["ok"] is False and tried["kind"] == "no_account"
    assert tried["message"] == (
        "Cloudflare Workers AI needs its account ID as well as the API token: enter it next to the token, "
        "or set CLOUDFLARE_ACCOUNT_ID on the server."
    )
    with pytest.raises(LLMError) as caught:
        build_provider_client(settings, "cloudflare")
    assert caught.value.kind == "auth" and "CLOUDFLARE_ACCOUNT_ID" in caught.value.message

    health = client.get("/api/v1/health/env").json()
    assert health["variables"]["CLOUDFLARE_API_TOKEN"] is True and health["variables"]["CLOUDFLARE_ACCOUNT_ID"] is False
    entry = next(item for item in health["providers"] if item["id"] == "cloudflare")
    assert entry == {
        "id": "cloudflare",
        "label": "Cloudflare Workers AI",
        "configured": False,
        "model": "@cf/zai-org/glm-4.7-flash",
        "models": ["@cf/zai-org/glm-4.7-flash", "@cf/google/gemma-4-26b-a4b-it"],
        "key_variable": "CLOUDFLARE_API_TOKEN",
        "model_variable": "CLOUDFLARE_MODEL",
        "account_variable": "CLOUDFLARE_ACCOUNT_ID",
        "key_set": True,
        "account_set": False,
    }
    assert health["llm"] is None and "CLOUDFLARE_ACCOUNT_ID" in health["needs"]["llm"]

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    health = client.get("/api/v1/health/env").json()
    entry = next(item for item in health["providers"] if item["id"] == "cloudflare")
    assert (entry["configured"], entry["key_set"], entry["account_set"]) == (True, True, True)
    assert health["llm"]["provider"] == "cloudflare" and health["needs"]["llm"] is None
    # Neither value is anything a health check gives away.
    assert "cf-test-token" not in json.dumps(health) and ACCOUNT not in json.dumps(health)
    assert ACCOUNT in get_settings().secret_values()


def test_an_address_without_the_account_in_it_needs_no_account_id(monkeypatch, client: TestClient):
    """CLOUDFLARE_BASE_URL can name an address of one's own - a gateway. When
    it has no ``{account_id}`` in it there is nothing to fill in, so the token
    is all that is needed, and nobody is asked for an account ID."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token")
    monkeypatch.setenv("CLOUDFLARE_BASE_URL", "https://gateway.example.test/workers-ai/v1/")
    settings = get_settings()
    assert settings.needs_account("cloudflare") is False and settings.missing_for("cloudflare") is None
    assert settings.api_key_for("cloudflare") == "cf-test-token" and settings.llm_chain == ["cloudflare"]
    assert settings.needs()["llm"] is None

    recorder = answering(tool_calls=CONFIRM_CALL)
    tried = probe_provider(settings, "cloudflare", transport=recorder.transport)
    assert tried["ok"] is True and tried["tool_call"] is True
    assert {str(request.url) for request in recorder.requests} == {"https://gateway.example.test/workers-ai/v1/chat/completions"}

    health = client.get("/api/v1/health/env").json()
    entry = next(item for item in health["providers"] if item["id"] == "cloudflare")
    # Like any other provider: a key, and no account for the dashboard to ask for.
    assert entry["configured"] is True and not {"account_variable", "key_set", "account_set"} & set(entry)
    assert health["needs"]["llm"] is None and health["llm"]["provider"] == "cloudflare"
    # An account ID that comes along anyway changes nothing about where the request goes.
    brought = settings.with_overrides(api_keys={"cloudflare": f"{ACCOUNT}:cf-browser"})
    assert brought.base_url_for("cloudflare") == "https://gateway.example.test/workers-ai/v1"
    assert brought.api_key_for("cloudflare") == "cf-browser"
    # The other providers never need one.
    assert not any(settings.needs_account(name) for name in ("groq", "sambanova", "pollinations", "deepseek", "nobody"))


def test_where_the_account_id_can_stand(monkeypatch):
    # In a variable of its own - in any case, with spaces around it.
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", f"  {ACCOUNT.upper()} ")
    settings = get_settings()
    assert (settings.cloudflare_account_id, settings.cloudflare_api_key) == (ACCOUNT, "cf-test-token")
    assert settings.llm_chain == ["cloudflare"]
    # In front of the token, where there is one variable for both.
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", f"{ACCOUNT}:cf-test-token")
    settings = get_settings()
    assert (settings.cloudflare_account_id, settings.cloudflare_api_key) == (ACCOUNT, "cf-test-token")
    # The variable of its own counts when both are there.
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", OTHER_ACCOUNT)
    assert get_settings().cloudflare_account_id == OTHER_ACCOUNT
    # The name Cloudflare's own example gives the token.
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-other-name")
    assert get_settings().api_key_for("cloudflare") == "cf-other-name"
    # What is not an account ID does not get into an address.
    for wrong in ("my-account", ACCOUNT[:-1], ACCOUNT + "0", f"{ACCOUNT}/../x", "g" * 32, ""):
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", wrong)
        assert get_settings().cloudflare_account_id is None and get_settings().llm_chain == []
    assert clean_account_id(None) is None and clean_account_id(f" {ACCOUNT} ") == ACCOUNT


@pytest.mark.parametrize(
    "value, expected",
    [
        (f"{ACCOUNT}:cf-token", (ACCOUNT, "cf-token")),
        (f"{ACCOUNT.upper()}: cf-token ", (ACCOUNT, "cf-token")),
        ("cf-token", (None, "cf-token")),
        (f"{ACCOUNT}:", (ACCOUNT, None)),
        ("not-an-account:cf-token", (None, "cf-token")),
        ("", (None, None)),
        (None, (None, None)),
    ],
)
def test_an_account_id_in_front_of_a_key(value, expected):
    assert split_account_key(value) == expected


def test_the_account_id_comes_along_with_a_request(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-server")
    server = get_settings()
    # Both from the browser.
    both = server.with_overrides(api_keys={"cloudflare": f"{ACCOUNT}:cf-browser"})
    assert (both.cloudflare_account_id, both.cloudflare_api_key) == (ACCOUNT, "cf-browser")
    assert both.llm_chain == ["groq", "cloudflare"]
    assert both.with_overrides(primary="cloudflare").llm_chain == ["cloudflare", "groq"]
    # In one go as well: the first choice is judged by what the request brought.
    assert server.with_overrides(api_keys={"cloudflare": f"{ACCOUNT}:cf-browser"}, primary="cloudflare").llm_provider == "cloudflare"
    # A token alone is not enough for a server that has no account ID ...
    alone = server.with_overrides(api_keys={"cloudflare": "cf-browser"}, primary="cloudflare")
    assert alone.cloudflare_api_key == "cf-browser" and alone.llm_chain == ["groq"]
    # ... and is, for one that has: each side can hold either half.
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", OTHER_ACCOUNT)
    assert get_settings().with_overrides(api_keys={"cloudflare": "cf-browser"}).base_url_for("cloudflare").endswith(
        f"/accounts/{OTHER_ACCOUNT}/ai/v1"
    )
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-server")
    half = get_settings().with_overrides(api_keys={"cloudflare": f"{ACCOUNT}:"})
    assert (half.cloudflare_account_id, half.cloudflare_api_key) == (ACCOUNT, "cf-server")
    assert half.llm_chain == ["groq", "cloudflare"]
    # Nothing of it stays behind on the server.
    assert get_settings().llm_chain == ["groq"]


def test_trying_cloudflare_out_stops_at_what_goes_for_the_whole_account(monkeypatch):
    asked: List[str] = []

    def used_up(request: httpx.Request) -> httpx.Response:
        asked.append(json.loads(request.content)["model"])
        return cloudflare_error(429, 4006, DAY_USED_UP)

    settings = get_settings().with_overrides(api_keys={"cloudflare": f"{ACCOUNT}:cf-browser"})
    result = probe_provider(settings, "cloudflare", transport=httpx.MockTransport(used_up))
    # The day's allowance is the account's: the second model would say the same.
    assert asked == ["@cf/zai-org/glm-4.7-flash"] and len(result["models"]) == 1
    assert (result["ok"], result["kind"]) == (False, "rate_limit") and "daily free allocation" in result["message"]

    def working(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(body["model"])
        assert request.url.path == f"/client/v4/accounts/{ACCOUNT}/ai/v1/chat/completions"
        if body["model"].endswith("gemma-4-26b-a4b-it"):
            # No capacity for one model says nothing about the other.
            return cloudflare_error(429, 3040, "Capacity temporarily exceeded, please try again.")
        return httpx.Response(200, json=chat_completion(body["model"], tool_calls=CONFIRM_CALL))

    asked.clear()
    result = probe_provider(settings, "cloudflare", transport=httpx.MockTransport(working))
    assert asked == ["@cf/zai-org/glm-4.7-flash", "@cf/google/gemma-4-26b-a4b-it"]
    first, second = result["models"]
    assert (first["ok"], first["tool_call"]) == (True, True)
    assert first["message"] == "Cloudflare Workers AI accepted the key and @cf/zai-org/glm-4.7-flash called the test tool."
    assert (second["ok"], second["kind"]) == (False, "rate_limit")
    assert "cf-browser" not in json.dumps(result) and ACCOUNT not in json.dumps(result)


def test_model_names_can_start_with_an_at_sign():
    assert split_models("@cf/zai-org/glm-4.7-flash, @hf/nousresearch/hermes-2-pro-mistral-7b,openai") == [
        "@cf/zai-org/glm-4.7-flash",
        "@hf/nousresearch/hermes-2-pro-mistral-7b",
        "openai",
    ]
    # Still names only: nothing that could be a path, an option or a sentence.
    assert split_models("/etc/passwd, -rf, two words, .hidden, @cf/ok") == ["@cf/ok"]
    settings = get_settings().with_overrides(models={"cloudflare": "@cf/openai/gpt-oss-20b"})
    assert settings.models_for("cloudflare") == ["@cf/openai/gpt-oss-20b"]


# --------------------------------------------------------------------------- #
# SambaNova
# --------------------------------------------------------------------------- #
def test_sambanova_request(monkeypatch):
    recorder = answering(tool_calls=READ_CALL)
    client = client_for("sambanova", recorder, monkeypatch)
    response = client.complete(MESSAGES, TOOLS, max_output_tokens=900, timeout=30)
    request, body = recorder.requests[0], recorder.bodies[0]
    assert str(request.url) == "https://api.sambanova.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer samba-test"
    # Its API refuses a request with a field it does not list: only listed ones are sent.
    assert set(body) == {"model", "messages", "max_tokens", "temperature", "tools", "tool_choice"}
    assert body["model"] == "DeepSeek-V3.1" == DEFAULT_MODELS["sambanova"][0]
    assert body["max_tokens"] == 900 and body["temperature"] == 0.2 and body["tool_choice"] == "auto"
    assert (response.provider, response.tool_calls[0].name) == ("sambanova", "read_file")
    assert client.model_env_var == "SAMBANOVA_MODEL" and client.assumed_tokens_per_minute == 200_000
    # Its limits are per model: one model's says nothing about the next.
    assert client.limits_per_account is False and client.daily_limit_per_account is False


def test_sambanova_settings_and_a_parameter_a_model_does_not_take(monkeypatch):
    def handler(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        if "reasoning_effort" in body:
            return httpx.Response(
                400,
                json={
                    "error": {"message": "Unknown parameter: 'reasoning_effort'.", "type": "invalid_request_error", "param": "reasoning_effort", "code": "unknown_parameter"},
                    "request_id": "req-1",
                },
            )
        return httpx.Response(200, json=chat_completion(body["model"], content="ok"))

    recorder = Recorder(handler)
    client = client_for(
        "sambanova",
        recorder,
        monkeypatch,
        SAMBANOVA_MODEL="gpt-oss-120b",
        SAMBANOVA_REASONING_EFFORT="high",
        SAMBANOVA_TEMPERATURE="0",
        SAMBANOVA_BASE_URL="https://samba.example.test/v1/",
    )
    assert client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content == "ok"
    first, second = recorder.bodies
    assert str(recorder.requests[0].url) == "https://samba.example.test/v1/chat/completions"
    assert (first["model"], first["reasoning_effort"], first["temperature"]) == ("gpt-oss-120b", "high", 0.0)
    assert "reasoning_effort" not in second and second["temperature"] == 0.0


def samba_limit(day_left: Optional[int]) -> httpx.Response:
    headers = {"x-ratelimit-limit-requests": "20", "x-ratelimit-remaining-requests": "0", "x-ratelimit-limit-requests-day": "20"}
    if day_left is not None:
        headers["x-ratelimit-remaining-requests-day"] = str(day_left)
    return httpx.Response(
        429,
        headers=headers,
        json={"error": {"message": "Rate limit exceeded", "type": "rate_limit_exceeded", "param": None, "code": "insufficient_quota"}, "request_id": "req-2"},
    )


@pytest.mark.parametrize("day_left, daily", [(0, True), (7, False), (None, False)])
def test_a_header_says_whether_sambanovas_minute_or_its_day_is_used_up(monkeypatch, day_left, daily):
    error = error_of(client_for("sambanova", Recorder(lambda request, body: samba_limit(day_left)), monkeypatch))
    # "insufficient_quota" is its code for a rate limit, not for an empty account.
    assert (error.kind, error.daily, error.status_code) == ("rate_limit", daily, 429)
    # Its words are the same for the minute and the day, so what the header says is added to them.
    said = "sambanova returned HTTP 429: Rate limit exceeded"
    assert error.message == (f"{said}. This model's requests for today are used up (20 a day)." if daily else said)


def test_what_is_added_to_a_providers_words_starts_a_sentence_of_its_own(monkeypatch):
    """Providers end their messages with or without a full stop; what is added
    to them reads as the next sentence either way."""
    for said, expected in (
        ("Authentication error", "cloudflare returned HTTP 401: Authentication error. The token has to be"),
        ("Authentication error.", "cloudflare returned HTTP 401: Authentication error. The token has to be"),
    ):
        error = error_of(client_for("cloudflare", Recorder(lambda request, body, said=said: cloudflare_error(401, 10000, said)), monkeypatch))
        assert error.message.startswith(expected) and error.message.endswith("is used. Check the CLOUDFLARE API key."), error.message


def test_a_refusal_while_requests_are_left_is_about_the_days_tokens(monkeypatch):
    """SambaNova's free tier also counts tokens, 200,000 a day for each model,
    and a long conversation uses those up before its twenty requests. No
    header counts them: what says so is that requests are left, for this
    minute and for today, and the request was refused all the same."""

    def refusal(minute_left: Optional[int], day_left: Optional[int], message: str = "Rate limit exceeded") -> httpx.Response:
        headers = {"x-ratelimit-limit-requests": "20", "x-ratelimit-limit-requests-day": "20"}
        if minute_left is not None:
            headers["x-ratelimit-remaining-requests"] = str(minute_left)
        if day_left is not None:
            headers["x-ratelimit-remaining-requests-day"] = str(day_left)
        return httpx.Response(429, headers=headers, json={"error": {"message": message, "type": "rate_limit_exceeded", "code": "insufficient_quota"}})

    def classify(provider: str, response: httpx.Response) -> LLMError:
        return error_of(client_for(provider, Recorder(lambda request, body: response), monkeypatch))

    error = classify("sambanova", refusal(12, 7))
    assert (error.kind, error.daily) == ("rate_limit", True)
    assert error.message == (
        "sambanova returned HTTP 429: Rate limit exceeded. Requests are left, so this is taken to be the model's limit of tokens per day."
    )
    # The minute is used up and the day is not: that is the minute, and it passes.
    assert classify("sambanova", refusal(0, 7)).daily is False
    # Without the headers nothing is concluded.
    assert classify("sambanova", refusal(None, None)).daily is False
    assert classify("sambanova", refusal(12, None)).daily is False and classify("sambanova", refusal(None, 7)).daily is False
    # The day's requests are named as what they are.
    assert "requests for today are used up (20 a day)" in classify("sambanova", refusal(0, 0)).message
    # A provider that is not known to count tokens by the day is not second-guessed.
    assert classify("deepseek", refusal(12, 7)).daily is False


def test_a_model_whose_tokens_for_the_day_are_gone_is_set_aside(monkeypatch):
    asked: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(body["model"])
        if body["model"] == "DeepSeek-V3.1":
            return httpx.Response(
                429,
                headers={"x-ratelimit-remaining-requests": "19", "x-ratelimit-remaining-requests-day": "11"},
                json={"error": {"message": "Rate limit exceeded", "type": "rate_limit_exceeded", "code": "insufficient_quota"}},
            )
        return httpx.Response(200, json=chat_completion(body["model"], content="its own allowance"))

    monkeypatch.setenv("SAMBANOVA_API_KEY", "samba-test")
    pool = create_llm_client(get_settings(), transport=httpx.MockTransport(handler))
    error = failure(pool)
    assert (error.daily, error.switched_to) == (True, "sambanova / gpt-oss-120b")
    # It is not asked again in this run - not after a pause either.
    for _ in range(3):
        assert pool.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "its own allowance"
    assert asked == ["DeepSeek-V3.1", "gpt-oss-120b", "gpt-oss-120b", "gpt-oss-120b"]
    assert [item["out"] for item in pool.usage()] == ["its daily limit is used up", None]


def test_the_day_is_named_once_and_without_a_number_nobody_gave(monkeypatch):
    def refusal(message: str, **headers: str) -> httpx.Response:
        return httpx.Response(429, headers=headers, json={"error": {"message": message, "type": "rate_limit_exceeded", "code": "insufficient_quota"}})

    # No header with the size of the allowance: none is made up.
    bare = refusal("Rate limit exceeded.", **{"x-ratelimit-remaining-requests-day": "0"})
    error = error_of(client_for("sambanova", Recorder(lambda request, body: bare), monkeypatch))
    assert error.daily and error.message == "sambanova returned HTTP 429: Rate limit exceeded. This model's requests for today are used up."
    # A refusal that says "per day" itself needs nothing added.
    worded = refusal("Requests per day limit exceeded for model DeepSeek-V3.1.", **{"x-ratelimit-remaining-requests-day": "0"})
    error = error_of(client_for("sambanova", Recorder(lambda request, body: worded), monkeypatch))
    assert error.daily and error.message == "sambanova returned HTTP 429: Requests per day limit exceeded for model DeepSeek-V3.1."


def test_sambanovas_day_is_counted_for_each_model(monkeypatch):
    asked: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(body["model"])
        if body["model"] == "DeepSeek-V3.1":
            return samba_limit(0)
        return httpx.Response(200, json=chat_completion(body["model"], content="twenty more"))

    monkeypatch.setenv("SAMBANOVA_API_KEY", "samba-test")
    pool = create_llm_client(get_settings(), transport=httpx.MockTransport(handler))
    assert isinstance(pool, FailoverClient) and pool.chain == ["sambanova / DeepSeek-V3.1", "sambanova / gpt-oss-120b"]
    error = failure(pool)
    assert (error.daily, error.switched_to) == (True, "sambanova / gpt-oss-120b")
    assert pool.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "twenty more"
    assert [item["out"] for item in pool.usage()] == ["its daily limit is used up", None]
    assert asked == ["DeepSeek-V3.1", "gpt-oss-120b"]


def test_a_reply_sambanova_could_not_read_is_not_lost(monkeypatch):
    """A tool call with a line break inside a string: SambaNova fails the
    request and hands back what the model wrote. The call is in there."""
    written = '{"name": "read_file", "arguments": {"path": "a\n.py"}}'

    def unreadable(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": {"message": "Failed to parse tool call JSON from the model output.", "type": "server_error", "code": None, "error_model_output": written}, "request_id": "req-3"},
        )

    client = client_for("sambanova", Recorder(unreadable), monkeypatch)
    response = client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert response.recovered is True and response.provider == "sambanova"
    assert [(call.name, call.arguments) for call in response.tool_calls] == [("read_file", {"path": "a\n.py"})]

    # When there is no call in it, it is the model's reply that failed - not the server.
    def rambling(request: httpx.Request, body: Dict[str, Any]) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": {"message": "Failed to parse tool call JSON from the model output.", "type": "server_error", "model_output": "<tool>read_file a.py"}},
        )

    error = error_of(client_for("sambanova", Recorder(rambling), monkeypatch))
    assert error.kind == "tool_format" and error.failed_generation == "<tool>read_file a.py"
    # A server error without the model's words stays a server error.
    plain = error_of(client_for("sambanova", Recorder(lambda request, body: httpx.Response(500, text="Internal Server Error")), monkeypatch))
    assert plain.kind == "server"


def test_a_sambanova_model_that_was_withdrawn_is_replaced_by_one_it_still_lists(monkeypatch):
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.method == "GET":
            listed = ["gpt-oss-120b", "E5-Mistral-7B-Instruct", "Meta-Llama-3.3-70B-Instruct", "Whisper-Large-v3"]
            return httpx.Response(200, json={"object": "list", "data": [{"id": name, "object": "model"} for name in listed]})
        body = json.loads(request.content)
        if body["model"] == "DeepSeek-V3.1":
            # Its documented answer for a model that is gone: HTTP 410 and a sentence.
            return httpx.Response(410, json={"error": "The model DeepSeek-V3.1 has been deprecated."})
        return httpx.Response(200, json=chat_completion(body["model"], content="still here"))

    monkeypatch.setenv("SAMBANOVA_API_KEY", "samba-test")
    notices: List[str] = []
    client = build_provider_client(get_settings(), "sambanova", transport=httpx.MockTransport(handler), on_notice=notices.append)
    assert client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content == "still here"
    # gpt-oss-120b is the provider's second entry in the chain already: the next known one stands in.
    assert SAMBANOVA_MODEL_PREFERENCES[:3] == ("DeepSeek-V3.1", "gpt-oss-120b", "Meta-Llama-3.3-70B-Instruct")
    assert client.model == "Meta-Llama-3.3-70B-Instruct" and "using 'Meta-Llama-3.3-70B-Instruct' instead" in notices[0]
    assert calls == ["POST /v1/chat/completions", "GET /v1/models", "POST /v1/chat/completions"]
    # "Gone" is about the model even when nothing else is said - and with no list to pick from, that is the answer.
    gone = error_of(client_for("sambanova", Recorder(lambda request, body: httpx.Response(410, json={"error": "Gone"})), monkeypatch))
    assert gone.kind == "model_unavailable" and "Set SAMBANOVA_MODEL to a model your account can use." in gone.message


# --------------------------------------------------------------------------- #
# Pollinations
# --------------------------------------------------------------------------- #
def test_pollinations_request(monkeypatch):
    recorder = Recorder(lambda request, body: httpx.Response(200, json=chat_completion("gpt-5.4-nano-2026-03-17", tool_calls=READ_CALL)))
    client = client_for("pollinations", recorder, monkeypatch)
    response = client.complete(MESSAGES, TOOLS, max_output_tokens=900, timeout=30)
    request, body = recorder.requests[0], recorder.bodies[0]
    assert str(request.url) == "https://gen.pollinations.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk_test"
    # Its own name for the OpenAI model it offers as its default; the answer names the model.
    assert body["model"] == "openai" == DEFAULT_MODELS["pollinations"][0] and response.model == "gpt-5.4-nano-2026-03-17"
    assert body["max_tokens"] == 900 and body["temperature"] == 0.2 and body["tool_choice"] == "auto"
    # Many makers' models, each with its own idea of reasoning: nothing is asked for unless it is set.
    assert set(body) == {"model", "messages", "max_tokens", "temperature", "tools", "tool_choice"}
    assert response.tool_calls[0].name == "read_file" and client.model_env_var == "POLLINATIONS_MODEL"

    monkeypatch.setenv("POLLINATIONS_REASONING_EFFORT", "low")
    monkeypatch.setenv("POLLINATIONS_TEMPERATURE", "default")
    monkeypatch.setenv("POLLINATIONS_MODEL", "deepseek/deepseek-v4.1-flash")
    again = answering(content="ok")
    build_provider_client(get_settings(), "pollinations", transport=again.transport).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert again.bodies[0]["model"] == "deepseek/deepseek-v4.1-flash" and again.bodies[0]["reasoning_effort"] == "low"
    assert "temperature" not in again.bodies[0]


@pytest.mark.parametrize(
    "text, kind, model_only, status, said",
    [
        (OUT_OF_CREDIT, "quota", False, 402, "doesn't have enough credits. Please top up or complete a quest, then try again."),
        (PAID_MODEL, "quota", True, 402, "This model needs the paid balance. Please top up, then try again."),
        (KEY_BUDGET, "quota", False, 402, "has reached its budget. Please raise the key budget, then try again."),
        (QUEST_LOCKED, "quota", False, 402, "Its Quest balance is locked."),
        (NOT_ALLOWED, "model_unavailable", False, 403, "isn't allowed to use openai. Please allow it for this key, then try again."),
    ],
)
def test_a_notice_from_pollinations_is_a_failure_not_an_answer(monkeypatch, text, kind, model_only, status, said):
    """Pollinations answers HTTP 200 with a chat reply that tells a person to
    top up. Taken for the model's answer it would be the end of a task."""
    error = error_of(client_for("pollinations", Recorder(lambda request, body: notice(text)), monkeypatch))
    assert (error.kind, error.model_only, error.status_code, error.provider) == (kind, model_only, status, "pollinations")
    assert error.message.startswith("pollinations did not answer the request: ") and said in error.message
    # The links are for somebody at a chat window; where to look is said once, plainly.
    assert "](<" not in error.message and "ref=agent" not in error.message
    assert error.message.endswith("(The balance and the key's budget are shown at https://enter.pollinations.ai.)")
    assert "contact whoever runs the app" not in error.message


def test_what_a_model_wrote_is_an_answer_whatever_it_says(monkeypatch):
    # The same words, but tokens were spent on them: a model wrote this (it was asked about the notice).
    quoted = chat_completion("openai", content=OUT_OF_CREDIT)
    quoted["usage"] = {"prompt_tokens": 420, "completion_tokens": 61, "total_tokens": 481}
    client = client_for("pollinations", Recorder(lambda request, body: httpx.Response(200, json=quoted)), monkeypatch)
    assert client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content == OUT_OF_CREDIT
    # A reply that calls tools is one, and so is any other text.
    with_calls = chat_completion("openai", content=OUT_OF_CREDIT, tool_calls=READ_CALL)
    with_calls["usage"] = {"total_tokens": 0}
    client = client_for("pollinations", Recorder(lambda request, body: httpx.Response(200, json=with_calls)), monkeypatch)
    assert client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).tool_calls[0].name == "read_file"
    plain = client_for("pollinations", answering(content="Please top up the tank, then try again."), monkeypatch)
    assert plain.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content.startswith("Please top up")
    # Nobody else's replies are looked at this way.
    groq = groq_client(Recorder(lambda request, body: notice(OUT_OF_CREDIT, model=body["model"])))
    assert groq.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content == OUT_OF_CREDIT


def pollinations_error(status: int, code: str, message: str, **details: Any) -> httpx.Response:
    error: Dict[str, Any] = {"message": message, "code": code, "timestamp": "2026-11-19T10:00:00.000Z"}
    if details:
        error["details"] = details
    return httpx.Response(status, json={"success": False, "error": error, "status": status})


@pytest.mark.parametrize(
    "response, kind, model_only",
    [
        # The same refusals as plain errors, which is how a request for JSON output gets them.
        (pollinations_error(402, "INSUFFICIENT_BALANCE", "Insufficient balance. This request costs about $0.0012, but your available balance is $0.0000. Top up at https://enter.pollinations.ai/top-up."), "quota", False),
        (pollinations_error(402, "INSUFFICIENT_BALANCE", "This model requires the paid balance, so the Quest balance can't cover it. This request costs about $0.0100, but your available paid balance is $0.0000."), "quota", True),
        (pollinations_error(402, "QUEST_POLLEN_ONLY", "This model needs the paid balance, but this API key only spends the Quest balance."), "quota", True),
        (pollinations_error(402, "KEY_BUDGET_EXHAUSTED", "API key budget too low. This request costs about $0.0012, but this key has $0.0000 left."), "quota", False),
        (pollinations_error(401, "UNAUTHORIZED", "A valid API key is required. Get one at https://enter.pollinations.ai/keys"), "auth", False),
        (pollinations_error(429, "RATE_LIMITED", "Too many requests. Please slow down."), "rate_limit", False),
        # The company behind a model is throttling or down: Pollinations reports a bad gateway.
        (pollinations_error(502, "BAD_GATEWAY", "Rate limit reached", name="UpstreamError", upstreamStatus=429, upstreamHost="upstream.example.test"), "server", False),
    ],
)
def test_pollinations_errors_in_their_plain_form(monkeypatch, response, kind, model_only):
    error = error_of(client_for("pollinations", Recorder(lambda request, body: response), monkeypatch))
    assert (error.kind, error.model_only) == (kind, model_only), error.message
    assert error.message.startswith(f"pollinations returned HTTP {response.status_code}: ")


def test_an_empty_account_at_pollinations_takes_out_both_of_its_models(monkeypatch):
    asked: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(f"{request.url.host} {body['model']}")
        if request.url.host == "gen.pollinations.ai":
            return notice(OUT_OF_CREDIT, model=body["model"])
        return httpx.Response(200, json=chat_completion(body["model"], content="carries on"))

    monkeypatch.setenv("POLLINATIONS_API_KEY", "sk_test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    pool = create_llm_client(get_settings(), transport=httpx.MockTransport(handler))
    assert isinstance(pool, FailoverClient)
    assert pool.chain == ["pollinations / openai", "deepseek / deepseek-flash", "pollinations / openai/gpt-oss-20b"]
    error = failure(pool)
    assert (error.kind, error.switched_to) == ("quota", "deepseek / deepseek-flash")
    assert pool.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "carries on"
    assert [item["out"] for item in pool.usage()] == ["its account has no credit left", None, "its account has no credit left"]
    assert asked == ["gen.pollinations.ai openai", "api.deepseek.com deepseek-flash"]


def test_a_model_only_bought_credit_pays_for_leaves_the_other_one_in_use(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "anthropic/claude-sonnet-5":
            return notice(PAID_MODEL, model=body["model"])
        return httpx.Response(200, json=chat_completion(body["model"], content="paid for with earned credit"))

    monkeypatch.setenv("POLLINATIONS_API_KEY", "sk_test")
    monkeypatch.setenv("POLLINATIONS_MODEL", "anthropic/claude-sonnet-5, openai")
    pool = create_llm_client(get_settings(), transport=httpx.MockTransport(handler))
    error = failure(pool)
    assert (error.kind, error.model_only, error.switched_to) == ("quota", True, "pollinations / openai")
    assert pool.complete(MESSAGES, [], max_output_tokens=10, timeout=10).content == "paid for with earned credit"
    assert [item["out"] for item in pool.usage()] == ["its account has no credit for this model", None]


def test_trying_a_pollinations_key_out(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "openai":
            return httpx.Response(200, json=chat_completion("gpt-5.4-nano-2026-03-17", tool_calls=CONFIRM_CALL))
        return notice(NOT_ALLOWED.replace("use openai", f"use {body['model']}"), model=body["model"])

    settings = get_settings().with_overrides(api_keys={"pollinations": "sk_browser"})
    result = probe_provider(settings, "pollinations", transport=httpx.MockTransport(handler))
    first, second = result["models"]
    assert (first["ok"], first["tool_call"], first["answered_by"]) == (True, True, "gpt-5.4-nano-2026-03-17")
    assert first["message"] == "Pollinations accepted the key and openai (answered by gpt-5.4-nano-2026-03-17) called the test tool."
    # A key that is restricted to some models says so for the others - it is not "ok, answered in text".
    assert (second["ok"], second["kind"]) == (False, "model_unavailable")
    assert "isn't allowed to use openai/gpt-oss-20b" in second["message"]
    # An empty account ends the test at the first model.
    empty = probe_provider(settings, "pollinations", transport=httpx.MockTransport(lambda request: notice(OUT_OF_CREDIT)))
    assert (empty["ok"], empty["kind"], len(empty["models"])) == (False, "quota", 1)


# --------------------------------------------------------------------------- #
# DeepSeek
# --------------------------------------------------------------------------- #
def test_deepseek_request(monkeypatch):
    recorder = answering(tool_calls=READ_CALL)
    client = client_for("deepseek", recorder, monkeypatch)
    response = client.complete(MESSAGES, TOOLS, max_output_tokens=900, timeout=30)
    request, body = recorder.requests[0], recorder.bodies[0]
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert body["model"] == "deepseek-flash" and body["max_tokens"] == 900 and body["temperature"] == 0.2
    # Thinking is on unless a request says otherwise - and with tools DeepSeek then
    # refuses a conversation whose earlier replies come without their reasoning.
    assert body["thinking"] == {"type": "disabled"} and "reasoning_effort" not in body
    assert response.tool_calls[0].name == "read_file" and client.model_env_var == "DEEPSEEK_MODEL"
    # It is paid for by the token: one model, and none that costs more behind it.
    assert DEFAULT_MODELS["deepseek"] == ("deepseek-flash",) and get_settings().models_for("deepseek") == ["deepseek-flash"]

    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_TEMPERATURE", "0.7")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/")
    again = answering(content="ok")
    build_provider_client(get_settings(), "deepseek", transport=again.transport).complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert str(again.requests[0].url) == "https://api.deepseek.com/v1/chat/completions"
    assert (again.bodies[0]["model"], again.bodies[0]["temperature"]) == ("deepseek-v4-pro", 0.7)
    assert again.bodies[0]["thinking"] == {"type": "disabled"}


def test_a_reasoning_a_reply_came_with_is_not_sent_back(monkeypatch):
    """Whatever a provider adds to a reply stays out of the conversation: the
    next request - to DeepSeek or to anybody - is made of what all of them read."""
    recorder = answering(content=None, tool_calls=READ_CALL, reasoning_content="The file has to be read first.")
    client = client_for("deepseek", recorder, monkeypatch)
    first = client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30)
    assert set(first.assistant_message) == {"role", "content", "tool_calls"}
    history = MESSAGES + [first.assistant_message, {"role": "tool", "tool_call_id": "c1", "content": "x = 1"}]
    client.complete(history, TOOLS, max_output_tokens=100, timeout=30)
    assert "reasoning_content" not in json.dumps(recorder.bodies[1]["messages"])


def test_deepseek_without_a_balance(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.host == "api.deepseek.com":
            return httpx.Response(402, json={"error": {"message": "Insufficient Balance", "type": "unknown_error", "param": None, "code": "invalid_request_error"}})
        return httpx.Response(200, json=chat_completion(body["model"], content="carries on"))

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    alone = error_of(build_provider_client(get_settings(), "deepseek", transport=httpx.MockTransport(handler)))
    assert (alone.kind, alone.model_only, alone.status_code) == ("quota", False, 402)
    assert alone.message == (
        "deepseek returned HTTP 402: Insufficient Balance. DeepSeek's API is paid for from the account's balance: "
        "it has to be topped up at platform.deepseek.com."
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    pool = create_llm_client(get_settings(), transport=httpx.MockTransport(handler))
    assert isinstance(pool, FailoverClient) and pool.chain == ["deepseek / deepseek-flash", "openai / gpt-6-luna"]
    assert failure(pool).switched_to == "openai / gpt-6-luna"
    assert pool.usage()[0]["out"] == "its account has no credit left"


def test_deepseek_keeps_the_line_open_with_empty_lines(monkeypatch):
    """While a request waits its turn DeepSeek sends empty lines; the answer follows them."""
    waited = "\n\n\n" + json.dumps(chat_completion("deepseek-flash", content="worth the wait"))
    client = client_for(
        "deepseek",
        Recorder(lambda request, body: httpx.Response(200, content=waited.encode(), headers={"content-type": "application/json"})),
        monkeypatch,
    )
    assert client.complete(MESSAGES, TOOLS, max_output_tokens=100, timeout=30).content == "worth the wait"


def test_a_deepseek_model_that_is_gone_is_not_swapped_for_a_dearer_one(monkeypatch):
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        return httpx.Response(400, json={"error": {"message": "Model Not Exist", "type": "invalid_request_error", "param": None, "code": "invalid_request_error"}})

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
    client = build_provider_client(get_settings(), "deepseek", transport=httpx.MockTransport(handler))
    error = error_of(client)
    assert error.kind == "model_unavailable" and client.model == "deepseek-chat"
    assert calls == ["POST /chat/completions"]


# --------------------------------------------------------------------------- #
# Their place among the others
# --------------------------------------------------------------------------- #
def test_where_the_four_stand_in_the_chain(monkeypatch):
    # What renews itself first - the larger allowances before the smaller - then what spends a balance.
    assert LLM_PROVIDERS == (
        "groq",
        "gemini",
        "nvidia",
        "zai",
        "cloudflare",
        "sambanova",
        "openrouter",
        "pollinations",
        "deepseek",
        "openai",
        "anthropic",
    )
    for name, key in (("GROQ", "gsk"), ("OPENROUTER", "sk-or"), ("SAMBANOVA", "samba"), ("POLLINATIONS", "sk_p"), ("DEEPSEEK", "sk-d")):
        monkeypatch.setenv(f"{name}_API_KEY", key)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    assert create_llm_client(get_settings()).chain == [
        "groq / openai/gpt-oss-120b",
        "cloudflare / @cf/zai-org/glm-4.7-flash",
        "sambanova / DeepSeek-V3.1",
        "openrouter / openrouter/free",
        "pollinations / openai",
        "deepseek / deepseek-flash",
        "groq / openai/gpt-oss-20b",
        "cloudflare / @cf/google/gemma-4-26b-a4b-it",
        "sambanova / gpt-oss-120b",
        "pollinations / openai/gpt-oss-20b",
    ]
    monkeypatch.setenv("JULES_LLM_ORDER", "deepseek, cloudflare")
    assert get_settings().llm_chain == ["deepseek", "cloudflare", "groq", "sambanova", "openrouter", "pollinations"]
    monkeypatch.setenv("JULES_LLM_PROVIDER", "pollinations")
    assert get_settings().llm_chain[:3] == ["pollinations", "deepseek", "cloudflare"]
    monkeypatch.setenv("JULES_LLM_FALLBACK", "false")
    assert get_settings().llm_slots == [("pollinations", "openai")]


def test_health_lists_the_four(monkeypatch, client: TestClient):
    for variables in KEYS.values():
        for variable, value in variables.items():
            monkeypatch.setenv(variable, value)
    health = client.get("/api/v1/health/env").json()
    assert [item["id"] for item in health["providers"]] == list(LLM_PROVIDERS)
    assert [item["id"] for item in health["providers"] if item["configured"]] == ["cloudflare", "sambanova", "pollinations", "deepseek"]
    by_id = {item["id"]: item for item in health["providers"]}
    assert by_id["sambanova"] == {
        "id": "sambanova",
        "label": "SambaNova",
        "configured": True,
        "model": "DeepSeek-V3.1",
        "models": ["DeepSeek-V3.1", "gpt-oss-120b"],
        "key_variable": "SAMBANOVA_API_KEY",
        "model_variable": "SAMBANOVA_MODEL",
    }
    assert (by_id["pollinations"]["label"], by_id["pollinations"]["models"]) == ("Pollinations", ["openai", "openai/gpt-oss-20b"])
    assert (by_id["deepseek"]["label"], by_id["deepseek"]["models"]) == ("DeepSeek", ["deepseek-flash"])
    # Only the one that needs an account says anything about one.
    assert [item["id"] for item in health["providers"] if "account_variable" in item] == ["cloudflare"]
    for variable in ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "SAMBANOVA_API_KEY", "POLLINATIONS_API_KEY", "DEEPSEEK_API_KEY"):
        assert health["variables"][variable] is True
    assert health["llm"]["chain"][:4] == [
        {"provider": "cloudflare", "model": "@cf/zai-org/glm-4.7-flash"},
        {"provider": "sambanova", "model": "DeepSeek-V3.1"},
        {"provider": "pollinations", "model": "openai"},
        {"provider": "deepseek", "model": "deepseek-flash"},
    ]
    text = json.dumps(health)
    assert not any(value in text for variables in KEYS.values() for value in variables.values())
    assert set(value for variables in KEYS.values() for value in variables.values()) <= set(get_settings().secret_values())


# --------------------------------------------------------------------------- #
# A task
# --------------------------------------------------------------------------- #
def provider_calls(handler: Callable[[httpx.Request], httpx.Response]) -> Callable[[Callable[[str], None]], Any]:
    """The real provider clients of the environment, talking to ``handler``."""
    return lambda on_notice: create_llm_client(get_settings(), transport=httpx.MockTransport(handler), on_notice=on_notice)


def test_a_task_goes_on_past_an_empty_account_and_a_used_up_day(
    client: TestClient, app_env, install_deps, github: FakeGitHub, monkeypatch
):
    """Groq's day is used up and so are Cloudflare's neurons. SambaNova does
    two steps and reaches its twenty requests, Pollinations has no credit and
    says so in a reply, DeepSeek finishes. One conversation, five providers -
    and Pollinations' notice is nobody's answer."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-" + "c" * 40)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)
    monkeypatch.setenv("SAMBANOVA_API_KEY", "samba-" + "s" * 36)
    monkeypatch.setenv("POLLINATIONS_API_KEY", "sk_" + "p" * 40)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-" + "d" * 32)
    steps = [
        ("read_file", {"path": "calc.py"}),
        ("replace_in_file", {"path": "calc.py", "old_text": "return a - b", "new_text": "return a + b"}),
        ("run_tests", {"command": CHECK_ADD}),
        ("finish", {"summary": "add() subtracted instead of adding; fixed the operator.", "pr_title": "Fix add()"}),
    ]
    asked: List[str] = []
    done = {"steps": 0, "sambanova": 0}
    histories: Dict[str, List[Any]] = {}

    def next_step(model: str) -> httpx.Response:
        name, arguments = steps[done["steps"]]
        done["steps"] += 1
        call = {"id": f"call_{done['steps']}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        return httpx.Response(200, json=chat_completion(model, tool_calls=[call]))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(f"{request.url.host} {body['model']}")
        histories[request.url.host] = body["messages"]
        if request.url.host == "api.groq.com":
            return httpx.Response(
                429,
                json={"error": {"code": "rate_limit_exceeded", "message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, Used 199800, Requested 4788. Please try again in 14m24s."}},
            )
        if request.url.host == "api.cloudflare.com":
            assert request.url.path == f"/client/v4/accounts/{ACCOUNT}/ai/v1/chat/completions"
            return cloudflare_error(429, 4006, DAY_USED_UP)
        if request.url.host == "api.sambanova.ai":
            done["sambanova"] += 1
            return next_step(body["model"]) if done["sambanova"] <= 2 else samba_limit(0)
        if request.url.host == "gen.pollinations.ai":
            return notice(OUT_OF_CREDIT, model=body["model"])
        assert request.url.host == "api.deepseek.com" and body["thinking"] == {"type": "disabled"}
        return next_step(body["model"])

    install_deps(provider_calls(handler))
    _events, task = run_stream(client, TASK)
    assert task["status"] == "completed" and task["outcome"] == "pull_request" and not task.get("unfinished")
    assert task["tests_status"] == "passed" and [change["path"] for change in task["changed_files"]] == ["calc.py"]
    assert task["llm_chain"] == [
        "groq / openai/gpt-oss-120b",
        "cloudflare / @cf/zai-org/glm-4.7-flash",
        "sambanova / DeepSeek-V3.1",
        "pollinations / openai",
        "deepseek / deepseek-flash",
        "groq / openai/gpt-oss-20b",
        "cloudflare / @cf/google/gemma-4-26b-a4b-it",
        "sambanova / gpt-oss-120b",
        "pollinations / openai/gpt-oss-20b",
    ]
    # Each provider's first model in turn. Cloudflare's and Pollinations' second
    # models are not asked: what stopped the first goes for the account.
    assert asked == [
        "api.groq.com openai/gpt-oss-120b",
        "api.cloudflare.com @cf/zai-org/glm-4.7-flash",
        "api.sambanova.ai DeepSeek-V3.1",
        "api.sambanova.ai DeepSeek-V3.1",
        "api.sambanova.ai DeepSeek-V3.1",
        "gen.pollinations.ai openai",
        "api.deepseek.com deepseek-flash",
        "api.deepseek.com deepseek-flash",
    ]
    # What the task says about the models it asked - the ones that never were are not in it.
    assert [(item["provider"], item["model"], item["requests"], item["failures"], item["out"]) for item in task["llm_usage"]] == [
        ("groq", "openai/gpt-oss-120b", 0, 1, "its daily limit is used up"),
        ("cloudflare", "@cf/zai-org/glm-4.7-flash", 0, 1, "its daily limit is used up"),
        ("sambanova", "DeepSeek-V3.1", 2, 1, "its daily limit is used up"),
        ("pollinations", "openai", 0, 1, "its account has no credit left"),
        ("deepseek", "deepseek-flash", 2, 0, None),
    ]
    assert (task["llm_provider"], task["llm_model"]) == ("deepseek", "deepseek-flash")
    # The notice was a failure of the provider: it is in the log as one and nowhere in the conversation or the result.
    log = messages(task)
    assert any("pollinations did not answer the request" in line for line in log)
    assert any("switching to deepseek / deepseek-flash" in line for line in log)
    record = json.dumps(task) + json.dumps(github.pulls["acme/demo"][0])
    assert "Pollinations account" not in json.dumps(histories["api.deepseek.com"]) and "top up" not in github.pulls["acme/demo"][0]["body"]
    # The conversation DeepSeek was handed is the one SambaNova had: the same two steps, with their results.
    carried = [message for message in histories["api.deepseek.com"] if message["role"] in ("assistant", "tool")]
    assert [message["role"] for message in carried] == ["assistant", "tool", "assistant", "tool", "assistant", "tool"]
    # No key and no account ID appears anywhere in what the task leaves behind.
    for secret in ("cf-" + "c" * 40, ACCOUNT, "samba-" + "s" * 36, "sk_" + "p" * 40, "sk-" + "d" * 32):
        assert secret not in record


def test_keys_for_the_four_can_come_with_the_request(client: TestClient, app_env, install_deps, monkeypatch):
    deps = install_deps(fix_add_script())
    run_stream(
        client,
        TASK,
        headers={
            "X-Cloudflare-Key": f"{ACCOUNT}:cf-browser",
            "X-SambaNova-Key": "samba-browser",
            "X-Pollinations-Key": "sk_browser",
            "X-DeepSeek-Key": "sk-browser",
        },
    )
    seen = deps.settings_seen[0]
    assert (seen.cloudflare_account_id, seen.cloudflare_api_key) == (ACCOUNT, "cf-browser")
    assert (seen.sambanova_api_key, seen.pollinations_api_key, seen.deepseek_api_key) == ("samba-browser", "sk_browser", "sk-browser")
    assert seen.llm_chain == ["groq", "cloudflare", "sambanova", "pollinations", "deepseek"]
    # They were for that task: the server has none of its own.
    assert get_settings().llm_chain == ["groq"]


def test_trying_the_four_out_through_the_api(client: TestClient, app_env, monkeypatch):
    import apps.api.routers.llm as llm_router

    seen: List[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=chat_completion(json.loads(request.content)["model"], tool_calls=CONFIRM_CALL))

    original = llm_router.probe_provider
    monkeypatch.setattr(
        llm_router, "probe_provider", lambda settings, name, **kwargs: original(settings, name, transport=httpx.MockTransport(handler), **kwargs)
    )

    def try_out(name: str, headers: Optional[Dict[str, str]] = None, **body: str) -> Dict[str, Any]:
        seen.clear()
        response = client.post("/api/v1/llm/test", json={"provider": name, **body}, headers=headers or {})
        assert response.status_code == 200, response.text
        return response.json()

    for name, header, key, host in (
        ("sambanova", "X-SambaNova-Key", "samba-browser", "api.sambanova.ai"),
        ("pollinations", "X-Pollinations-Key", "sk_browser", "gen.pollinations.ai"),
        ("deepseek", "X-DeepSeek-Key", "sk-browser", "api.deepseek.com"),
    ):
        result = try_out(name, {header: key})
        assert (result["ok"], result["tool_call"], result["key_source"]) == (True, True, "request"), result
        assert {request.url.host for request in seen} == {host} and len(seen) == len(DEFAULT_MODELS[name])
        assert all(request.headers["authorization"] == f"Bearer {key}" for request in seen)
        assert try_out(name)["kind"] == "no_key"

    # Cloudflare: account ID and token in the one header.
    result = try_out("cloudflare", {"X-Cloudflare-Key": f"{ACCOUNT}:cf-browser"}, model="@cf/openai/gpt-oss-20b")
    assert (result["ok"], result["key_source"], result["model"]) == (True, "request", "@cf/openai/gpt-oss-20b")
    assert seen[0].url.path == f"/client/v4/accounts/{ACCOUNT}/ai/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer cf-browser"
    # The token without the account: nothing is sent, and it is said what is missing.
    result = try_out("cloudflare", {"X-Cloudflare-Key": "cf-browser"})
    assert (result["ok"], result["kind"], result["key_source"]) == (False, "no_account", "request") and seen == []
    assert try_out("cloudflare")["kind"] == "no_key"
    # The server holds the token, the browser brings the account - and the other way round.
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-server")
    assert try_out("cloudflare")["kind"] == "no_account"
    result = try_out("cloudflare", {"X-Cloudflare-Key": f"{ACCOUNT}:"})
    assert (result["ok"], result["key_source"]) == (True, "server") and seen[0].headers["authorization"] == "Bearer cf-server"
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", OTHER_ACCOUNT)
    result = try_out("cloudflare", {"X-Cloudflare-Key": "cf-browser"})
    assert (result["ok"], result["key_source"]) == (True, "request")
    assert seen[0].url.path == f"/client/v4/accounts/{OTHER_ACCOUNT}/ai/v1/chat/completions"
    assert try_out("cloudflare")["key_source"] == "server"
