"""Several providers behind one client, and the agent carrying on with the
next one when the provider in use reaches a limit or fails."""
from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from apps.api.services.agent import CodingAgent
from apps.api.services.common import Deadline
from apps.api.services.llm import (
    COOLDOWN_ROUTER_PICK,
    COOLDOWN_UNKNOWN_RATE_LIMIT,
    PATIENCE_ALONE,
    PATIENCE_UNPROVEN,
    FailoverClient,
    LLMError,
)
from apps.api.services.sandbox import RepoCheckout

from .conftest import DEMO_FILES
from .support import GitRemote, LocalWorkspace, ScriptedLLM

FINISH = [("finish", {"summary": "done", "pr_title": "Done"})]
READ = [("read_file", {"path": "calc.py"})]
GROQ_LIMIT = (
    "groq returned HTTP 429: Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` "
    "service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 7441, Requested 4788. "
    "Please try again in 31.7175s."
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def provider(name: str, steps: List[Any], model: str = "", **options: Any) -> ScriptedLLM:
    llm = ScriptedLLM(steps, model=model or f"{name}-model", **options)
    llm.provider = name
    return llm


def rate_limit(message: str = "slow down", retry_after: Any = 30.0, **options: Any) -> LLMError:
    return LLMError(message, kind="rate_limit", status_code=429, retry_after=retry_after, **options)


def ask(pool: FailoverClient) -> Any:
    return pool.complete([{"role": "user", "content": "hi"}], [], max_output_tokens=100, timeout=10)


def failure(pool: FailoverClient) -> LLMError:
    with pytest.raises(LLMError) as caught:
        ask(pool)
    return caught.value


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #
def test_rate_limit_hands_over_to_the_next_provider_and_stays_there():
    clock = Clock()
    groq = provider("groq", ["one", rate_limit(GROQ_LIMIT, 31.7), "never asked again"])
    gemini = provider("gemini", ["two", "three", "four"])
    pool = FailoverClient([groq, gemini], clock=clock)
    assert pool.chain == ["groq / groq-model", "gemini / gemini-model"]
    assert (ask(pool).content, pool.provider) == ("one", "groq")

    error = failure(pool)
    assert error.switched_to == "gemini / gemini-model" and error.provider == "groq" and error.kind == "rate_limit"
    assert (pool.provider, pool.model) == ("gemini", "gemini-model")
    assert ask(pool).content == "two" and ask(pool).provider == "gemini"

    # Groq's limit has lifted long ago; a provider that works is not left for that.
    clock.sleep(600)
    assert (ask(pool).content, pool.provider) == ("four", "gemini")
    assert pool.switches == 1 and len(groq.requests) == 2
    assert [(item["provider"], item["requests"], item["failures"]) for item in pool.usage()] == [("groq", 1, 1), ("gemini", 3, 0)]


def test_when_the_fallback_is_limited_too_the_first_provider_is_used_again_once_it_is_free():
    clock = Clock()
    groq = provider("groq", [rate_limit(retry_after=20.0), "groq again"])
    gemini = provider("gemini", ["gemini", rate_limit(retry_after=45.0)])
    pool = FailoverClient([groq, gemini], clock=clock)
    assert failure(pool).switched_to == "gemini / gemini-model"
    assert ask(pool).content == "gemini"
    clock.sleep(25)  # past groq's 20 seconds
    assert failure(pool).switched_to == "groq / groq-model"
    assert ask(pool).content == "groq again"


def test_all_providers_limited_the_shortest_wait_is_reported_and_its_provider_answers_next():
    clock = Clock()
    groq = provider("groq", [rate_limit(retry_after=40.0), "not reached"])
    gemini = provider("gemini", [rate_limit(retry_after=8.0), "after the wait"])
    openrouter = provider("openrouter", [rate_limit(retry_after=25.0)])
    pool = FailoverClient([groq, gemini, openrouter], clock=clock)
    assert failure(pool).switched_to == "gemini / gemini-model"
    assert failure(pool).switched_to == "openrouter / openrouter-model"
    error = failure(pool)  # nobody is ready now
    assert error.switched_to is None and error.kind == "rate_limit"
    assert error.retry_after == pytest.approx(8.0) and error.next_provider == "gemini / gemini-model"
    assert pool.provider == "gemini"
    # The repeated request goes to that provider - whether or not the caller's clock is the pool's.
    assert ask(pool).content == "after the wait"


def test_a_single_limited_provider_reports_its_own_wait():
    pool = FailoverClient([provider("groq", [rate_limit(retry_after=None)])], clock=Clock())
    error = failure(pool)
    assert error.switched_to is None and error.next_provider is None
    assert error.retry_after == COOLDOWN_UNKNOWN_RATE_LIMIT


@pytest.mark.parametrize("kind", ["auth", "quota", "model_unavailable", "bad_request", "blocked"])
def test_a_provider_that_cannot_be_used_is_out_for_the_rest_of_the_run(kind):
    clock = Clock()
    groq = provider("groq", [LLMError("groq says no", kind=kind)])
    gemini = provider("gemini", [rate_limit(retry_after=5.0), "back"])
    openrouter = provider("openrouter", ["or", rate_limit(retry_after=30.0)])
    pool = FailoverClient([groq, gemini, openrouter], clock=clock)
    assert failure(pool).switched_to == "gemini / gemini-model"
    assert failure(pool).switched_to == "openrouter / openrouter-model"
    assert ask(pool).content == "or"
    clock.sleep(3600)
    # groq is first in the chain and would be "ready", but it is out.
    assert failure(pool).switched_to == "gemini / gemini-model"
    assert ask(pool).content == "back"
    assert pool.usage()[0]["out"] is not None and len(groq.requests) == 1


def test_daily_limit_takes_a_provider_out_even_if_it_names_a_short_wait():
    clock = Clock()
    gemini = provider("gemini", [rate_limit("quota per day used up", retry_after=12.0, daily=True)])
    groq = provider("groq", ["a", rate_limit(retry_after=10.0), "b"])
    pool = FailoverClient([gemini, groq], clock=clock)
    assert failure(pool).switched_to == "groq / groq-model"
    assert ask(pool).content == "a"
    clock.sleep(60)
    error = failure(pool)  # gemini's 12 seconds are over, but its day is not
    assert error.switched_to is None and error.next_provider is None and error.retry_after == pytest.approx(10.0)
    assert ask(pool).content == "b"


def test_fatal_failure_while_the_others_are_only_limited_becomes_a_wait():
    clock = Clock()
    groq = provider("groq", [rate_limit(retry_after=15.0), "after the wait"])
    gemini = provider("gemini", [LLMError("gemini returned HTTP 400: API key not valid.", kind="auth", status_code=400)])
    pool = FailoverClient([groq, gemini], clock=clock)
    assert failure(pool).switched_to == "gemini / gemini-model"
    error = failure(pool)
    assert error.kind == "rate_limit" and error.switched_to is None
    assert error.retry_after == pytest.approx(15.0) and error.next_provider == "groq / groq-model"
    assert error.message == "gemini returned HTTP 400: API key not valid. groq is used next; it is rate limited for a moment."
    assert ask(pool).content == "after the wait"


def test_fatal_failure_of_the_only_provider_left_stays_fatal():
    pool = FailoverClient(
        [provider("groq", [LLMError("no", kind="auth")]), provider("gemini", [LLMError("no either", kind="quota")])],
        clock=Clock(),
    )
    assert failure(pool).switched_to == "gemini / gemini-model"
    error = failure(pool)
    assert (error.kind, error.switched_to, error.next_provider) == ("quota", None, None)


def test_one_refused_reply_is_retried_by_the_same_model_two_in_a_row_are_not():
    refused = lambda: LLMError("refused", kind="tool_format", status_code=400)  # noqa: E731
    groq = provider("groq", [refused(), "fine", refused(), refused()])
    gemini = provider("gemini", ["from gemini"])
    pool = FailoverClient([groq, gemini], clock=Clock())
    assert failure(pool).switched_to is None
    assert ask(pool).content == "fine"  # a good reply in between resets the count
    assert failure(pool).switched_to is None
    assert failure(pool).switched_to == "gemini / gemini-model"
    assert ask(pool).content == "from gemini"


@pytest.mark.parametrize("kind", ["server", "network", "too_large"])
def test_outage_or_oversized_request_moves_on_when_someone_is_ready_and_is_left_to_the_caller_otherwise(kind):
    clock = Clock()
    groq = provider("groq", [LLMError("groq is down", kind=kind), "groq is back"])
    gemini = provider("gemini", [rate_limit(retry_after=50.0)])
    pool = FailoverClient([groq, gemini], clock=clock)
    assert failure(pool).switched_to == "gemini / gemini-model"
    clock.sleep(70)  # past the pause after an outage or an oversized request
    assert failure(pool).switched_to == "groq / groq-model"
    assert ask(pool).content == "groq is back"

    lonely = FailoverClient([provider("groq", [LLMError("down", kind=kind)]), provider("gemini", [rate_limit(retry_after=50.0)])], clock=Clock())
    assert failure(lonely).switched_to == "gemini / gemini-model"
    again = FailoverClient([provider("gemini", [rate_limit(retry_after=50.0)]), provider("groq", [LLMError("down", kind=kind)])], clock=Clock())
    assert failure(again).switched_to == "groq / groq-model"
    error = failure(again)  # gemini is limited, groq failed: the caller retries or shrinks as it sees fit
    assert (error.kind, error.switched_to, error.next_provider, again.provider) == (kind, None, None, "groq")


def test_pool_needs_a_provider_and_closes_all_of_them():
    with pytest.raises(LLMError):
        FailoverClient([])
    members = [provider("groq", []), provider("gemini", [])]
    FailoverClient(members).close()
    assert all(member.closed for member in members)


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #
@pytest.fixture
def checkout(remote: GitRemote, tmp_path) -> RepoCheckout:
    remote.create("acme/demo", DEMO_FILES)
    workspace = LocalWorkspace(tmp_path / "sandbox")
    return RepoCheckout.clone(workspace, clone_url=remote.clone_url("acme/demo"), branch="main")


def make_agent(checkout: RepoCheckout, llm: Any, **options: Any) -> Tuple[CodingAgent, List[Tuple[str, str]], List[float]]:
    events: List[Tuple[str, str]] = []
    waits: List[float] = []
    options.setdefault("deadline", Deadline(300))
    options.setdefault("sleep", waits.append)
    agent = CodingAgent(llm=llm, checkout=checkout, emit=lambda level, message: events.append((level, message)), **options)
    return agent, events, waits


def run(agent: CodingAgent):
    return agent.run(prompt="Fix add()", repository="acme/demo", branch="main", file_listing="Repository files:\ncalc.py")


def test_agent_carries_on_with_the_next_provider_instead_of_waiting(checkout: RepoCheckout):
    """The run from production: Groq's 8,000 tokens per minute were used up and
    the task waited minutes in total, then failed. With a second provider the
    agent does not wait at all."""
    groq = provider("groq", [READ, rate_limit(GROQ_LIMIT, 31.7175)], model="openai/gpt-oss-120b", tokens_per_minute=8000)
    gemini = provider("gemini", [READ, FINISH], model="gemini-flash-latest")
    gemini.assumed_tokens_per_minute = 250_000
    agent, events, waits = make_agent(checkout, FailoverClient([groq, gemini], clock=Clock()))
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 3 and outcome.model == "gemini-flash-latest"
    assert waits == []  # no pause anywhere
    assert ("info", "groq's rate limit is reached (8,000 tokens per minute): switching to gemini / gemini-flash-latest") in events
    assert not [message for _level, message in events if message.startswith("Waiting")]
    # Each provider is asked for what its own limits allow.
    assert groq.output_token_limits == [4096, 2400] and gemini.output_token_limits == [8192, 8192]
    # The conversation continues where it was: Gemini sees what Groq read.
    assert [message["role"] for message in gemini.requests[0]] == ["system", "user", "assistant", "tool"]


def test_agent_waits_only_when_every_provider_is_limited(checkout: RepoCheckout):
    clock = Clock()
    groq = provider("groq", [rate_limit(retry_after=40.0)])
    gemini = provider("gemini", [rate_limit(retry_after=8.0), FINISH])
    agent, events, _waits = make_agent(checkout, FailoverClient([groq, gemini], clock=clock), sleep=clock.sleep)
    assert run(agent).finished
    waiting = [message for _level, message in events if message.startswith("Waiting")]
    assert waiting == ["Waiting 8s for gemini's rate limit (every provider is rate limited)"]
    assert clock.now == pytest.approx(1008.5)


def test_agent_says_why_it_changed_providers(checkout: RepoCheckout):
    groq = provider("groq", [LLMError("groq returned HTTP 401: Invalid API Key. Check the GROQ API key.", kind="auth", status_code=401)])
    gemini = provider("gemini", [LLMError("gemini returned HTTP 503 (UNAVAILABLE): The model is overloaded.", kind="server", status_code=503)])
    openrouter = provider("openrouter", [FINISH])
    agent, events, _waits = make_agent(checkout, FailoverClient([groq, gemini, openrouter], clock=Clock()))
    assert run(agent).finished
    assert [event for event in events if "switching to" in event[1]] == [
        ("warn", "groq returned HTTP 401: Invalid API Key. Check the GROQ API key: switching to gemini / gemini-model"),
        ("warn", "gemini returned HTTP 503 (UNAVAILABLE): The model is overloaded: switching to openrouter / openrouter-model"),
    ]


def test_model_that_keeps_writing_invalid_calls_is_replaced_and_the_new_one_is_not_blamed(checkout: RepoCheckout):
    refused = lambda: LLMError(  # noqa: E731
        "groq refused the model's reply because it was not a valid tool call (it tried to call a tool named 'commentary', which does not exist).",
        kind="tool_format",
        status_code=400,
        rejected_tool="commentary",
    )
    groq = provider("groq", [refused(), refused()])
    gemini = provider("gemini", [FINISH])
    agent, events, _waits = make_agent(checkout, FailoverClient([groq, gemini], clock=Clock()))
    assert run(agent).finished
    assert ("warn", "The model's reply was not a valid tool call (it called 'commentary', which is not one of its tools); asking it again (1/3)") in events
    assert (
        "warn",
        "groq's model keeps answering with invalid tool calls (it called 'commentary', which is not one of its tools): switching to gemini / gemini-model",
    ) in events
    # The second request to groq carried the correction; the one to gemini does not:
    # it was about a reply gemini never wrote.
    assert "Your previous reply was rejected" in groq.requests[1][-1]["content"]
    assert all("Your previous reply was rejected" not in str(message.get("content")) for message in gemini.requests[0])


def test_request_too_large_for_one_provider_goes_to_the_next_without_losing_context(checkout: RepoCheckout):
    groq = provider("groq", [LLMError("groq returned HTTP 413: Request too large for model: Limit 8000, Requested 9000", kind="too_large", status_code=413)])
    gemini = provider("gemini", [FINISH])
    gemini.assumed_tokens_per_minute = 250_000
    agent, events, _waits = make_agent(checkout, FailoverClient([groq, gemini], clock=Clock()))
    assert run(agent).finished
    assert ("info", "The request is larger than groq's limits allow: switching to gemini / gemini-model") in events
    assert not [message for _level, message in events if "retrying with less context" in message]
    assert agent.context_budget == 160_000


def test_without_a_fallback_the_failure_says_how_to_get_one(checkout: RepoCheckout):
    groq = provider("groq", [rate_limit(GROQ_LIMIT, 600.0)])
    agent, _events, _waits = make_agent(checkout, groq)
    outcome = run(agent)
    assert outcome.stop_reason == "llm_error"
    assert outcome.error.endswith(
        "The provider's rate limit left no room to finish within this run's time budget. Add a key for a "
        "second provider (Gemini or OpenRouter, in Settings): the agent then carries on with it instead of waiting."
    )

    # With fallbacks configured and all of them exhausted, the advice would be noise.
    pool = FailoverClient([provider("groq", [rate_limit(retry_after=600.0)]), provider("gemini", [rate_limit(retry_after=900.0)])], clock=Clock())
    agent, _events, _waits = make_agent(checkout, pool)
    assert run(agent).error.endswith("left no room to finish within this run's time budget.")


def test_daily_limit_of_the_only_provider_fails_at_once(checkout: RepoCheckout):
    gemini = provider("gemini", [rate_limit("gemini returned HTTP 429 (RESOURCE_EXHAUSTED): You exceeded your current quota.", 12.0, daily=True)])
    agent, _events, waits = make_agent(checkout, gemini)
    outcome = run(agent)
    assert outcome.stop_reason == "llm_error" and waits == []
    assert "That is a daily limit: it does not lift during a run." in outcome.error


def test_recovered_reply_is_logged_and_executed(checkout: RepoCheckout):
    from apps.api.services.llm import LLMResponse, ToolCall

    class Recovering(ScriptedLLM):
        def complete(self, messages, tools, *, max_output_tokens, timeout):
            response = super().complete(messages, tools, max_output_tokens=max_output_tokens, timeout=timeout)
            if len(self.requests) == 1:
                call = ToolCall(id="call_recovered_1", name="read_file", arguments={"path": "calc.py"})
                return LLMResponse(
                    content="",
                    tool_calls=[call],
                    finish_reason="tool_calls",
                    model=self.model,
                    assistant_message={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": call.id, "type": "function", "function": {"name": "read_file", "arguments": '{"path": "calc.py"}'}}],
                    },
                    provider="groq",
                    recovered=True,
                )
            return response

    llm = Recovering([READ, FINISH])
    llm.provider = "groq"
    agent, events, _waits = make_agent(checkout, llm)
    assert run(agent).finished and agent.files_read == ["calc.py"]
    assert ("info", "groq refused the model's reply over its form; the call in it was used as meant (read_file)") in events
    assert len(llm.requests) == 2  # nothing was asked for again


# --------------------------------------------------------------------------- #
# Several models of one provider
# --------------------------------------------------------------------------- #
def test_a_second_model_of_the_same_provider_has_its_own_limit():
    """Limits are counted per model: when one is used up, the provider's next
    model carries on - before any waiting."""
    clock = Clock()
    big = provider("groq", ["one", rate_limit(GROQ_LIMIT, 31.7), "big again"], model="gpt-oss-120b")
    small = provider("groq", ["two", rate_limit(retry_after=50.0)], model="gpt-oss-20b")
    pool = FailoverClient([big, small], clock=clock)
    assert pool.chain == ["groq / gpt-oss-120b", "groq / gpt-oss-20b"]
    assert ask(pool).content == "one"
    error = failure(pool)
    assert error.switched_to == "groq / gpt-oss-20b" and (pool.provider, pool.model) == ("groq", "gpt-oss-20b")
    assert ask(pool).content == "two"
    clock.sleep(35)  # the first model's limit has lifted when the second one's is reached
    assert failure(pool).switched_to == "groq / gpt-oss-120b"
    assert ask(pool).content == "big again"
    assert [(item["model"], item["requests"], item["failures"]) for item in pool.usage()] == [
        ("gpt-oss-120b", 2, 1),
        ("gpt-oss-20b", 1, 1),
    ]


def test_a_rejected_key_takes_out_every_model_of_that_provider():
    clock = Clock()
    first = provider("openrouter", [LLMError("openrouter returned HTTP 401: No auth credentials found", kind="auth", status_code=401)], model="some:free")
    groq = provider("groq", ["groq answers", "and again"])
    router = provider("openrouter", ["never asked"], model="openrouter/free")
    pool = FailoverClient([first, groq, router], clock=clock)
    assert failure(pool).switched_to == "groq / groq-model"
    assert ask(pool).content == "groq answers"
    assert [(item["model"], item["out"]) for item in pool.usage()] == [
        ("some:free", "its API key was rejected"),
        ("groq-model", None),
        ("openrouter/free", "its API key was rejected"),
    ]
    assert router.requests == []


def test_a_model_that_is_gone_leaves_the_providers_other_model_in_use():
    clock = Clock()
    gone = provider("gemini", [LLMError("gemini returned HTTP 404: models/x is not found", kind="model_unavailable", status_code=404)], model="x")
    lite = provider("gemini", ["lite answers"], model="gemini-3.5-flash-lite")
    pool = FailoverClient([gone, lite], clock=clock)
    assert failure(pool).switched_to == "gemini / gemini-3.5-flash-lite"
    assert ask(pool).content == "lite answers"


def test_the_model_a_router_chose_is_reported():
    """`openrouter/free` picks a model per request; which ones answered is part of the result."""

    class Router(ScriptedLLM):
        served = ["upstage/solar-pro-3:free", "upstage/solar-pro-3:free", "google/gemma-4-31b-it:free"]

        def complete(self, messages, tools, *, max_output_tokens, timeout):
            response = super().complete(messages, tools, max_output_tokens=max_output_tokens, timeout=timeout)
            response.model = self.served[len(self.requests) - 1]
            return response

    router = Router(["a", "b", "c"], model="openrouter/free")
    router.provider = "openrouter"
    pool = FailoverClient([router, provider("groq", [])], clock=Clock())
    for _ in range(3):
        ask(pool)
    assert pool.usage()[0]["served"] == {"upstage/solar-pro-3:free": 2, "google/gemma-4-31b-it:free": 1}
    assert "served" not in pool.usage()[1]


UPSTREAM = (
    "openrouter returned HTTP 429: Provider returned error (google/gemma-4-31b-it:free is temporarily "
    "rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits)"
)
FREE_PER_DAY = (
    "openrouter returned HTTP 429: Rate limit exceeded: free-models-per-day. Add 10 credits to unlock "
    "1000 free model requests per day"
)


def test_a_router_whose_pick_is_throttled_is_asked_again_shortly():
    """`openrouter/free` picks a model for every request. One that is throttled
    upstream says nothing about the next pick: the router is free again in a moment."""
    clock = Clock()
    router = provider("openrouter", [rate_limit(UPSTREAM, None), "another model answers"], model="openrouter/free")
    pool = FailoverClient([router], clock=clock)
    error = failure(pool)
    assert error.kind == "rate_limit" and error.retry_after == COOLDOWN_ROUTER_PICK == 3.0 and not error.switched_to
    clock.sleep(COOLDOWN_ROUTER_PICK)
    assert ask(pool).content == "another model answers"

    # With somebody else ready the request goes there at once - and the router is not out for long.
    clock = Clock()
    router = provider("openrouter", [rate_limit(UPSTREAM, None), "the router again"], model="openrouter/free")
    groq = provider("groq", ["groq answers", rate_limit(GROQ_LIMIT, 31.7)])
    pool = FailoverClient([router, groq], clock=clock)
    assert failure(pool).switched_to == "groq / groq-model"
    assert ask(pool).content == "groq answers"
    clock.sleep(5)
    assert failure(pool).switched_to == "openrouter / openrouter/free"
    assert ask(pool).content == "the router again"


def test_a_single_model_that_is_throttled_upstream_is_left_alone_for_a_while():
    clock = Clock()
    pinned = provider("openrouter", [rate_limit(UPSTREAM, None)], model="google/gemma-4-31b-it:free")
    pool = FailoverClient([pinned], clock=clock)
    assert failure(pool).retry_after == COOLDOWN_UNKNOWN_RATE_LIMIT
    # A limit of the account, which names when it lifts, is honoured for a router too.
    router = provider("openrouter", [rate_limit("openrouter returned HTTP 429: Rate limit exceeded: free-models-per-min.", 20.0)], model="openrouter/free")
    assert failure(FailoverClient([router], clock=clock)).retry_after == 20.0


def test_openrouters_daily_allowance_is_one_for_all_its_free_models():
    clock = Clock()
    pinned = provider("openrouter", [rate_limit(FREE_PER_DAY, None, daily=True)], model="baidu/ernie-5.5-flash-preview:free")
    groq = provider("groq", ["groq answers", rate_limit(GROQ_LIMIT, 31.7)])
    router = provider("openrouter", ["never asked"], model="openrouter/free")
    paid = provider("openrouter", ["a paid model answers"], model="openai/gpt-5.5-mini")
    pool = FailoverClient([pinned, groq, router, paid], clock=clock)
    assert failure(pool).switched_to == "groq / groq-model"
    assert ask(pool).content == "groq answers"
    # The router would be refused for the same reason: it is not asked. A paid model is another matter.
    assert failure(pool).switched_to == "openrouter / openai/gpt-5.5-mini"
    assert ask(pool).content == "a paid model answers" and router.requests == []
    assert [(item["model"], item["out"]) for item in pool.usage()] == [
        ("baidu/ernie-5.5-flash-preview:free", "its daily limit is used up"),
        ("groq-model", None),
        ("openrouter/free", "its daily limit is used up"),
        ("openai/gpt-5.5-mini", None),
    ]
    # Another provider's daily limit is counted per model.
    big = provider("groq", [rate_limit("groq returned HTTP 429: tokens per day (TPD): Limit 200000", 600.0, daily=True)], model="openai/gpt-oss-120b")
    small = provider("groq", ["the smaller model answers"], model="openai/gpt-oss-20b")
    pool = FailoverClient([big, small], clock=clock)
    assert failure(pool).switched_to == "groq / openai/gpt-oss-20b" and ask(pool).content == "the smaller model answers"


def test_the_agent_asks_the_router_again_when_its_pick_was_throttled(checkout: RepoCheckout):
    clock = Clock()
    router = provider("openrouter", [READ, rate_limit(UPSTREAM, None), rate_limit(UPSTREAM, None), FINISH], model="openrouter/free")
    agent, events, _waits = make_agent(checkout, FailoverClient([router], clock=clock), sleep=clock.sleep)
    outcome = run(agent)
    assert outcome.finished and outcome.turns == 2
    waiting = [message for _level, message in events if message.startswith("Waiting")]
    assert waiting == ["Waiting 4s for openrouter's rate limit"] * 2


# --------------------------------------------------------------------------- #
# How long a model is waited for
# --------------------------------------------------------------------------- #
class Timed(ScriptedLLM):
    """Every answer takes ``seconds`` of the pool's clock."""

    def __init__(self, steps, clock: Clock, seconds: float, **options: Any) -> None:
        super().__init__(steps, **options)
        self.clock, self.seconds = clock, seconds

    def complete(self, messages, tools, *, max_output_tokens, timeout):
        self.clock.now += self.seconds
        return super().complete(messages, tools, max_output_tokens=max_output_tokens, timeout=timeout)


def test_patience_with_a_model_depends_on_what_it_has_shown():
    """Production, 2026-11-12: Gemini accepted a request and did not answer.
    The run waited 90 of its 270 seconds for it - while another provider was
    free to take over."""
    clock = Clock()
    slow = Timed(["one", "two", rate_limit(retry_after=30.0)], clock, 20.0, model="slow-model")
    slow.provider = "openrouter"
    other = provider("gemini", ["three"])
    pool = FailoverClient([slow, other], clock=clock)
    # It has not answered yet in this run, and another provider is ready.
    assert pool.patience() == PATIENCE_UNPROVEN == 45.0
    ask(pool)
    # It answers, and takes 20 seconds to: three times that plus a margin is still plausible.
    assert pool.patience() == 70.0
    ask(pool)
    assert failure(pool).switched_to == "gemini / gemini-model"
    # The one in use now has shown nothing yet; the other is free again in 30 seconds.
    # Giving up before that would only add a wait, so: until then, and a little longer.
    assert pool.patience() == 45.0
    clock.sleep(10)
    assert pool.patience() == 45.0  # free in 20 s: 35 s would be enough, the 45 s stay
    # With nobody else to turn to, waiting is all there is.
    assert FailoverClient([provider("groq", [])], clock=clock).patience() == PATIENCE_ALONE == 90.0
    lonely = FailoverClient(
        [provider("groq", [LLMError("groq returned HTTP 401", kind="auth", status_code=401)]), provider("gemini", [])],
        clock=clock,
    )
    failure(lonely)
    assert lonely.provider == "gemini" and lonely.patience() == PATIENCE_ALONE


def test_patience_never_exceeds_the_longest_wait():
    clock = Clock()
    slow = Timed(["one"], clock, 60.0)
    slow.provider = "openrouter"
    pool = FailoverClient([slow, provider("gemini", [rate_limit(retry_after=200.0)])], clock=clock)
    ask(pool)
    assert pool.patience() == PATIENCE_ALONE  # 3 x 60 s would be more


def test_the_agent_waits_no_longer_than_the_pool_advises(remote: GitRemote, tmp_path):
    remote.create("acme/demo", DEMO_FILES)
    checkout = RepoCheckout.clone(LocalWorkspace(tmp_path / "sandbox"), clone_url=remote.clone_url("acme/demo"), branch="main")
    timeouts: List[float] = []

    class Recording(ScriptedLLM):
        def complete(self, messages, tools, *, max_output_tokens, timeout):
            timeouts.append(timeout)
            return super().complete(messages, tools, max_output_tokens=max_output_tokens, timeout=timeout)

    first = Recording([READ, FINISH])
    first.provider = "openrouter"
    pool = FailoverClient([first, provider("groq", [])])
    agent = CodingAgent(llm=pool, checkout=checkout, emit=lambda _level, _message: None, deadline=Deadline(300), sleep=lambda _s: None)
    assert agent.run(prompt="Look", repository="acme/demo", branch="main").finished
    # First request: nothing is known about the model, and another one is ready.
    assert timeouts[0] == 45.0 and 45.0 <= timeouts[1] <= 90.0
    # A single provider is given the full time.
    alone = Recording([READ, FINISH])
    timeouts.clear()
    agent = CodingAgent(llm=alone, checkout=checkout, emit=lambda _level, _message: None, deadline=Deadline(300), sleep=lambda _s: None)
    agent.run(prompt="Look", repository="acme/demo", branch="main")
    assert timeouts == [90.0, 90.0]
