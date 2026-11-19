"""Sandbox network settings: host list, network mode, limits."""
from __future__ import annotations

from apps.api.config import (
    clean_hosts,
    get_settings,
    SANDBOX_HOSTS,
    SANDBOX_NETWORK_MODES,
)


def _minimal_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("E2B_API_KEY", "e2b_test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")


def test_sandbox_network_modes_are_valid(monkeypatch):
    _minimal_env(monkeypatch)
    for mode in SANDBOX_NETWORK_MODES:
        monkeypatch.setenv("JULES_SANDBOX_NETWORK", mode)
        assert get_settings().sandbox_network == mode


def test_sandbox_network_env_aliases(monkeypatch):
    _minimal_env(monkeypatch)

    monkeypatch.setenv("JULES_SANDBOX_NETWORK", "limited")
    assert get_settings().sandbox_network == "restricted"

    monkeypatch.setenv("JULES_SANDBOX_NETWORK", "restrict")
    assert get_settings().sandbox_network == "restricted"

    monkeypatch.setenv("JULES_SANDBOX_NETWORK", "closed")
    assert get_settings().sandbox_network == "restricted"

    monkeypatch.setenv("JULES_SANDBOX_NETWORK", "unrestricted")
    assert get_settings().sandbox_network == "open"

    monkeypatch.setenv("JULES_SANDBOX_NETWORK", "bogus")
    assert get_settings().sandbox_network == "auto"


def test_clean_hosts_parses_and_validates():
    hosts = clean_hosts("registry.npmjs.org, *.example.com, 10.0.0.0/8")
    assert hosts == ("registry.npmjs.org", "*.example.com", "10.0.0.0/8")

    # Wildcard "*" and "0.0.0.0/0" are rejected
    hosts = clean_hosts("*, 0.0.0.0/0, registry.npmjs.org")
    assert hosts == ("registry.npmjs.org",)

    # Case-insensitive, deduplicated
    hosts = clean_hosts("REGISTRY.NPMJS.ORG, registry.npmjs.org")
    assert hosts == ("registry.npmjs.org",)

    # Empty / whitespace / None
    assert clean_hosts("") == ()
    assert clean_hosts("   ") == ()
    assert clean_hosts(None) == ()


def test_sandbox_hosts_includes_extra_hosts(monkeypatch):
    _minimal_env(monkeypatch)
    monkeypatch.setenv("JULES_SANDBOX_ALLOWED_HOSTS", "example.com, github.com")
    s = get_settings()
    hosts = s.sandbox_hosts
    assert "example.com" in hosts
    # github.com is already in SANDBOX_HOSTS, not duplicated
    assert hosts.count("github.com") == 1
    assert all(h in SANDBOX_HOSTS or h == "example.com" for h in hosts)


def test_limits_sandbox_for(monkeypatch):
    _minimal_env(monkeypatch)

    monkeypatch.setenv("JULES_SANDBOX_NETWORK", "restricted")
    s = get_settings()
    assert s.limits_sandbox_for(outside_text=False) is True
    assert s.limits_sandbox_for(outside_text=True) is True

    monkeypatch.setenv("JULES_SANDBOX_NETWORK", "open")
    s = get_settings()
    assert s.limits_sandbox_for(outside_text=False) is False
    assert s.limits_sandbox_for(outside_text=True) is False

    monkeypatch.setenv("JULES_SANDBOX_NETWORK", "auto")
    s = get_settings()
    assert s.limits_sandbox_for(outside_text=False) is False
    assert s.limits_sandbox_for(outside_text=True) is True
