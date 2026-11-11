"""Small shared helpers: time budgets, secret redaction and text clipping."""
from __future__ import annotations

import re
import time
from typing import Callable, Iterable, List

# (level, message) - how every component reports progress to the task log.
Emit = Callable[[str, str], None]

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\r")
_TOKEN_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"(x-access-token:)[^@\s]+@", re.IGNORECASE),
    re.compile(r"(authorization:\s*(?:basic|bearer|token)\s+)[A-Za-z0-9+/=._-]{8,}", re.IGNORECASE),
)


class Deadline:
    """A wall-clock budget for one task run."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self._end = self._started + max(0.0, seconds)
        self.budget = max(0.0, seconds)

    def remaining(self) -> float:
        return max(0.0, self._end - self._clock())

    def elapsed(self) -> float:
        return self._clock() - self._started

    def expired(self) -> bool:
        return self.remaining() <= 0.0


class Redactor:
    """Masks credentials before text reaches logs, the UI or a pull request."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets: List[str] = []
        self.add(*secrets)

    def add(self, *secrets: str) -> None:
        for secret in secrets:
            if secret and len(secret) >= 8 and secret not in self._secrets:
                self._secrets.append(secret)
        self._secrets.sort(key=len, reverse=True)

    def __call__(self, text: str) -> str:
        if not text:
            return text
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, "***")
        text = _TOKEN_PATTERNS[0].sub("***", text)
        text = _TOKEN_PATTERNS[1].sub("***", text)
        text = _TOKEN_PATTERNS[2].sub(r"\1***@", text)
        text = _TOKEN_PATTERNS[3].sub(r"\1***", text)
        return text


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def clip(text: str, limit: int, *, tail_share: float = 0.7) -> str:
    """Shorten ``text`` to about ``limit`` characters, keeping the start and
    (mostly) the end - the end of command output is usually the useful part."""
    if limit <= 0 or len(text) <= limit:
        return text
    tail = int(limit * tail_share)
    head = max(0, limit - tail)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[... {omitted} characters omitted ...]\n{text[-tail:] if tail else ''}"


def first_line(text: str, limit: int = 72) -> str:
    """The first non-empty line of ``text``, shortened for use as a title."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"
    return ""
