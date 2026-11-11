"""What the result of a test command means.

A non-zero exit code is not always a failing test. ``npm test`` in a project
created with ``npm init`` runs the placeholder script
``echo "Error: no test specified" && exit 1``: it exits with 1 although there
is nothing that could have failed. Reporting that as "the tests failed" is as
wrong as reporting "the tests passed" for a run that executed zero tests, so
both are recognised here and reported as what they are: *no tests to run*.
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

# A run that had nothing to test says so in a few lines. Longer output is never
# classified as "no tests": it may contain real results (for example from other
# packages of a workspace), and hiding a real failure is the worse mistake.
_MAX_NO_TESTS_OUTPUT_LINES = 40

_MISSING_SCRIPT_RES = (
    re.compile(r'missing script:?\s*"?test"?(?![\w:.-])', re.IGNORECASE),  # npm, pnpm
    re.compile(r'Command "test" not found'),  # yarn 1
    re.compile(r'Couldn\'t find a script named "test"'),  # yarn 2+
)
_NO_TEST_FILES_RES = (
    re.compile(r"No tests found, exiting with code \d"),  # jest
    re.compile(r"No test files found"),  # vitest, mocha
    re.compile(r"No tests executed!"),  # phpunit
    re.compile(r"^0 examples, 0 failures", re.MULTILINE),  # rspec
    re.compile(r"^(?:#|ℹ) tests 0\s*$", re.MULTILINE),  # node --test
)
_PYTEST_NOTHING_RE = re.compile(r"no tests ran|collected 0 items")
_UNITTEST_NOTHING_RE = re.compile(r"^Ran 0 tests in ", re.MULTILINE)
_MAKE_NO_TARGET_RE = re.compile(r"No rule to make target [`']test'")
_GO_RESULT_RE = re.compile(r"^(?:ok|FAIL|---|panic)\b", re.MULTILINE)
_CARGO_RUNNING_RE = re.compile(r"^running (\d+) tests?$", re.MULTILINE)
# Signs that tests were executed after all. With one of them in the output a run
# is judged by its exit code whatever else it says: in a workspace one package
# can have the placeholder script while another has real, failing tests.
_REAL_RESULT_RES = (
    # "3 passing", "1 failing" (mocha); "1 failed, 2 passed" (jest, vitest, pytest)
    re.compile(r"\b[1-9]\d*\s+(?:tests?\s+|specs?\s+|examples?\s+)?(?:passing|failing|passed|failed)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:not ok|ok)\s+\d+\b", re.MULTILINE),  # TAP: "not ok 2 - adds"
    re.compile(r"^\s*(?:PASS|FAIL|FAILED)\s+\S", re.MULTILINE),  # per-file / per-test result lines
    re.compile(r"\bAssertionError\b|^Traceback \(most recent call last\)", re.MULTILINE),
)


def no_tests_reason(exit_code: int, output: str) -> str:
    """Why a test command had nothing to run, or ``""`` if it did run tests.

    Only the tools' own, unambiguous messages are recognised. Anything else is
    judged by its exit code as before.
    """
    text = output or ""

    # Tools that report per package or per target: decided by their own counts.
    if "[no test files]" in text and not _GO_RESULT_RE.search(text):
        return "the Go packages have no test files"
    cargo_counts = _CARGO_RUNNING_RE.findall(text)
    if cargo_counts and exit_code == 0 and all(count == "0" for count in cargo_counts):
        return "cargo found no tests to run"

    if sum(1 for line in text.splitlines() if line.strip()) > _MAX_NO_TESTS_OUTPUT_LINES:
        return ""
    if any(pattern.search(text) for pattern in _REAL_RESULT_RES):
        return ""
    if "Error: no test specified" in text:
        return "the \"test\" script in package.json is npm's placeholder"
    if any(pattern.search(text) for pattern in _MISSING_SCRIPT_RES):
        return 'package.json has no "test" script'
    if any(pattern.search(text) for pattern in _NO_TEST_FILES_RES):
        return "the test runner found no tests"
    if exit_code == 5 and _PYTEST_NOTHING_RE.search(text):
        return "pytest collected no tests"
    if _UNITTEST_NOTHING_RE.search(text):
        return "no tests were found to run"
    if _MAKE_NO_TARGET_RE.search(text):
        return 'there is no "test" make target'
    return ""


@dataclass
class VerificationRun:
    """One ``run_tests`` call of the agent."""

    command: str
    exit_code: int
    duration: float
    output_tail: str
    timed_out: bool = False
    # Why the command had nothing to test ("" when it did run tests).
    no_tests: str = ""

    @property
    def outcome(self) -> str:
        """``passed``, ``failed``, ``timed_out`` or ``no_tests``."""
        if self.timed_out:
            return "timed_out"
        if self.no_tests:
            return "no_tests"
        return "passed" if self.exit_code == 0 else "failed"

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    def as_record(self, redact: Callable[[str], str] = lambda text: text) -> Dict[str, Any]:
        return {
            "command": redact(self.command),
            "exit_code": self.exit_code,
            "passed": self.passed,
            "timed_out": self.timed_out,
            "duration": round(self.duration, 1),
            "outcome": self.outcome,
            "note": self.no_tests,
        }


def tests_status_of(runs: List[VerificationRun]) -> str:
    """``passed`` / ``failed`` (the last run that really tested something),
    ``no_tests`` (test commands were run, but there are no tests) or ``not_run``."""
    if not runs:
        return "not_run"
    real = [run for run in runs if run.outcome != "no_tests"]
    if not real:
        return "no_tests"
    return "passed" if real[-1].passed else "failed"


# --------------------------------------------------------------------------- #
# Documentation-only changes
# --------------------------------------------------------------------------- #
_DOC_EXTENSIONS = {".md", ".markdown", ".rst", ".adoc", ".asciidoc"}
_DOC_NAMES = {
    "readme",
    "license",
    "licence",
    "copying",
    "notice",
    "authors",
    "contributors",
    "maintainers",
    "changelog",
    "changes",
    "history",
    "contributing",
    "code_of_conduct",
    "security",
    "roadmap",
}


def is_documentation_path(path: str) -> bool:
    """True for prose that no test run could exercise: Markdown/reST/AsciiDoc
    files and the usual top-level documents (README, LICENSE, CHANGELOG ...).

    Deliberately narrow: ``.mdx`` contains code, and ``.txt`` is only counted
    for well-known document names (``requirements.txt`` is not documentation).
    """
    name = posixpath.basename(path).lower()
    stem, extension = posixpath.splitext(name)
    if extension in _DOC_EXTENSIONS:
        return True
    return extension in ("", ".txt") and stem in _DOC_NAMES


def is_docs_only(paths: Iterable[str]) -> bool:
    listed = list(paths)
    return bool(listed) and all(is_documentation_path(path) for path in listed)
