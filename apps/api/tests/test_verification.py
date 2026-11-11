"""What a test command's result means (see ``services/verification.py``).

The outputs below are what the tools really print; the first one is the run
that prompted this module: ``npm test`` in a repository whose ``test`` script is
npm's placeholder was reported as "the last test run failed".
"""
from __future__ import annotations

import pytest

from apps.api.services import verification
from apps.api.services.verification import (
    VerificationRun,
    is_docs_only,
    is_documentation_path,
    no_tests_reason,
)

# Not imported by name: pytest would collect a function called tests_... as a test.
status_of = verification.tests_status_of

NPM_PLACEHOLDER = """
> kensheets@1.0.0 test
> echo "Error: no test specified" && exit 1

Error: no test specified
"""

NPM_MISSING_SCRIPT = """npm error Missing script: "test"
npm error
npm error To see a list of scripts, run:
npm error   npm run
npm error A complete log of this run can be found in: /home/user/.npm/_logs/2026-11-11T15_00_00_000Z-debug-0.log
"""

NODE_TEST_EMPTY = """TAP version 13
1..0
# tests 0
# suites 0
# pass 0
# fail 0
# cancelled 0
# skipped 0
# todo 0
# duration_ms 13.767701
"""

UNITTEST_EMPTY = """
----------------------------------------------------------------------
Ran 0 tests in 0.000s

OK
"""

JEST_EMPTY = """No tests found, exiting with code 1
Run with `--passWithNoTests` to exit with code 0
In /home/user/repo
  12 files checked.
  testMatch: **/__tests__/**/*.[jt]s?(x), **/?(*.)+(spec|test).[jt]s?(x) - 0 matches
  testPathIgnorePatterns: /node_modules/ - 12 matches
  testRegex:  - 0 matches
Pattern:  - 0 matches
"""

GO_NO_TEST_FILES = "?   \texample.com/app\t[no test files]\n?   \texample.com/app/util\t[no test files]\n"
GO_MIXED = "?   \texample.com/app\t[no test files]\nok  \texample.com/app/util\t0.003s\n"
GO_FAILING = "?   \texample.com/app\t[no test files]\n--- FAIL: TestAdd (0.00s)\nFAIL\texample.com/app/util\t0.003s\n"

CARGO_EMPTY = """   Compiling app v0.1.0 (/home/user/repo)
    Finished test [unoptimized + debuginfo] target(s) in 0.52s
     Running unittests src/main.rs (target/debug/deps/app-0a1b2c3d4e5f6071)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
"""
CARGO_REAL = CARGO_EMPTY.replace("running 0 tests", "running 3 tests").replace("0 passed", "3 passed")


@pytest.mark.parametrize(
    "exit_code, output, reason",
    [
        (1, NPM_PLACEHOLDER, "the \"test\" script in package.json is npm's placeholder"),
        (1, NPM_MISSING_SCRIPT, "package.json has no \"test\" script"),
        (1, 'error Command "test" not found.\n', "package.json has no \"test\" script"),
        (1, " ERR_PNPM_NO_SCRIPT  Missing script: test\n", "package.json has no \"test\" script"),
        (1, JEST_EMPTY, "the test runner found no tests"),
        (1, "No test files found, exiting with code 1\n\ninclude: **/*.{test,spec}.?(c|m)[jt]s?(x)\n", "the test runner found no tests"),
        (1, "Error: No test files found\n", "the test runner found no tests"),
        (0, NODE_TEST_EMPTY, "the test runner found no tests"),  # exit 0, but nothing was tested
        (5, "\nno tests ran in 0.00s\n", "pytest collected no tests"),
        (5, "collected 0 items\n\n============ no tests ran in 0.01s ============\n", "pytest collected no tests"),
        (0, UNITTEST_EMPTY, "no tests were found to run"),  # exit 0, but nothing was tested
        (2, "make: *** No rule to make target 'test'.  Stop.\n", "there is no \"test\" make target"),
        (0, GO_NO_TEST_FILES, "the Go packages have no test files"),
        (0, CARGO_EMPTY, "cargo found no tests to run"),
    ],
)
def test_commands_with_nothing_to_test_are_recognised(exit_code, output, reason):
    assert no_tests_reason(exit_code, output) == reason


@pytest.mark.parametrize(
    "exit_code, output",
    [
        (1, "AssertionError: 5 != -1\n"),
        (1, "FAILED tests/test_calc.py::test_add - assert -1 == 5\n1 failed in 0.02s\n"),
        (0, "1 passed in 0.01s\n"),
        (0, ""),
        # A different script is missing: that is a wrong command, not "no tests".
        (1, 'npm error Missing script: "test:unit"\n'),
        # pytest was pointed at a path that does not exist (usage error, exit 4).
        (4, "ERROR: file or directory not found: tests\n\nno tests ran in 0.00s\n"),
        # Real results next to a package without tests.
        (0, GO_MIXED),
        (1, GO_FAILING),
        (0, CARGO_REAL),
        # The placeholder of one workspace package inside a long log with real
        # results: never hide those behind "no tests".
        (1, NPM_PLACEHOLDER + "\n".join(f"  ✕ renders row {n}" for n in range(60))),
        # The same in a short log: one package has the placeholder (or no script),
        # another one ran real tests. The exit code decides.
        (1, NPM_PLACEHOLDER + "\n> web@1.0.0 test\n> mocha\n\n  2 passing (5ms)\n  1 failing\n"),
        (1, NPM_PLACEHOLDER + "\nFAIL src/sum.test.js\nTests:       1 failed, 2 passed, 3 total\n"),
        (1, NPM_PLACEHOLDER + "\n=========== 1 failed, 2 passed in 0.12s ===========\n"),
        (1, NPM_MISSING_SCRIPT + "TAP version 13\nok 1 - adds\nnot ok 2 - subtracts\n"),
        (1, NPM_MISSING_SCRIPT + "Traceback (most recent call last):\n  File \"t.py\", line 1\nAssertionError\n"),
    ],
)
def test_real_results_are_never_mistaken_for_no_tests(exit_code, output):
    assert no_tests_reason(exit_code, output) == ""


def run(exit_code: int, *, no_tests: str = "", timed_out: bool = False) -> VerificationRun:
    return VerificationRun(command="t", exit_code=exit_code, duration=0.1, output_tail="", timed_out=timed_out, no_tests=no_tests)


def test_outcome_of_a_single_run():
    assert (run(0).outcome, run(0).passed) == ("passed", True)
    assert (run(1).outcome, run(1).passed) == ("failed", False)
    assert (run(124, timed_out=True).outcome, run(124, timed_out=True).passed) == ("timed_out", False)
    # Neither a pass nor a failure - whatever the exit code was.
    for exit_code in (0, 1, 5):
        nothing = run(exit_code, no_tests="pytest collected no tests")
        assert (nothing.outcome, nothing.passed) == ("no_tests", False)
    assert run(1, no_tests="x").as_record() == {
        "command": "t",
        "exit_code": 1,
        "passed": False,
        "timed_out": False,
        "duration": 0.1,
        "outcome": "no_tests",
        "note": "x",
    }


def test_status_follows_the_last_run_that_tested_something():
    nothing = run(1, no_tests="npm's placeholder")
    assert status_of([]) == "not_run"
    assert status_of([nothing]) == "no_tests"
    assert status_of([nothing, nothing]) == "no_tests"
    assert status_of([run(0)]) == "passed"
    assert status_of([run(1), run(0)]) == "passed"
    assert status_of([run(0), run(1)]) == "failed"
    assert status_of([run(124, timed_out=True)]) == "failed"
    # A run with nothing to test neither rescues a failure nor spoils a pass.
    assert status_of([run(1), nothing]) == "failed"
    assert status_of([run(0), nothing]) == "passed"


@pytest.mark.parametrize(
    "path",
    ["spec.md", "README.md", "docs/guide/intro.rst", "LICENSE", "LICENSE.txt", "CHANGELOG", "docs/Design.MD", ".github/CONTRIBUTING.md", "notes.adoc"],
)
def test_documentation_paths(path):
    assert is_documentation_path(path)


@pytest.mark.parametrize(
    "path",
    ["requirements.txt", "src/app.js", "package.json", "docs/conf.py", "page.mdx", "robots.txt", "Makefile", "docs/diagram.png", "README.py"],
)
def test_paths_a_test_run_could_exercise(path):
    assert not is_documentation_path(path)


def test_docs_only_needs_every_changed_file_to_be_documentation():
    assert is_docs_only(["spec.md"]) and is_docs_only(["README.md", "docs/a.rst", "LICENSE"])
    assert not is_docs_only(["README.md", "src/app.js"])
    assert not is_docs_only([])  # no change is not a documentation change
