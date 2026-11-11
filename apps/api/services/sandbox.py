import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("sandbox_service")
logger.setLevel(logging.INFO)

class SandboxExecutionResult:
    def __init__(self, exit_code: int, stdout: str, stderr: str, files_created: Optional[list] = None):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.files_created = files_created or []

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "files_created": self.files_created,
            "success": self.success,
        }


class E2BSandboxManager:
    """
    Manages isolated Linux microVM environments for executing code, running tests,
    and making repository edits using the E2B Code Interpreter SDK.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self._sandbox = None

    def start(self):
        """Spawns an isolated microVM sandbox."""
        if not self.api_key:
            logger.warning("E2B_API_KEY not set. Running in local fallback simulation mode.")
            return None

        try:
            from e2b_code_interpreter import Sandbox
            self._sandbox = Sandbox(api_key=self.api_key)
            logger.info("Spawned E2B sandbox microVM successfully.")
            return self._sandbox
        except Exception as exc:
            logger.error(f"Failed to initialize E2B Sandbox: {exc}")
            raise

    def execute_command(self, command: str) -> SandboxExecutionResult:
        """Runs a bash command in the sandbox microVM."""
        if self._sandbox:
            try:
                # E2B commands execution
                proc = self._sandbox.commands.run(command)
                return SandboxExecutionResult(
                    exit_code=proc.exit_code or 0,
                    stdout=proc.stdout or "",
                    stderr=proc.stderr or "",
                )
            except Exception as e:
                logger.error(f"Sandbox command execution failed: {e}")
                return SandboxExecutionResult(exit_code=1, stdout="", stderr=str(e))
        else:
            # Fallback mock execution when running in development/tests without live API key
            logger.info(f"[SIMULATED SANDBOX CMD]: {command}")
            return SandboxExecutionResult(
                exit_code=0,
                stdout=f"Simulated execution output for: {command}\nProcess exited cleanly.",
                stderr="",
            )

    def read_file(self, filepath: str) -> str:
        """Reads the content of a file within the sandbox."""
        if self._sandbox:
            try:
                return self._sandbox.files.read(filepath)
            except Exception as e:
                logger.error(f"Failed to read file {filepath} in sandbox: {e}")
                raise
        else:
            logger.info(f"[SIMULATED SANDBOX READ]: {filepath}")
            return f"# Simulated content for {filepath}\nprint('Hello from sandbox')\n"

    def write_file(self, filepath: str, content: str) -> bool:
        """Writes content to a file in the sandbox microVM."""
        if self._sandbox:
            try:
                self._sandbox.files.write(filepath, content)
                return True
            except Exception as e:
                logger.error(f"Failed to write file {filepath} in sandbox: {e}")
                return False
        else:
            logger.info(f"[SIMULATED SANDBOX WRITE]: {filepath} ({len(content)} bytes)")
            return True

    def run_tests(self, test_command: str = "pytest") -> SandboxExecutionResult:
        """Runs test suites inside the sandbox."""
        logger.info(f"Running test suite: {test_command}")
        return self.execute_command(test_command)

    def close(self):
        """Terminates the sandbox microVM."""
        if self._sandbox:
            try:
                self._sandbox.kill()
                logger.info("Sandbox microVM closed.")
            except Exception as e:
                logger.warning(f"Error closing sandbox: {e}")
            finally:
                self._sandbox = None
