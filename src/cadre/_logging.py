"""Logging utilities for cadre."""

import sys
import warnings
from collections.abc import Generator
from contextlib import contextmanager


class Colors:
    """ANSI color codes for terminal output."""

    BLUE: str = "\033[94m"
    GREEN: str = "\033[92m"
    YELLOW: str = "\033[93m"
    RED: str = "\033[91m"
    CYAN: str = "\033[96m"
    RESET: str = "\033[0m"
    BOLD: str = "\033[1m"
    DIM: str = "\033[2m"

    @classmethod
    def is_tty(cls) -> bool:
        """Check if stdout is a TTY (supports colors)."""
        return sys.stdout.isatty()

    @classmethod
    def disable(cls) -> None:
        """Disable all colors."""
        cls.BLUE = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.RED = ""
        cls.CYAN = ""
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""


# Disable colors if not in a TTY
if not Colors.is_tty():
    Colors.disable()


def info(message: str) -> None:
    """Print an informational message."""
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {message}")


def warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {message}")


def error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {message}", file=sys.stderr)


@contextmanager
def suppress_runtime_warnings() -> Generator[None, None, None]:
    """Context manager to suppress specific runtime warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in log")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
        yield
