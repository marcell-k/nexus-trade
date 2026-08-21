from __future__ import annotations

import logging
import re
import sys

_STRATEGY_PREFIX = re.compile(r"^\S+\s*:\s*")


def _rgb(hex_color: str) -> str:
    """Return a 24-bit truecolor ANSI foreground escape for a '#rrggbb' string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"\033[38;2;{r};{g};{b}m"


class _Kanagawa:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    BLACK = _rgb("0d0c0c")
    RED = _rgb("c4746e")
    GREEN = _rgb("8a9a7b")
    YELLOW = _rgb("c4b28a")
    BLUE = _rgb("8ba4b0")
    MAGENTA = _rgb("a292a3")
    CYAN = _rgb("8ea4a2")
    FG = _rgb("c8c093")
    GREY = _rgb("a6a69c")
    BRIGHT_RED = _rgb("e46876")
    BRIGHT_GREEN = _rgb("87a987")
    BRIGHT_YELLOW = _rgb("e6c384")
    BRIGHT_BLUE = _rgb("7fb4ca")
    BRIGHT_MAGENTA = _rgb("938aa9")
    BRIGHT_CYAN = _rgb("7aa89f")
    BRIGHT_FG = _rgb("c5c9c5")

    FILL_STYLE = BOLD + BRIGHT_BLUE
    CLOSE_STYLE = BOLD + BRIGHT_YELLOW


_STRATEGY_PREFIX = re.compile(r"^\S+\s*:\s*")

_TAG_COLORS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^FillLog$"), _Kanagawa.FILL_STYLE),
    (re.compile(r"^CloseLog$"), _Kanagawa.CLOSE_STYLE),
    (re.compile(r"^PartClose"), _Kanagawa.BRIGHT_MAGENTA),
    (re.compile(r"^(Entry|EntryOK|BrktOK|BrktRange)$"), _Kanagawa.GREEN),
    (re.compile(r"^(Closed|ExternalClose|PosClosedDetect)$"), _Kanagawa.CYAN),
    (re.compile(r"^(OCOCancel|BracketOppCancel|ShutdownOrd)$"), _Kanagawa.BLUE),
    (re.compile(r"^Fill$"), _Kanagawa.GREEN),
    (re.compile(r"^Init$"), _Kanagawa.GREY),
    (re.compile(r".*(Fail|Crash)$"), _Kanagawa.BRIGHT_RED),
    (re.compile(r".*Reject$"), _Kanagawa.BRIGHT_YELLOW),
    (re.compile(r"^HB$"), _Kanagawa.GREY),
)

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: _Kanagawa.GREY,
    logging.WARNING: _Kanagawa.YELLOW,
    logging.ERROR: _Kanagawa.RED,
    logging.CRITICAL: _Kanagawa.BRIGHT_RED + _Kanagawa.BOLD,
}


def _resolve_color(level: int, message: str) -> str:
    stripped = _STRATEGY_PREFIX.sub("", message, count=1)
    tag = stripped.split(" ", 1)[0] if stripped else ""
    for pattern, color in _TAG_COLORS:
        if pattern.match(tag):
            return color
    return _LEVEL_COLORS.get(level, "")


class ColoredFormatter(logging.Formatter):
    """Colors the whole formatted line based on log tag / level. Console only."""

    def __init__(self, fmt: str, *, use_color: bool | None = None) -> None:
        super().__init__(fmt)
        self.use_color: bool = sys.stdout.isatty() if use_color is None else use_color

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if not self.use_color:
            return formatted
        color = _resolve_color(record.levelno, record.getMessage())
        return f"{color}{formatted}{_Kanagawa.RESET}" if color else formatted


def enable_ansi() -> None:
    """Enable VT100/truecolor processing on legacy Windows consoles (no-op elsewhere)."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except OSError:
        pass
